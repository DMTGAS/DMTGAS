import os
import sys
import argparse
import glob
import torch
import logging
import time
import torch.nn.functional as F
from tqdm import tqdm
from util.logging_util import init_logger
from util import cosmos_util, data_utils, bl_utils, utils as ut, plotpoint
from model_search import MTLAGL
from architect import Architect
from sklearn.metrics import roc_auc_score
from util.hv import HyperVolume



sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np

parser = argparse.ArgumentParser("mtlnas-train-search")
parser.add_argument("--dataset-name", type=str,
                    help="Name of the dataset from the TUDortmund")
parser.add_argument("--data-folder", type=str, default="",
                    help="Path to the folder where data will be stored (default is working directory).")
parser.add_argument("--output-folder", type=str, default=None,
                    help="Path to the output folder for saving the model (optional).")
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument("--tasks", type=str, default="gc,nc,lp",
                    help="Tasks to be performed (default is 'gc,nc,lp').")
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument("--batch-size", type=int, default=16,
                    help="Number of tasks in a mini-batch of tasks (default: 16).")
parser.add_argument("--embedding-dim", type=int, default=32,
                    help="Node embedding dimension (default: 16).")
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--fix_last', type=bool, default=False, help='fix last layer in design architectures.')
parser.add_argument("--weight-pareto", action="store_true", default=False, help="Pareto the multitask loss function.")
parser.add_argument("--epochs", type=int, default=10,
                    help="Number of training epochs (default: 10).")
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument("--normalization_type", type=str, default='l2',
                        help="type of normalizing all gradients (default: l2).")
parser.add_argument("--alpha", type=float, default=1.2,
                        help="parameter of generating the reference.")
parser.add_argument("--lamda", type=float, default=2,
                        help="parameter of convergence and diversity.")
parser.add_argument("--n_test_rays", type=int, default=50,
                        help="Number of test preference vectors for Pareto front generating methods.")
parser.add_argument("--reference_point", type=list, default=[2, 2],
                        help="Reference point for hyper-volume calculation.")

def run(args):
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.save = 'logs/search-{}'.format(args.save)
    print(args.save)
    if not os.path.exists(args.save):
        ut.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)
    dataset, train_val_test_ratio = data_utils.get_graph_dataset(args.dataset_name,
                                                                 destination_dir=args.data_folder)
    dataset = dataset.shuffle()
    train_dataloader, val_dataloader, test_dataloader = data_utils.get_dataloaders(dataset, args.batch_size,
                                                                                   "multi",
                                                                                   train_val_test_ratio=[0.7, 0.1,
                                                                                                         0.2],
                                                                                   num_workers=1, shuffle_train=True)
    data = [train_dataloader, val_dataloader, test_dataloader]
    output_gc_dim = dataset.num_classes
    output_nc_dim = dataset[0].node_y.size(1)
    # hidden_size = 32
    model = MTLAGL(args.tasks,
                   dataset.num_node_features,
                   args.embedding_dim,
                   output_gc_dim,
                   output_nc_dim,
                   args.embedding_dim,
                   epsilon=args.epsilon, with_conv_linear=args.with_conv_linear, args=args
                   )
    model = model.to(device)
    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

    genotype = train(model, data, args)
    if args.weight_pareto:
        test_result, test_hv = infercosmos(model, test_dataloader, args, False) # test data
        valid_result, valid_hv = infercosmos(model, val_dataloader, args, True) # val data
        test_hv_point = bl_utils.log_task_hv_and_point(test_hv)
        valid_hv_point = bl_utils.log_task_hv_and_point(valid_hv)
        logging.info(test_hv_point)
        logging.info(valid_hv_point)
    else:
        test_result = infer(model, test_dataloader, args, False) # test data
        valid_result = infer(model, val_dataloader, args, True) # val data
    test_acc = bl_utils.log_task_accs_and_losses(test_result)
    valid_acc = bl_utils.log_task_accs_and_losses(valid_result)
    logging.info(test_acc)
    logging.info(valid_acc)
    return genotype

    # res = 'seed={},genotype={},saved_dir={}'.format(args.seed, genotype, args.save)
    # filename = 'exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (
    # args.dataset_name, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    # fw = open(filename, 'w+')
    # fw.write('\t'.join(res))
    # fw.close()
    # print('searched res for {} saved in {}'.format(args.dataset_name, filename))


def train(model, dataloader, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
                                                           eta_min=args.learning_rate_min)
    architect = Architect(model, args)  # send model to compute validation loss
    search_cost = 0

    for epoch in range(args.epochs):
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
        if args.weight_pareto:
            train_result = pareto_train_search(args, dataloader, model, architect, optimizer, lr, epoch)
        else:
            train_result = train_search(args, dataloader, model, architect, optimizer, lr, epoch)
        scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)
        if epoch % int(args.epochs/10) == 0:
            train_acc = bl_utils.log_task_accs_and_losses(train_result)
            logging.info('epoch=%s, explore_num=%s' + train_acc, epoch, model.explore_num)
            if args.weight_pareto:
                test_result, test_hv = infercosmos(model, dataloader[2], args, False, None, epoch)  # test data
                valid_result, valid_hv = infercosmos(model, dataloader[1], args, True, None, epoch)  # val data
                test_hv_point = bl_utils.log_task_hv_and_point(test_hv)
                valid_hv_point = bl_utils.log_task_hv_and_point(valid_hv)
                logging.info(test_hv_point)
                logging.info(valid_hv_point)
            else:
                test_result = infer(model, dataloader[2], args, False)  # test data
                valid_result = infer(model, dataloader[1], args, True)  # val data
        ut.save(model, os.path.join(args.save, 'weights.pt'))
    logging.info('The search process costs %.2fs', search_cost)
    return genotype

def pareto_train_search(args, dataloader, model, architect, optimizer, lr, epoch):
    model.train()
    epoch_stats = bl_utils.EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader[0], desc="Batch")):
        valid_search = next(iter(dataloader[1]))
        _, valid_search_batch, _ = data_utils.multi_task_train_test_split(valid_search, True, tasks=args.tasks)
        valid_search_batch = valid_search_batch[0]
        valid_search_batch = valid_search_batch.to(device)
        _, train_batch, _ = data_utils.multi_task_train_test_split(batch, True, tasks=args.tasks)
        train_batch = train_batch[0]
        train_batch = train_batch.to(device)

        # step 1: sample alphas
        weight = {}
        sol = []
        if isinstance(args.alpha, list):
            sol = torch.from_numpy(
                np.random.dirichlet(args.alpha, 1).astype(np.float32).flatten()
            ).cuda()
        elif args.alpha > 0:
            sol = torch.from_numpy(
                np.random.dirichlet([args.alpha for _ in range(len(args.tasks))], 1).astype(
                    np.float32).flatten()
            ).cuda()
        else:
            raise ValueError(f"Unknown value for alpha: {args.alpha}, expecting list or float.")

        for i, t in enumerate(args.tasks):
            weight[t] = float(sol[i])
        print('\nweight_val:')
        print(weight)

        # update architect
        architect.step(train_batch, valid_search_batch, lr, optimizer, args.unrolled, weight)
        # architect.step_MGDA(train_batch, valid_search_batch, lr, optimizer, args.unrolled)

        # Dict to tensor
        ww = []
        for k in weight.items():
            ww.append(k[1])
        ww = torch.tensor(ww).cuda()

        # optimization step
        losses = {}
        optimizer.zero_grad()
        # Forward pass
        gc_train_logit, nc_train_logit, lp_train_logit = model(train_batch, ww)
        # Evaluate Loss and Accuracy
        # GC
        gc_loss = nc_loss = lp_loss = 0
        for i, t in enumerate(args.tasks):
            if "gc" == t:
                losses[t] = F.cross_entropy(gc_train_logit, train_batch.y)
                with torch.no_grad():
                    gc_acc = ut.get_accuracy(gc_train_logit, train_batch.y)
                epoch_stats.update("gc", train_batch, losses[t], gc_acc, True)
            # NC
            if "nc" == t:
                node_labels = train_batch.node_y.argmax(1)
                train_mask = train_batch.train_mask.squeeze()
                losses[t] = F.cross_entropy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
                with torch.no_grad():
                    nc_acc = ut.get_accuracy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
                epoch_stats.update("nc", train_batch, losses[t], nc_acc, True)
            # LP
            if "lp" == t:
                train_link_labels = data_utils.get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
                losses[t] = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
                with torch.no_grad():
                    train_labels = train_link_labels.detach().cpu().numpy()
                    train_predictions = lp_train_logit.detach().cpu().numpy()
                    lp_acc = roc_auc_score(train_labels, train_predictions.squeeze())
                epoch_stats.update("lp", train_batch, losses[t], lp_acc, True)
            if i > 0:
                loss = loss + weight[t] * losses[t]
            else:
                loss = weight[t] * losses[t]
        # dict to tensor
        lossess = []
        for k in losses.items():
            lossess.append(k[1])
        lossess = torch.tensor(lossess).cuda()

        cossim = torch.nn.functional.cosine_similarity(lossess, ww, dim=0)
        loss -= args.lamda * cossim

        # Record
        print('\ncossim:')
        print(cossim)
        tasks_epoch_stats = epoch_stats.get_average_stats()
        bl_utils.print_train_epoch_stats(epoch, tasks_epoch_stats)

        # Backprop and update parameters
        loss.backward()
        optimizer.step()
    tasks_epoch_stats = epoch_stats.get_average_stats()
    bl_utils.print_train_epoch_stats(epoch, tasks_epoch_stats)
    return tasks_epoch_stats

def train_search(args, dataloader, model, architect, optimizer, lr, epoch):
    model.train()
    epoch_stats = bl_utils.EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader[0], desc="Batch")):
        valid_search = next(iter(dataloader[1]))
        _, valid_search_batch, _ = data_utils.multi_task_train_test_split(valid_search, True, tasks=args.tasks)
        valid_search_batch = valid_search_batch[0]
        valid_search_batch = valid_search_batch.to(device)
        _, train_batch, _ = data_utils.multi_task_train_test_split(batch, True, tasks=args.tasks)
        train_batch = train_batch[0]
        train_batch = train_batch.to(device)
        # update architect
        architect.step(train_batch, valid_search_batch, lr, optimizer, args.unrolled, None)

        optimizer.zero_grad()
        # Forward pass
        gc_train_logit, nc_train_logit, lp_train_logit = model(train_batch)
        # Evaluate Loss and Accuracy
        # GC
        gc_loss = nc_loss = lp_loss = 0
        if "gc" in args.tasks:
            gc_loss = F.cross_entropy(gc_train_logit, train_batch.y)
            with torch.no_grad():
                gc_acc = ut.get_accuracy(gc_train_logit, train_batch.y)
            epoch_stats.update("gc", train_batch, gc_loss, gc_acc, True)
        # NC
        if "nc" in args.tasks:
            node_labels = train_batch.node_y.argmax(1)
            train_mask = train_batch.train_mask.squeeze()
            nc_loss = F.cross_entropy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
            with torch.no_grad():
                nc_acc = ut.get_accuracy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
            epoch_stats.update("nc", train_batch, nc_loss, nc_acc, True)
        # LP
        if "lp" in args.tasks:
            train_link_labels = data_utils.get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
            lp_loss = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
            with torch.no_grad():
                train_labels = train_link_labels.detach().cpu().numpy()
                train_predictions = lp_train_logit.detach().cpu().numpy()
                lp_acc = roc_auc_score(train_labels, train_predictions.squeeze())
            epoch_stats.update("lp", train_batch, lp_loss, lp_acc, True)

        loss = gc_loss + nc_loss + lp_loss

        # Backprop and update parameters
        loss.backward()
        optimizer.step()
    tasks_epoch_stats = epoch_stats.get_average_stats()
    bl_utils.print_train_epoch_stats(epoch, tasks_epoch_stats)
    return tasks_epoch_stats

def infercosmos(model, dataloader, args, is_vaild, test_rays=None, epoch=0):
    model.eval()
    epoch_stats = bl_utils.EpochStats(args)
    score_values = np.array([])
    if test_rays is None:
        test_rays = cosmos_util.circle_points(args.n_test_rays, dim=len(args.tasks))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
        _, test_batch, _ = data_utils.multi_task_train_test_split(batch, False, tasks=args.tasks)
        test_batch = test_batch[0]
        test_batch = test_batch.to(device)
        with torch.no_grad():
            if "gc" in args.tasks:
                epoch_stats.updatenum("gc", test_batch)
            if "nc" in args.tasks:
                num_test_nodes = epoch_stats.updatenum("nc", test_batch)
            else:
                num_test_nodes = None
            if "lp" in args.tasks:
                num_lp_edges = epoch_stats.updatenum("lp", test_batch)
            else:
                num_lp_edges = None

            s = []
            for ray_index, ray in enumerate(test_rays):
                ray = torch.from_numpy(ray.astype(np.float32)).cuda()
                ray /= ray.sum()
                gc_test_logit, nc_test_logit, lp_test_logit = model(test_batch, ray)
                ttmp = []
                # GC
                if "gc" in args.tasks:
                    gc_loss = F.cross_entropy(gc_test_logit, test_batch.y)
                    with torch.no_grad():
                        gc_acc = ut.get_accuracy(gc_test_logit, test_batch.y)
                    epoch_stats.updatecosmos("gc", test_batch, gc_loss, gc_acc, ray_index, num_test_nodes, num_lp_edges)
                    ttmp.append(gc_loss)
                #NC
                if "nc" in args.tasks:
                    node_labels = test_batch.node_y.argmax(1)
                    train_mask = test_batch.train_mask.squeeze()
                    test_mask = (train_mask==0).float()
                    nc_loss = F.cross_entropy(nc_test_logit[test_mask==1], node_labels[test_mask==1])
                    with torch.no_grad():
                        nc_acc = ut.get_accuracy(nc_test_logit[test_mask==1], node_labels[test_mask==1])
                    epoch_stats.updatecosmos("nc", test_batch, nc_loss, nc_acc, ray_index, num_test_nodes, num_lp_edges)
                    ttmp.append(nc_loss)
                # LP
                if "lp" in args.tasks:
                    test_link_labels = data_utils.get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
                    lp_loss = F.binary_cross_entropy_with_logits(lp_test_logit.squeeze(), test_link_labels)
                    with torch.no_grad():
                        test_labels = test_link_labels.detach().cpu().numpy()
                        test_predictions = lp_test_logit.detach().cpu().numpy()
                        lp_acc = roc_auc_score(test_labels, test_predictions.squeeze())
                    epoch_stats.updatecosmos("lp", test_batch, lp_loss, lp_acc, ray_index, num_test_nodes, num_lp_edges)
                    ttmp.append(lp_loss)
                s.append(ttmp)
            if score_values.size == 0:
                score_values = np.array(s)
            else:
                score_values += np.array(s)

    score_values /= len(dataloader)
    hv = HyperVolume(args.reference_point)

    # Computing hyper-volume for many objectives is expensive
    volume = hv.compute(score_values) if score_values.shape[1] < 5 else -1

    if is_vaild:
        # pareto_front = cosmos_util.ParetoFront(args.tasks, args.output_folder,  "val_{:03d}".format(epoch))
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        string = os.path.join(args.save, 'pf')
        if not os.path.exists(string):
            os.mkdir(string)
        string = os.path.join(string, "val_{:03d}".format(epoch))
        pareto_front = plotpoint.PointScatter(len(args.tasks), legend=args.tasks, saveName=string)
        ut.save_point(score_values, string)
    else:
        # pareto_front = cosmos_util.ParetoFront(args.tasks, args.output_folder, "test_{:03d}".format(epoch))
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        string = os.path.join(args.save, 'pf')
        if not os.path.exists(string):
            os.mkdir(string)
        string = os.path.join(string, "test_{:03d}".format(epoch))
        pareto_front = plotpoint.PointScatter(len(args.tasks), legend=args.tasks, saveName=string)
        ut.save_point(score_values, string)

    # pareto_front.append(score_values)
    # pareto_front.plot()
    pareto_front.add(score_values)
    pareto_front.add(score_values)
    pareto_front.draw()
    pareto_front.show()
    pareto_front.close()

    # if len(args.tasks) == 2:
    #     if is_vaild:
    #         pareto_front = cosmos_util.ParetoFront(['f1', 'f2'], args.save, 'val')
    #     else:
    #         pareto_front = cosmos_util.ParetoFront(['f1', 'f2'], args.save, 'test')
    #     pareto_front.append(score_values)
    #     pareto_front.plot()

    result = {
        "scores": score_values.tolist(),
        "hv": volume,
    }
    print("hv:")
    print(volume)

    tasks_test_stats = epoch_stats.get_average_stats_cosmos(args)
    if is_vaild:
        bl_utils.print_val_stats(tasks_test_stats)
    else:
        bl_utils.print_test_stats(tasks_test_stats)
    return tasks_test_stats, result

def infer(model, dataloader, args, is_vaild):
    model.eval()
    epoch_stats = bl_utils.EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
        _, test_batch, _ = data_utils.multi_task_train_test_split(batch, False, tasks=args.tasks)
        test_batch = test_batch[0]
        test_batch = test_batch.to(device)
        with torch.no_grad():
            gc_test_logit, nc_test_logit, lp_test_logit = model(test_batch)
            # GC
            if "gc" in args.tasks:
                gc_loss = F.cross_entropy(gc_test_logit, test_batch.y)
                with torch.no_grad():
                    gc_acc = ut.get_accuracy(gc_test_logit, test_batch.y)
                epoch_stats.update("gc", test_batch, gc_loss, gc_acc, False)
            #NC
            if "nc" in args.tasks:
                node_labels = test_batch.node_y.argmax(1)
                train_mask = test_batch.train_mask.squeeze()
                test_mask = (train_mask==0).float()
                nc_loss = F.cross_entropy(nc_test_logit[test_mask==1], node_labels[test_mask==1])
                with torch.no_grad():
                    nc_acc = ut.get_accuracy(nc_test_logit[test_mask==1], node_labels[test_mask==1])
                epoch_stats.update("nc", test_batch, nc_loss, nc_acc, False)
            # LP
            if "lp" in args.tasks:
                test_link_labels = data_utils.get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
                lp_loss = F.binary_cross_entropy_with_logits(lp_test_logit.squeeze(), test_link_labels)
                with torch.no_grad():
                    test_labels = test_link_labels.detach().cpu().numpy()
                    test_predictions = lp_test_logit.detach().cpu().numpy()
                    lp_acc = roc_auc_score(test_labels, test_predictions.squeeze())
                epoch_stats.update("lp", test_batch, lp_loss, lp_acc, False)

    tasks_test_stats = epoch_stats.get_average_stats()
    if is_vaild:
        bl_utils.print_val_stats(tasks_test_stats)
    else:
        bl_utils.print_test_stats(tasks_test_stats)
    return tasks_test_stats

if __name__ == "__main__":
    args = parser.parse_args()
    args.save = 'test'
    args.dataset_name = 'PROTEINS'
    args.tasks = ["gc", "nc", "lp"]
    args.reference_point = [2, 2, 2]
    args.weight_pareto = True
    args.gpu = 2
    args.batch_size = 256
    res = []
    args.epochs = 100
    alpha = [0.1, 0.5, 1, 1.2, 10]
    lamda = [0.001, 0.01, 2, 3, 8]
    args.n_test_rays = 25
    for i in range(5):
        print('searched {}-th for {}...'.format(i + 1, args.dataset_name))
        args.save = '{}-th-{}-'.format(i + 1, args.dataset_name)
        args.alpha = alpha[i]
        args.lamda = lamda[i]
        print('alpha:{}, lamda:{}'.format(args.alpha, args.lamda))
        for k, j in enumerate(args.tasks):
            args.save += j
            if k != len(args.tasks)-1:
                args.save += '-'
        seed = np.random.randint(0, 10000)
        args.seed = seed
        ut.set_seeds(seed=args.seed)
        ut.print_arguments(args)
        genotype = run(args)
        res.append('seed={},genotype={},saved_dir={}'.format(seed, genotype, args.save))
    filename = 'exp_res/%s-searched_res-epoch%s-' % (args.dataset_name, args.epochs)
    for k, j in enumerate(args.tasks):
        filename += j
        if k != len(args.tasks) - 1:
            filename += '-'
    filename += '.txt'
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.dataset_name, filename))


