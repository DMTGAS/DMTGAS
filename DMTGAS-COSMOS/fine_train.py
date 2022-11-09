import os
import sys
import time
import torch.nn.functional as F
import torch
from tqdm import tqdm
import logging
import torch.utils
import torch.backends.cudnn as cudnn
import util.data_utils as data_utils
import util.utils as ut
import util.bl_utils as bl_utils
from sklearn.metrics import roc_auc_score
from model import MTLAGL as Network
from util.hv import HyperVolume
from util import cosmos_util
import numpy as np
from util import plotpoint

def main(exp_args):
    global train_args
    train_args = exp_args
    train_args.save = 'logs/tune-{}-th-{}-'.format(train_args.ind+1, train_args.dataset_name)
    for k, j in enumerate(train_args.tasks):
        train_args.save += j
        if k != len(train_args.tasks) - 1:
            train_args.save += '-'
    if not os.path.exists(train_args.save):
        os.mkdir(train_args.save)

    global device
    device = torch.device('cuda:%d' % train_args.gpu if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)
    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    dataset, train_val_test_ratio = data_utils.get_graph_dataset(train_args.dataset_name,
                                                                 destination_dir=train_args.data_folder)
    dataset = dataset.shuffle()
    train_dataloader, val_dataloader, test_dataloader = data_utils.get_dataloaders(dataset, train_args.batch_size,
                                                                                   "multi",
                                                                                   train_val_test_ratio=[0.7, 0.1,
                                                                                                         0.2],
                                                                                   num_workers=1, shuffle_train=True)
    data = [train_dataloader, val_dataloader, test_dataloader]
    output_gc_dim = dataset.num_classes
    output_nc_dim = dataset[0].node_y.size(1)
    model = Network(train_args.tasks,
                    genotype,
                   dataset.num_node_features,
                   train_args.hidden_size,
                   output_gc_dim,
                   output_nc_dim,
                   hidden_size,
                    num_layers=train_args.num_layers, in_dropout=train_args.in_dropout,
                    out_dropout=train_args.out_dropout, act=train_args.activation,
                    args=train_args
                   )
    model = model.to(device)
    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, ut.count_parameters_in_MB(model), train_args.__dict__)

    best_test_acc_sum, best_val_acc_sum, best_test_acc, best_valid_acc, best_test_hv_all, best_valid_hv_all = train(model, data)

    return best_test_acc_sum, best_val_acc_sum, best_test_acc, best_valid_acc, best_test_hv_all, best_valid_hv_all, train_args

def train(model, dataloader):
    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
            # momentum=args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))

    best_val_acc_sum = best_test_acc_sum = best_valid_hv = 0
    best_valid_hv_all = best_test_hv_all = {
        "scores": [],
        "hv": 0,
    }
    for epoch in range(train_args.epochs):
        if train_args.weight_pareto:
            train_result = train_search_pareto(train_args, dataloader[0], model, optimizer, epoch)
        else:
            train_result = train_search(train_args, dataloader[0], model, optimizer, epoch)
        train_acc_sum, train_acc = bl_utils.task_accs_and_losses(train_result)
        if train_args.cos_lr:
            scheduler.step()
        if epoch % int(train_args.epochs/10) == 0:
        # if epoch % 1 == 0:
            if train_args.weight_pareto:
                valid_result, valid_hv = infercosmos(model, dataloader[1], train_args, True, epoch)  # val data
                test_result, test_hv = infercosmos(model, dataloader[2], train_args, False, epoch)  # test data
                # test_hv_point = bl_utils.log_task_hv_and_point(test_hv)
                # valid_hv_point = bl_utils.log_task_hv_and_point(valid_hv)
                # logging.info(test_hv_point)
                # logging.info(valid_hv_point)
                test_acc_sum, test_acc = bl_utils.task_accs_and_losses(test_result)
                valid_acc_sum, valid_acc = bl_utils.task_accs_and_losses(valid_result)
                logging.info(test_acc_sum)
                logging.info(valid_acc_sum)
                if valid_acc_sum > best_val_acc_sum:
                    best_val_acc_sum = valid_acc_sum
                    best_valid_acc = valid_acc
                    best_valid_hv_all = valid_hv
                    best_test_acc_sum = test_acc_sum
                    best_test_acc = test_acc
                    best_test_hv_all = test_hv
                # if valid_hv["hv"] > best_valid_hv:
                #     best_val_acc_sum = valid_acc_sum
                #     best_valid_acc = valid_acc
                #     best_valid_hv = valid_hv["hv"]
                #     best_valid_hv_all = valid_hv
                #     best_test_acc_sum = test_acc_sum
                #     best_test_acc = test_acc
                #     best_test_hv_all = test_hv

                logging.info('epoch=%s, lr=%s, train_acc=%s, valid_acc=%s, test_acc=%s, val_hv=%s, test_hv=%s', epoch,
                             scheduler.get_lr()[0], train_acc_sum, best_val_acc_sum, best_test_acc_sum, best_valid_hv_all["hv"], best_test_hv_all["hv"])
            else:
                valid_result = infer(model, dataloader[1], train_args, True)  # val data
                test_result = infer(model, dataloader[2], train_args, False)  # test data
                test_acc_sum, test_acc = bl_utils.task_accs_and_losses(test_result)
                valid_acc_sum, valid_acc = bl_utils.task_accs_and_losses(valid_result)
                logging.info(test_acc_sum)
                logging.info(valid_acc_sum)
                if valid_acc_sum > best_val_acc_sum:
                    best_val_acc_sum = valid_acc_sum
                    best_valid_acc = valid_acc
                    best_test_acc_sum = test_acc_sum
                    best_test_acc = test_acc
                logging.info('epoch=%s, lr=%s, train_obj=%s, train_acc=%f, valid_acc=%s, test_acc=%s', epoch,
                             scheduler.get_lr()[0], train_acc_sum, train_acc, best_val_acc_sum, best_test_acc_sum)

        ut.save(model, os.path.join(train_args.save, 'weights.pt'))

    return best_test_acc_sum, best_val_acc_sum, best_test_acc, best_valid_acc, best_test_hv_all, best_valid_hv_all

def train_search_pareto(args, dataloader, model, optimizer, epoch):
    model.train()
    epoch_stats = bl_utils.EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
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

def train_search(args, dataloader, model, optimizer, epoch):
    model.train()
    epoch_stats = bl_utils.EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
        _, train_batch, _ = data_utils.multi_task_train_test_split(batch, True, tasks=args.tasks)
        train_batch = train_batch[0]
        train_batch = train_batch.to(device)

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

def infercosmos(model, dataloader, args, is_vaild, epoch=0, test_rays=None):
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
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        string = os.path.join(args.save, 'pf')
        if not os.path.exists(string):
            os.mkdir(string)
        string = os.path.join(string, "val_{:03d}".format(epoch))
        pareto_front = plotpoint.PointScatter(len(args.tasks), legend=args.tasks, saveName=string)
        ut.save_point(score_values, string)
    else:
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        string = os.path.join(args.save, 'pf')
        if not os.path.exists(string):
            os.mkdir(string)
        string = os.path.join(string, "test_{:03d}".format(epoch))
        pareto_front = plotpoint.PointScatter(len(args.tasks), legend=args.tasks, saveName=string)
        ut.save_point(score_values, string)

    pareto_front.add(score_values)
    pareto_front.add(score_values)
    pareto_front.draw()
    pareto_front.show()
    pareto_front.close()

    # if len(args.tasks) == 2:
    #     if is_vaild:
    #         pareto_front = cosmos_util.ParetoFront(args.tasks, args.save, 'val')
    #     else:
    #         pareto_front = cosmos_util.ParetoFront(args.tasks, args.save, 'test')
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

if __name__ == '__main__':
    main()