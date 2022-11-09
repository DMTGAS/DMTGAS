import os
import pprint
from datetime import datetime
import time
import ast
import argparse
import pickle
import logging
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from util.logging_util import init_logger
import util.bl_utils as bl_utils
from fine_train import main
import util.utils as ut

sane_space ={'model': 'MTLAGL',
         'hidden_size': hp.choice('hidden_size', [16, 32, 64, 128, 256]),
         'learning_rate': hp.uniform("lr", -3, -2),
         'weight_decay': hp.uniform("wr", -5, -3),
         'optimizer': hp.choice('opt', ['adagrad', 'adam']),
         'in_dropout': hp.choice('in_dropout', [0, 1, 2, 3, 4, 5, 6]),
         'out_dropout': hp.choice('out_dropout', [0, 1, 2, 3, 4, 5, 6]),
         'activation': hp.choice('act', ['relu', 'elu']),
          'lamda':hp.choice('lamda', [0.001, 0.01, 2, 3, 8]),
         'alpha':hp.choice('alpha', [0.1, 0.5, 1, 1.2, 10])
}

def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument("--tasks", type=str, help="Tasks to be performed (default is 'gc,nc,lp').")
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--dataset-name', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
    # parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    # parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--hyper_epoch', type=int, default=5, help='epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=400, help='epoch in train GNNs.')
    parser.add_argument('--opepochs', type=int, default=20, help='epoch in train GNNs in hyperopt.')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    parser.add_argument('--fix_last', type=bool, default=True, help='fix last layer in design architectures.')
    parser.add_argument("--data-folder", type=str, default="",
                        help="Path to the folder where data will be stored (default is working directory).")

    global args1
    args1 = parser.parse_args()


class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)
    args.learning_rate = 10**args.learning_rate
    args.weight_decay = 10**args.weight_decay
    args.in_dropout = args.in_dropout / 10.0
    args.out_dropout = args.out_dropout / 10.0
    args.alpha = args.alpha
    args.lamda = args.lamda
    args.dataset_name = args1.dataset_name
    args.save = '{}_{}'.format(args.dataset_name, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    args1.save = 'logs/tune-{}'.format(args.save)
    args.epochs = args1.opepochs
    args.arch = args1.arch
    args.gpu = args1.gpu
    args.num_layers = args1.num_layers
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    args.with_linear = args1.with_linear
    args.with_layernorm = args1.with_layernorm
    args.cos_lr = args1.cos_lr
    args.fix_last = args1.fix_last
    args.data_folder = args1.data_folder
    args.tasks = args1.tasks
    args.batch_size = args1.batch_size
    args.weight_pareto = args1.weight_pareto
    args.normalization_type = args1.normalization_type
    # args.alpha = args1.alpha
    # args.lamda = args1.lamda
    args.n_test_rays = args1.n_test_rays
    args.reference_point = args1.reference_point
    args.ind = 0
    return args

def objective(args):
    args = generate_args(args)
    vali_acc_sum, test_acc_sum, vali_acc, test_acc, best_test_hv_all, best_valid_hv_all, args = main(args)
    return {
        'loss': -vali_acc_sum,
        'test_acc_sum': test_acc_sum,
        'vali_acc': vali_acc,
        'test_acc': test_acc,
        'best_test_hv': best_test_hv_all["hv"],
        'best_valid_hv': best_valid_hv_all["hv"],
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():

    lines = open(args1.arch_filename, 'r').readlines()
    suffix = args1.arch_filename.split('_')[-1][:-4]  # need to re-write the suffix?

    test_res = []
    arch_set = set()
    sane_space['learning_rate'] = hp.uniform("lr", -3, -1.6)
    sane_space['in_dropout'] = hp.choice('in_dropout', [0, 1])
    sane_space['out_dropout'] = hp.choice('out_dropout', [0, 1])
    sane_space['hidden_size'] = hp.choice('hidden_size', [64, 128, 256, 512, 1024])
    sane_space['lamda'] = hp.choice('lamda', [0.001, 0.01, 2, 3, 8])
    sane_space['alpha'] = hp.choice('alpha', [0.1, 0.5, 1, 1.2, 10])

    for ind, l in enumerate(lines):
        try:
            # if ind > 0:
            #     continue
            print('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]

            log_data_dir = parts[2].split('=')[1]
            log_lines = open(log_data_dir+'/log.txt', 'r').readlines()
            log_parts = log_lines[0]
            log_parts1 = log_parts.strip().split('=')
            log_tel = ast.literal_eval(log_parts1[1].strip())
            # log_tel = re.findall('\'tasks\':[[](^")+[]] ', log_parts)
            args1.dataset_name = log_tel['dataset_name']
            args1.tasks = log_tel['tasks']
            # args1.batch_size = log_tel['batch_size']
            args1.batch_size = 128
            args1.weight_pareto = log_tel['weight_pareto']
            args1.normalization_type = log_tel['normalization_type']
            args1.alpha = log_tel['alpha']
            args1.lamda = log_tel['lamda']
            args1.n_test_rays = log_tel['n_test_rays']
            args1.reference_point = log_tel['reference_point']

            path = 'logs/tune-{}-th-{}-'.format(ind+1, args1.dataset_name)
            for k, j in enumerate(args1.tasks):
                path += j
                if k != len(args1.tasks) - 1:
                    path += '-'

            if not os.path.exists(path):
                os.mkdir(path)
            log_filename = os.path.join(path, 'log.txt')
            init_logger('fine-tune', log_filename, logging.INFO, False)

            args1.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            start = time.time()
            start = time.time()
            trials = Trials()
            args1.ind = ind
            #tune with validation acc, and report the test accuracy with the best validation acc
            best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)),
                        max_evals=args1.hyper_epoch, trials=trials)

            space = hyperopt.space_eval(sane_space, best)
            print('best space is ', space)
            res['best_space'] = space
            args = generate_args(space)
            print('best args from space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            record_time_res = []
            c_vali_acc, c_test_acc_sum = 0, 0
            #report the test acc with the best vali acc
            for d in trials.results:
                if -d['loss'] > c_vali_acc:
                    c_vali_acc = -d['loss']
                    c_test_acc_sum = d['test_acc_sum']
                    c_test_acc = d['test_acc']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_acc, c_test_acc))
            res['test_acc_sum'] = c_test_acc_sum
            print('test_acc_sum={}'.format(c_test_acc_sum))
            res['test_acc'] = c_test_acc
            pprint.pprint(c_test_acc)

            #print('test_res=', res)

            test_accs_sum = []
            test_acc_all = {}
            args.epochs = args1.epochs
            test_acc_a = []
            best_test_hv_a = []
            tt = 5
            for i in range(tt):
                vali_acc_sum, test_acc_sum, vali_acc, test_acc, best_test_hv_all, best_valid_hv_all, test_args = main(args)
                print('cal std: times:{}, valid_Acc_sum:{}, test_acc_sum:{}, valid_hv:{}, test_hv:{}'.format(i, vali_acc_sum, test_acc_sum, best_valid_hv_all["hv"], best_valid_hv_all["hv"] ))
                pprint.pprint(test_acc)
                test_acc_a.append(test_acc)
                best_test_hv_a.append(best_test_hv_all)
                test_acc_all = bl_utils.dict_add(test_acc_all, test_acc)
                test_accs_sum.append(test_acc_sum)
            test_accs_sum = np.array(test_accs_sum)
            print('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs_sum), np.std(test_accs_sum)))

            saved_dir = ut.save_result(args, test_acc_a, best_test_hv_a, path, "multitask_COSMOS_result")
            print("Result saved at path:", saved_dir)

            test_acc_all = bl_utils.dict_di(test_acc_all, tt)
            pprint.pprint(test_acc_all)

            test_res.append(res)
            test_res.append(res)
            with open('tuned_res/%s_res_%s.pkl' % (args1.dataset_name, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}**************'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind+1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('finsh tunining {} archs, saved in {}'.format(len(arch_set), 'tuned_res/%s_res_%s.pkl' % (args1.dataset_name, suffix)))

if __name__ == '__main__':
    get_args()
    args1.arch_filename = 'exp_res/PROTEINS-searched_res-epoch500-gc-nc-lp.txt'
    args1.hyper_epoch = 50
    args1.opepochs = 100
    args1.epochs = 1000
    args1.gpu = 1
    if args1.arch_filename:
        run_fine_tune()
