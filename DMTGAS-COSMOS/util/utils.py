import random
import numpy as np
import torch
import os
import shutil

def set_seeds(seed=None):
    if not seed:
        seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def print_arguments(args):
    print("--- Arguments: ")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("---------------\n")

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def save_point(point, name):
    """ Saves the point. """
    filename = os.path.join('{}.txt'.format(name))
    with open(filename, 'w') as f:
        for i in point:
            for j in i:
                f.write(f"{j:>7.4f}\t")
            f.write("\n")

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def save_result(args, accs, best_test_hv_all, output_folder, name):
    """ Saves the acc, hv result. """
    accs_array = np.zeros((len(accs[0]), len(accs)))
    str_stats = "Task:  Avg. +- Std.  |  Max.  |  Min.\n"
    for i_index, i in enumerate(accs):
        for j_index, j in enumerate(i):
            accs_array[j_index, i_index] = i[j]

    avg = accs_array.mean(axis=1)
    std = accs_array.std(axis=1)
    min = accs_array.min(axis=1)
    max = accs_array.max(axis=1)
    for index, task in enumerate(args.tasks):
        str_stats += f"{task:>4}:{avg[index]:>7.4f}+-{std[index]:<7.4f}|{max[index]:^8.4f}|{min[index]:^8.4f}\n"

    hv_all = 0.0
    for cv_fold_stats in best_test_hv_all:
        hv = cv_fold_stats["hv"]
        hv_all += hv
    hv_all = hv_all / len(best_test_hv_all)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    filename = os.path.join(output_folder, '{}.txt'.format(name))
    with open(filename, 'w') as f:
        f.write("--- Cross Validation Results:\n")
        f.write("-- Accuracies\n")
        f.write(str_stats)
        f.write("--hv:\n")
        f.write(f"{hv_all:>7.4f}")

    return filename