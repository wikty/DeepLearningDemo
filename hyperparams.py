import os
import sys
import shutil
import argparse
import subprocess


import config
from lib.utils import Params
from lib.hyperparam_optim.grid_search import Searcher


def train_model(exp_dir, data_dir, restore_checkpoint):
    args = [sys.executable, 
            'train.py',
            '--exp-dir', exp_dir,
            '--data-dir', data_dir]
    if restore_checkpoint is not None:
        args.extend(['--restore-checkpoint', restore_checkpoint])
    subprocess.run(args, check=True, shell=True)


if __name__ == '__main__':
    data_dir = config.data_dir
    exp_dir = config.base_model_dir


    params_filename = 'params.json'

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory containing the datasets")
    parser.add_argument('--exp-dir', 
                        default=exp_dir, 
                        help="The directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--restore-checkpoint', 
                        default=None,
                        help="The name of checkpoint to restore model")
    parser.add_argument('--job',
                        choices=['lr', 'bz', 'ed', 'all'],
                        help="The hyperparameters name want to search.")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    exp_dir = args.exp_dir
    restore_checkpoint = args.restore_checkpoint
    job = args.job
    msg = "Data directory not exists: {}"
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = "Experiment directory not exists: {}"
    assert os.path.isdir(exp_dir), msg.format(exp_dir)
    params_file = os.path.join(exp_dir, params_filename)
    msg = "Experiment config file not exists: {}"
    assert os.path.isfile(params_file), msg.format(params_file)

    # config your experiment learning rates
    hyperparams = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [10, 50, 100],
        'embedding_dim': [50, 100, 200]
    }

    searcher = Searcher(exp_dir, params_filename, train_model, {
        'data_dir': data_dir,
        'exp_dir': exp_dir,
        'restore_checkpoint': restore_checkpoint
    })

    searcher.run({
        'learning_rate': [1e-4, 1e-3, 1e-2],

    })

    # if job == 'lr':
    #     search_hyperparam(('learning_rate', hyperparams['learning_rate']),
    #                       model_dir, data_dir, checkpoint, params_filename)
    # elif job == 'bz':
    #     search_hyperparam(('batch_size', hyperparams['batch_size']),
    #                       model_dir, data_dir, checkpoint, params_filename)
    # elif job == 'ed':
    #     search_hyperparam(('embedding_dim', hyperparams['embedding_dim']),
    #                       model_dir, data_dir, checkpoint, params_filename)
    # elif job == 'all':
    #     search_all(hyperparams, model_dir, data_dir, checkpoint, 
    #                params_filename)
    # else:
    #     print('Please specify the valid job name!')
    