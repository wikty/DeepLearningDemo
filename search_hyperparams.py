import os
import sys
import shutil
import argparse
import subprocess


import config
from lib.utils import Params
from lib.experiment import DatasetCfg, ExperimentCfg
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
    dataset_cfg = DatasetCfg(config.data_dir)
    exp_cfg = ExperimentCfg(config.base_model_dir)
    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=dataset_cfg.data_dir(), 
                        help="The directory containing the datasets")
    parser.add_argument('--exp-dir', 
                        default=exp_cfg.experiment_dir(), 
                        help="The directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--restore-checkpoint', 
                        default=exp_cfg.best_checkpoint(),
                        help="The name of checkpoint to restore model")
    parser.add_argument('--job',
                        choices=['lr', 'bz', 'ed', 'all'],
                        default='all',
                        help="The hyperparameters name want to search.")

    # parse command line arguments
    args = parser.parse_args()
    restore_checkpoint = args.restore_checkpoint
    job = args.job
    dataset_cfg.set_data_dir(args.data_dir)
    exp_cfg.set_experiment_dir(args.exp_dir)

    # config your experiment learning rates
    hyperparams = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [10, 50, 100],
        'embedding_dim': [50, 100, 200]
    }

    searcher = Searcher(
        exp_cfg.experiment_dir(), 
        exp_cfg.params_filename(), 
        train_model, {
        'data_dir': dataset_cfg.data_dir(),
        'exp_dir': exp_cfg.experiment_dir(),
        'restore_checkpoint': restore_checkpoint
    })

    if job == 'lr':
        searcher.run({
            'learning_rate': hyperparams['learning_rate']
        })
    elif job == 'bz':
        searcher.run({
            'batch_size': hyperparams['batch_size']
        })
    elif job == 'ed':
        searcher.run({
            'embedding_dim': hyperparams['embedding_dim']
        })
    elif job == 'all':
        searcher.run(hyperparams)
    else:
        print('Please specify the valid job name!')