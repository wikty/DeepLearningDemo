import os
import json
import argparse

import config
from lib.experiment import ExperimentCfg, DatasetCfg
from lib.experiment.summary import summary


if __name__ == '__main__':
    dataset_cfg = DatasetCfg(config.data_dir)
    exp_cfg = ExperimentCfg(config.base_model_dir)

    exp_dir = config.base_model_dir
    params_filename = config.params_filename

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', 
                        default=exp_cfg.experiment_dir(), 
                        help="The directory containing experiment results.")
    parser.add_argument('--checkpoint',
                        default=exp_cfg.best_checkpoint(),
                        help='The checkpoint of experiment want to summary.')
    parser.add_argument('--dataset',
                        default=dataset_cfg.val_name(),
                        help='The dataset of experiment want to summary')
    parser.add_argument('--find-best',
                        default=False,
                        help="Flag to enable to find the best model.",
                        type=bool)
    parser.add_argument('--output-format',
                        choices=['table', 'csv'],
                        default='table',
                        help="The format of output.")

    args = parser.parse_args()
    exp_cfg.set_experiment_dir(args.exp_dir)

    metrics = summary(
        exp_cfg.experiment_dir(), 
        exp_cfg.params_filename(), 
        exp_cfg.metrics_filename(args.checkpoint, args.dataset)
    )

    # print summary information to console
    if args.find_best:
        row = metrics.max('accuracy')
        print(row['experiment_dir'])
    elif args.output_format == 'table':
        print(metrics.tabulate())
    elif args.output_format == 'csv':
        print(metrics.csv())
    else:
        print('Invalid output format!')