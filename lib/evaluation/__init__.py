import os
import argparse


def get_parser(data_dir, exp_dir):
    
    def to_dir(path):
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError("{} isn't a valid directory!".format(
                path))
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        type=to_dir,
                        help="The directory containing the datasets")
    parser.add_argument('--exp-dir', 
                        default=exp_dir, 
                        type=to_dir,
                        help="The experiment directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--restore-checkpoint', 
                        default='best',
                        help="The name of checkpoint to restore model")
    parser.add_argument('--dataset-name',
                        default='test',
                        help="The name of dataset that want to evaluate")

    return parser