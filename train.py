import os
import argparse

import torch

import config
from model import model_factory
from load_dataset import Loader
from lib.training import get_parser
from lib.training.pipeline import Pipeline
from lib.utils import (Params, Logger, Checkpoint, BestMetricRecorder, 
    ExperimentCfg, DatasetCfg)


class BestAccuracyRecorder(BestMetricRecorder):

    def __init__(self, init_value=0.0, metric_name='accuracy'):
        super().__init__(init_value, metric_name)

    def compare(self, current_value, best_value):
        if current_value > best_value:
            return 1
        return -1


def load_data(params, data_dir, train_name, val_name, 
              encoding='utf8', to_tensor=True, shuffle=True):
    data_loader = Loader(data_dir, params, encoding=encoding)
    trainloader = data_loader.create_loader(train_name, 
                                            params.train_size,
                                            encoding=encoding,
                                            batch_size=params.batch_size,
                                            to_tensor=to_tensor,
                                            to_cuda=params.cuda,
                                            shuffle=shuffle)
    valloader = data_loader.create_loader(val_name, 
                                          params.val_size,
                                          encoding=encoding,
                                          batch_size=params.batch_size,
                                          to_tensor=to_tensor,
                                          to_cuda=params.cuda,
                                          shuffle=shuffle)
    return trainloader, valloader



if __name__ == '__main__':
    # load parser
    parser = get_parser(
        data_dir=config.data_dir, 
        exp_dir=config.base_model_dir,
        restore_checkpoint=None
    )

    # parse command line arguments
    args = parser.parse_args()
    restore_checkpoint = args.restore_checkpoint
    dataset_cfg = DatasetCfg(args.data_dir)
    exp_cfg = ExperimentCfg(args.exp_dir)

    # set logger
    # Note: log file will be stored in the `exp_dir` directory
    logger = Logger.set(exp_cfg.train_log())

    # load experiment configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(exp_cfg.params_file())
    logger.info("- done.")

    # set params
    params.set('cuda', torch.cuda.is_available())

    # load datesets
    logger.info("Loading the datasets...")
    # add datasets parameters into params
    params.update(Params(dataset_cfg.params_file()))
    trainloader, valloader = load_data(params,
                                       dataset_cfg.data_dir(), 
                                       dataset_cfg.train_name(),
                                       dataset_cfg.val_name(),
                                       encoding='utf8')
    logger.info("- done.")
    
    # run training pipeline
    logger.info("Starting training for {} epoch(s)".format(
        params.num_epochs))
    Pipeline(
        dataset_cfg=dataset_cfg,
        experiment_cfg=exp_cfg,
        trainloader=trainloader,
        valloader=valloader,
        params=params,
        model_factory=model_factory,
        best_metric_recorder=BestAccuracyRecorder(),
        restore_checkpoint=restore_checkpoint,
        logger=logger
    ).run()
    logger.info("- done.")
