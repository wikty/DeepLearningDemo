import os
import json
import argparse

import torch

import config
from model import model_factory
from load_dataset import Loader
from lib.evaluation import get_parser
from lib.evaluation.pipeline import Pipeline
from lib.experiment import ExperimentCfg, DatasetCfg
from lib.utils import Params, Logger


def load_data(params, data_dir, dataset_name, dataset_size, 
              encoding='utf8', to_tensor=True, shuffle=True):
    loader = Loader(data_dir, params, encoding=encoding)
    dataset = loader.load(dataset_name, dataset_size, 
                          encoding=encoding,
                          batch_size=params.batch_size,
                          to_tensor=to_tensor,
                          to_cuda=params.cuda,
                          shuffle=shuffle)
    return dataset


if __name__ == '__main__':
    dataset_cfg = DatasetCfg(config.data_dir)
    exp_cfg = ExperimentCfg(config.base_model_dir)
    parser = get_parser(
        data_dir=dataset_cfg.data_dir(), 
        exp_dir=exp_cfg.experiment_dir(),
        restore_checkpoint=exp_cfg.best_checkpoint(),
        dataset_name=dataset_cfg.test_name()
    )
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    restore_checkpoint = args.restore_checkpoint
    dataset_cfg.set_data_dir(args.data_dir)
    exp_cfg.set_experiment_dir(args.exp_dir)

    # set logger
    logger = Logger.set(exp_cfg.evaluate_log())

    # load model configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(exp_cfg.params_file())
    # cuda flag
    params.set('cuda', torch.cuda.is_available())
    logger.info("- done.")

    # load datesets
    logger.info("Loading the {} dataset...".format(dataset_name))
    # add datasets parameters into params 
    params.update(Params(dataset_cfg.params_file()))
    dataset = load_data(params, dataset_cfg.data_dir(), dataset_name,
        params['{}_size'.format(dataset_name)])
    logger.info("- done.")

    logger.info("Starting evaluate model on test dataset...")
    Pipeline(
        dataset_cfg=dataset_cfg,
        experiment_cfg=exp_cfg,
        params=params,
        dataset=dataset,
        model_factory=model_factory,
        restore_checkpoint=restore_checkpoint,
        logger=logger
    ).run()
    logger.info('- done.')