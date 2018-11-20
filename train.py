import os
import argparse

import torch

import config
from model import model_factory
from load_dataset import Loader
from evaluate import evaluate
from lib.utils import (Params, Logger, RunningAvg, dump_to_json, 
    Checkpoint, ProgressBarWrapper, BestMetricRecorder, ExperimentCfg,
    DatasetCfg)
from lib.training import get_parser
from lib.training.pipeline import Pipeline


class BestAccuracyRecorder(BestMetricRecorder):

    def __init__(self, init_value=0.0, metric_name='accuracy'):
        super().__init__(init_value, metric_name)

    def compare(self, current_value, best_value):
        if current_value > best_value:
            return 1
        return -1


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

    train_name = dataset_cfg.train_name()
    val_name = dataset_cfg.val_name()

    # load experiment configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(exp_cfg.params_file())
    logger.info("- done.")

    # set cuda flag
    params.set('cuda', torch.cuda.is_available())

    # load datesets
    logger.info("Loading the datasets...")
    datasets_params = Params(dataset_cfg.params_file())
    loader = Loader(dataset_cfg.data_dir(), datasets_params, encoding='utf8')
    # add datasets parameters into params
    params.update(datasets_params)
    # make dataset loaders
    trainloader = loader.create_loader(train_name, params.train_size,
                                       encoding='utf8',
                                       batch_size=params.batch_size,
                                       to_tensor=True,
                                       to_cuda=params.cuda,
                                       shuffle=True)
    valloader = loader.create_loader(val_name, params.val_size,
                                     encoding='utf8',
                                     batch_size=params.batch_size,
                                     to_tensor=True,
                                     to_cuda=params.cuda,
                                     shuffle=True)
    logger.info("- done.")
    
    # create model, optimizer and so on.
    model, optimizer, criterion, metrics = model_factory(params)
    
    # run train pipeline
    num_epochs = params.num_epochs
    running_avg_steps = params.running_avg_steps
    logger.info("Starting training for {} epoch(s)".format(num_epochs))
    best_acc_recorder = BestAccuracyRecorder()
    checkpoint = Checkpoint(
        checkpoint_dir=exp_cfg.experiment_dir(),
        filename=exp_cfg.checkpoint_filename(),
        best_checkpoint=exp_cfg.best_checkpoint(),
        latest_checkpoint=exp_cfg.latest_checkpoint(),
        logger=logger)
    best_metrics_file = exp_cfg.best_metrics_file(val_name)
    latest_metrics_file = exp_cfg.latest_metrics_file(val_name)
    Pipeline(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        trainloader=trainloader,
        valloader=valloader,
        best_metrics_file=best_metrics_file,
        latest_metrics_file=latest_metrics_file,
        checkpoint=checkpoint,
        best_metric_recorder=best_acc_recorder,
        num_epochs=params.num_epochs,
        running_avg_steps=params.running_avg_steps,
        restore_checkpoint=restore_checkpoint,
        logger=logger
    ).run()
    logger.info("- done.")

