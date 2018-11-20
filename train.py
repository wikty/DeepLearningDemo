import os
import argparse

import torch

import config
from model import model_factory
from load_dataset import Loader
from evaluate import evaluate
from lib.utils import (Params, Logger, RunningAvg, dump_to_json, 
    Checkpoint, ProgressBarWrapper, BestMetricRecorder, ExperimentCfg)
from lib.training.pipeline import Pipeline


class BestAccuracyRecorder(BestMetricRecorder):

    def __init__(self, init_value=0.0, metric_name='accuracy'):
        super().__init__(init_value, metric_name)

    def compare(self, current_value, best_value):
        if current_value > best_value:
            return 1
        return -1


if __name__ == '__main__':
    data_dir = config.data_dir
    exp_dir = config.base_model_dir
    datasets_params_file = config.datasets_params_file
    train_name = config.train_name
    val_name = config.val_name

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory contains datasets.")
    parser.add_argument('--exp-dir', 
                        default=exp_dir, 
                        help="The experiment directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--checkpoint', 
                        default=None,
                        help="The name of checkpoint to restore model.")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    exp_dir = args.exp_dir
    checkpoint = args.checkpoint

    # check settings
    assert os.path.isdir(data_dir), "Data directory not exists"
    assert os.path.isdir(exp_dir), "Experiment directory not exists"
    exp_cfg = ExperimentCfg(exp_dir)
    assert os.path.isfile(exp_cfg.params_file()), (
        "Experiment parameters file not exists")

    # set logger
    # Note: log file will be stored in the `exp_dir` directory
    logger = Logger.set(exp_cfg.train_log())

    # load experiment configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(exp_cfg.params_file())
    logger.info("- done.")

    # set cuda flag
    params.set('cuda', torch.cuda.is_available())

    # load datesets
    logger.info("Loading the datasets...")
    datasets_params = Params(datasets_params_file)
    loader = Loader(data_dir, datasets_params, encoding='utf8')
    # add datasets parameters into params
    params.update(datasets_params)
    # make dateset loaders
    def trainloader(shuffle=True):
        return loader.load(train_name, params.train_size,
                           encoding='utf8',
                           batch_size=params.batch_size,
                           to_tensor=True,
                           to_cuda=params.cuda,
                           shuffle=shuffle)
    def valloader(shuffle=True):
        return loader.load(val_name, params.val_size,
                           encoding='utf8',
                           batch_size=params.batch_size,
                           to_tensor=True,
                           to_cuda=params.cuda,
                           shuffle=shuffle)
    logger.info("- done.")
    
    # create model, optimizer and so on.
    model, optimizer, criterion, metrics = model_factory(params)
    
    # run train pipeline
    num_epochs = params.num_epochs
    running_avg_steps = params.running_avg_steps
    logger.info("Starting training for {} epoch(s)".format(num_epochs))
    Pipeline(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        trainloader=trainloader,
        valloader=valloader,
        model_dir=exp_cfg.experiment_dir(),
        best_metrics_file=exp_cfg.metrics_file('best', val_name),
        latest_metrics_file=exp_cfg.metrics_file('latest', val_name),
        best_metric_recorder=BestAccuracyRecorder(),
        num_epochs=params.num_epochs,
        running_avg_steps=params.running_avg_steps,
        restore_checkpoint=checkpoint,
        logger=logger
    ).run()
    logger.info("- done.")

