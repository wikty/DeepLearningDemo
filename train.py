import os
import argparse

import torch

import config
from model import model_factory
from load_dataset import Loader
from evaluate import evaluate
from lib.utils import (Params, Logger, RunningAvg, dump_to_json, 
    Checkpoint, ProgressBarWrapper, BestMetricRecorder)
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
    model_dir = config.base_model_dir
    params_filename = config.params_file
    log_filename = config.train_log
    datasets_params_file = config.datasets_params_file
    train_name = config.train_name
    val_name = config.val_name
    best_metrics_filename = config.best_metrics_on_val_file
    last_metrics_filename = config.last_metrics_on_val_file

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory contains datasets.")
    parser.add_argument('--model-dir', 
                        default=model_dir, 
                        help="The directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--checkpoint', 
                        default=None,
                        help="The name of checkpoint to restore model.")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    checkpoint = args.checkpoint

    # check settings
    msg = "Data directory not exists: {}"
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = "Model directory not exists: {}"
    assert os.path.isdir(model_dir), msg.format(model_dir)
    params_file = os.path.join(model_dir, params_filename)
    msg = "Model config file not exists: {}"
    assert os.path.isfile(params_file), msg.format(params_file)
    best_metrics_file = os.path.join(model_dir, best_metrics_filename)
    last_metrics_file = os.path.join(model_dir, last_metrics_filename)

    # set logger
    # Note: log file will be stored in the `model_dir` directory
    logger = Logger.set(os.path.join(model_dir, log_filename))

    # load model configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(params_file)
    # cuda flag
    params.set('cuda', torch.cuda.is_available())
    logger.info("- done.")

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
    # run train and evaluate
    num_epochs = params.num_epochs
    running_avg_steps = params.running_avg_steps
    logger.info("Starting training for {} epoch(s)".format(num_epochs))
    best_acc_recorder = BestAccuracyRecorder()
    Pipeline(
        model_dir=model_dir,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        best_metric_recorder=best_acc_recorder,
        trainloader=trainloader,
        valloader=valloader,
        best_metrics_file=best_metrics_file,
        latest_metrics_file=last_metrics_file,
        num_epochs=num_epochs,
        running_avg_steps=running_avg_steps,
        restore_checkpoint=checkpoint,
        logger=logger
    ).run()
    # run(model, optimizer, criterion, metrics, trainloader, valloader, 
    #     num_epochs, running_avg_steps, model_dir, checkpoint,
    #     best_metrics_file, last_metrics_file)
    logger.info("- done.")

