from lib.utils import (Checkpoint, Logger, RunningAvg, ProgressBarWrapper,
    ContextVariable, dump_to_json)

import torch


class Pipeline(object):

    def __init__(self, model, optimizer, criterion, metrics,
        best_metric_recorder, trainloader, valloader, checkpoint,
        best_metrics_file, latest_metrics_file, num_epochs, 
        running_avg_steps, restore_checkpoint=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.best_metric_recorder = best_metric_recorder
        self.trainloader = trainloader
        self.valloader = valloader
        self.best_metrics_file = best_metrics_file
        self.latest_metrics_file = latest_metrics_file
        self.checkpoint = checkpoint
        self.num_epochs = num_epochs
        self.running_avg_steps = running_avg_steps
        self.restore_checkpoint = restore_checkpoint
        self.logger = logger if logger else Logger.get()

    def action_before_run(self, context={}):
        self.logger.info('Training pipeline start running...')
        # restore from a checkpoint if provide it
        if self.restore_checkpoint:
            extra = {}
            self.checkpoint.restore(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint=self.restore_checkpoint,
                extra=extra
            )
            context['checkpoint_extra'] = extra

    def action_before_epoch(self, context={}):
        assert ('epoch' in context)
        self.logger.info('Train Epoch - {}/{}'.format(
            context['epoch'], self.num_epochs))

    def action_before_train(self, context={}):
        assert ('trainset_stat' in context)
        self.logger.info('Start training...')
        self.logger.info('- Training set:')
        for name, value in context['trainset_stat'].items():
            self.logger.info('    {}: {}'.format(name, value))

    def action_after_train(self, context={}):
        assert ('training_metrics' in context)
        self.logger.info("- Training metrics:")
        for name, value in context['training_metrics'].items():
            self.logger.info('    * {}: {:05.3f}'.format(name, value))

    def action_before_evaluate(self, context={}):
        assert ('valset_stat' in context)
        self.logger.info('Start evaluating...')
        self.logger.info('- Validation set:')
        for name, value in context['valset_stat'].items():
            self.logger.info('    {}: {}'.format(name, value))

    def action_after_evaluate(self, context={}):
        assert ('evaluation_metrics' in context)
        self.logger.info("- Evaluation metrics:")
        for name, value in context['evaluation_metrics'].items():
            self.logger.info('    * {}: {:05.3f}'.format(name, value))

    def action_after_epoch(self, context={}):
        assert ('evaluation_metrics' in context) and ('epoch' in context)
        epoch = context['epoch']
        metrics_result = context['evaluation_metrics']
        # save the metrics result
        is_best = False
        if self.best_metric_recorder.improved(metrics_result):
            self.logger.info("- Found model with the best metric: {}".format(
                self.best_metric_recorder.value()))
            # Save best val metrics
            dump_to_json(metrics_result, self.best_metrics_file)
        else:
            # Save latest val metrics
            dump_to_json(metrics_result, self.latest_metrics_file)
        # freeze checkpoint
        self.checkpoint.freeze(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            is_best=is_best
        )

    def action_after_run(self, context={}):
        self.logger.info('Training pipeline is done!')

    def train(self, trainset, num_batches, epoch):
        loss_avg = RunningAvg()
        training_avg = RunningAvg()
        # set model to training mode
        self.model.train()
        # wrap the dataset to show a progress bar of iteration
        bar = ProgressBarWrapper(trainset, 
                                 num_batches,
                                 with_bar=False,
                                 with_index=True,
                                 prefix='Epoch-{}'.format(epoch),
                                 suffix='loss=None')
        for i, batch in bar:
            # training
            inputs, targets = batch
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update train loss progress
            loss_avg.step(loss.item())
            bar.set_suffix('loss={:05.3f}'.format(loss_avg()))
            # compute metrics in every `running_avg_steps` steps
            if i % self.running_avg_steps == 0:
                stat = {'loss': loss.item()}
                with torch.no_grad():
                    for name, metric in self.metrics.items():
                        stat[name] = metric(outputs, targets).item()
                training_avg.step(stat)
        return training_avg()

    def evaluate(self, dateset):
        evaluation_avg = RunningAvg()
        # set model to evaluation mode
        self.model.eval()
        # evaluate
        for batch in dateset:
            inputs, targets = batch
            stat = {}
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                stat['loss'] = loss.item()
                for name, metric in self.metrics.items():
                    stat[name] = metric(outputs, targets).item()
            evaluation_avg.step(stat)
        return evaluation_avg()


    def run(self, context={}):
        # action before run
        self.action_before_run(context)
        # run many epochs
        for epoch in range(self.num_epochs):
            # action before epoch
            context['epoch'] = epoch
            self.action_before_epoch(context)
            # train phase
            trainset = self.trainloader()
            trainsize = trainset.dataset_size
            num_batches = int(trainsize / trainset.batch_size)
            # action before train
            context['trainset_stat'] = {
                'size': trainsize,
                'batches': num_batches
            }
            self.action_before_train(context)
            training_metrics = self.train(trainset, num_batches, epoch)
            # action after train
            context['training_metrics'] = training_metrics
            self.action_after_train(context)
            # evaluate phase
            valset = self.valloader()
            valsize = valset.dataset_size
            context['valset_stat'] = {
                'size': valsize
            }
            # action before evaluate
            self.action_before_evaluate(context)
            evaluation_metrics = self.evaluate(valset)
            # action after evaluate
            context['evaluation_metrics'] = evaluation_metrics
            self.action_after_evaluate(context)
            # action after epoch
            self.action_after_epoch(context)
        # action after run
        self.action_after_run(context)