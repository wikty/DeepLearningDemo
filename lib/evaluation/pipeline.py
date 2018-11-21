import torch

from lib.utils import (Checkpoint, Logger, RunningAvg, dump_to_json,
    ProgressBarWrapper)


class Pipeline(object):

    def __init__(self, model_factory, dataset, params, dataset_cfg, 
                 experiment_cfg, restore_checkpoint, logger=None):
        items = model_factory(params)
        self.logger = logger if logger else Logger.get()
        self.model = items['model']
        # self.optimizer = items['optimizer']
        self.criterion = items['criterion']
        self.metrics = items['metrics']
        # restore model from checkpoint
        checkpoint = Checkpoint(
            checkpoint_dir=experiment_cfg.experiment_dir(),
            filename=experiment_cfg.checkpoint_filename(),
            best_checkpoint=experiment_cfg.best_checkpoint(),
            latest_checkpoint=experiment_cfg.latest_checkpoint(),
            logger=self.logger)
        status = checkpoint.restore(self.model, None, restore_checkpoint)
        assert status, "Restore model from the checkpoint: {}, failed".format(
            restore_checkpoint)
        self.dataset = dataset
        self.metrics_file = experiment_cfg.metrics_file(restore_checkpoint,
            self.dataset.dataset_name)

    def action_before_run(self, context={}):
        self.logger.info('Evaluation pipeline start running...')

    def action_before_evaluate(self, context={}):
        assert ('dataset_stat' in context)
        self.logger.info('- Dataset:')
        for name, value in context['dataset_stat'].items():
            self.logger.info('    {}: {}'.format(name, value))

    def action_after_evaluate(self, context={}):
        assert 'evaluation_metrics' in context
        self.logger.info("- Evaluation metrics:")
        for name, value in context['evaluation_metrics'].items():
            self.logger.info('    * {}: {:05.3f}'.format(name, value))
        self.logger.info("Save metrics results...")
        dump_to_json(context['evaluation_metrics'], self.metrics_file)
        self.logger.info("- done.")

    def action_after_run(self, context={}):
        self.logger.info('Evaluation pipeline is done!')

    def evaluate(self, num_batches):
        running_avg = RunningAvg()
        # set model to evaluation mode
        self.model.eval()
        # wrap the dataset to show a progress bar of iteration
        # prefix = 'Evaluate-{}'.format(self.dataset.dataset_name)
        prefix = 'Evaluate-{}'.format(num_batches)
        bar = ProgressBarWrapper(self.dataset, 
                                 num_batches,
                                 with_bar=False,
                                 with_index=True,
                                 prefix=prefix,
                                 suffix='batch=None')
        for i, batch in bar:
            inputs, targets = batch
            stat = {}
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                stat['loss'] = loss.item()
                for name, metric in self.metrics.items():
                    stat[name] = metric(outputs, targets).item()
            running_avg.step(stat)
            bar.set_suffix('batch={}'.format(i))
        return running_avg()

    def run(self, context={}):
        self.action_before_run(context)
        # action before evaluate
        num_batches = int(self.dataset.dataset_size / self.dataset.batch_size)
        context['dataset_stat'] = {
            'name': self.dataset.dataset_name,
            'size': self.dataset.dataset_size,
            'num_batches': num_batches
        }
        self.action_before_evaluate(context)
        # evaluate
        evaluation_metrics = self.evaluate(num_batches)
        # action after evaluate
        context['evaluation_metrics'] = evaluation_metrics
        self.action_after_evaluate(context)
        # action after run
        self.action_after_run(context)
