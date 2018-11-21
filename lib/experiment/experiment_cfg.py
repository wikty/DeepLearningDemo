import os


class ExperimentCfg(object):

    def __init__(self, experiment_dir,
                 params_file='params.json',
                 checkpoint_file='{checkpoint}.pth.tar',
                 metrics_file='{checkpoint}_metrics_on_{dataset}.json',
                 train_log='train.log',
                 evaluate_log='evaluate.log',
                 best_checkpoint='best',
                 latest_checkpoint='latest'):
        msg = "Experiment directory not exists"
        assert os.path.exists(experiment_dir), msg
        msg = "Experiment parameters file not exists"
        assert os.path.exists(os.path.join(experiment_dir, params_file)), msg
        self._experiment_dir = experiment_dir
        self._params_file = params_file
        self._checkpoint_file = checkpoint_file
        self._metrics_file = metrics_file
        self._train_log = train_log
        self._evaluate_log = evaluate_log
        self._best_checkpoint = best_checkpoint
        self._latest_checkpoint = latest_checkpoint

    def experiment_dir(self):
        return self._experiment_dir

    def params_filename(self):
        return self._params_file

    def params_file(self):
        return os.path.join(self._experiment_dir, 
                            self._params_file)

    def checkpoint_filename(self):
        return self._checkpoint_file

    def checkpoint_file(self, checkpoint):
        return os.path.join(self._experiment_dir, 
                            self._checkpoint_file.format(
                                checkpoint=checkpoint))

    def best_checkpoint(self):
        return self._best_checkpoint

    def best_checkpoint_file(self):
        return self.checkpoint_file(self._best_checkpoint)

    def latest_checkpoint(self):
        return self._latest_checkpoint

    def latest_checkpoint_file(self):
        return self.checkpoint_file(self._latest_checkpoint)

    def metrics_filename(self, checkpoint, dataset):
        return self._metrics_file.format(checkpoint=checkpoint,
                                         dataset=dataset)

    def metrics_file(self, checkpoint, dataset):
        return os.path.join(self._experiment_dir, 
                            self._metrics_file.format(
                                checkpoint=checkpoint, 
                                dataset=dataset))

    def best_metrics_file(self, dataset):
        return self.metrics_file(self._best_checkpoint, dataset)

    def latest_metrics_file(self, dataset):
        return self.metrics_file(self._latest_checkpoint, dataset)

    def train_log(self):
        return os.path.join(self._experiment_dir, 
                            self._train_log)

    def evaluate_log(self):
        return os.path.join(self._experiment_dir, 
                            self._evaluate_log)



