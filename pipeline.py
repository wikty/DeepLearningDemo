import os
import sys
import subprocess

import config


PYTHON = sys.executable


class Pipeline(object):

    def __init__(self, context={}):
        self.workers = []
        self.context = {}
        self.context.update(context)

    def append(self, worker, **worker_kwargs):
        self.workers.append((worker, worker_kwargs))

    def run(self, kwargs={}):
        self.context.update(kwargs)
        for worker, worker_kwargs in self.workers:
            results = worker(self.context, **worker_kwargs)
            if results is not None:
                msg = "worker must be return a dict or None."
                assert isinstance(results, dict), msg
                self.context.update(results)


def generate_arguments(l=[], args=[], context={}):
    for arg in args:
        if isinstance(arg, (tuple, list)):
            l.append(arg[0])
            l.append(str(arg[1]))
        elif isinstance(arg, str):
            l.append(arg)
            value = str(context.get(arg.strip('-').replace('-', '_'), ''))
            l.append(value)
    return l


def build(cxt):
    args = generate_arguments([PYTHON, cxt['build_dataset_script']], [
        '--data-dir',
        '--data-file',
        '--data-factor',
        '--train-factor',
        '--val-factor',
        '--test-factor',
        '--min-count-tag',
        '--min-count-word'], cxt)
    subprocess.run(args)


def train(cxt):
    args = generate_arguments([PYTHON, cxt['train_script']], [
        '--data-dir',
        '--exp-dir'], cxt)
    subprocess.run(args)


def refine(cxt):
    args = generate_arguments([PYTHON, cxt['refine_script']], [
        '--data-dir',
        '--exp-dir',
        ('--restore-checkpoint', 'best'),
        ('--job', 'all')], cxt)
    subprocess.run(args)


def summary(cxt):
    args = generate_arguments([PYTHON, cxt['summary_script']], [
        '--exp-dir',
        ('--find-best', True)], cxt)
    rtn = subprocess.run(args, capture_output=True, text=True)
    best_mode_dir = rtn.stdout.strip()
    return {'exp_dir': best_mode_dir}


def evaluate(cxt):
    args = generate_arguments([PYTHON, cxt['evaluate_script']], [
        '--exp-dir', '--data-dir'], cxt)
    subprocess.run(args)
    print('The directory of the best model: {}'.format(
        cxt['exp_dir']))


if __name__ == '__main__':
    p = Pipeline({
        'data_dir': config.data_dir,
        'exp_dir': config.base_model_dir,
        'data_file': os.path.join(config.data_dir, 'ner_dataset.csv'),
        'data_factor': 0.05,
        'train_factor': 0.7,
        'val_factor': 0.15,
        'test_factor': 0.15,
        'min_count_tag': config.min_count_tag,
        'min_count_word': config.min_count_word,
        'build_dataset_script': 'build_dataset.py',
        'train_script': 'train.py',
        'refine_script': 'search_hyperparams.py',
        'summary_script': 'summary_experiments.py',
        'evaluate_script': 'evaluate.py'
    })
    p.append(build)
    p.append(train)
    p.append(refine)
    p.append(summary)
    p.append(evaluate)
    p.run()