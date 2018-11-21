import os
import shutil

from lib.utils import Params


class Searcher(object):

    def __init__(self, experiment_dir, params_filename, learner, 
                 learner_kwargs, grid_search_dirname='search_gird'):
        assert os.path.isdir(experiment_dir)
        assert os.path.isfile(os.path.join(experiment_dir, params_filename))
        assert hasattr(learner, '__call__')
        assert isinstance(learner_kwargs, dict)
        self.experiment_dir = experiment_dir
        self.experiment_params = Params(os.path.join(experiment_dir, 
                                                     params_filename))
        self.params_filename = params_filename
        self.learner = learner
        self.learner_kwargs = learner_kwargs
        self.grid_search_dirname = grid_search_dirname

    def search(self, hyperparam, parent_dir):
        assert isinstance(hyperparam, tuple)
        name, candidates = hyperparam
        params = Params(os.path.join(parent_dir, self.params_filename))
        experiment_dirs = []
        for candidate in candidates:
            experiment = '{}_{}'.format(name, candidate)
            experiment_dir = os.path.join(parent_dir, experiment)
            # create experiment directory
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)
            experiment_dirs.append(experiment_dir)
            # create params file for this experiment
            params.set(name, candidate)
            params.dump(os.path.join(experiment_dir, self.params_filename))
            # run subprocess to train model
            self.learner(**self.learner_kwargs)
        return experiment_dirs

    def run(self, hyperparams):
        assert isinstance(hyperparams, dict)
        # create grid search directory
        parent_dir = os.path.join(self.experiment_dir, 
                                  self.grid_search_dirname)
        if os.path.isdir(parent_dir):
            shutil.rmtree(parent_dir)
        os.makedirs(parent_dir)
        # create experiment params file
        self.experiment_params.dump(os.path.join(parent_dir, 
                                                 self.params_filename))
        # gird search
        pds = [parent_dir]
        for name, candidates in hyperparams.items():
            new_pds = []
            for pd in pds:
                experiment_dirs = self.search((name, candidates), pd)
                new_pds.extend(experiment_dirs)
            pds = new_pds
        return parent_dir
