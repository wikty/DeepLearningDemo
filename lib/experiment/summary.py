import os

from lib.utils import load_from_json, Table


def extract(metrics, experiment_dir, params_filename, metrics_filename):
    # extract experiment result
    metrics_file = os.path.join(experiment_dir, metrics_filename)
    params_file = os.path.join(experiment_dir, params_filename)
    if os.path.isfile(metrics_file) and os.path.isfile(params_file):
        data = load_from_json(metrics_file)
        data.update(load_from_json(params_file))
        data['experiment_dir'] = experiment_dir
        metrics.append(data)
    # extract other experiments
    for subitem in os.listdir(experiment_dir):
        subdir = os.path.join(experiment_dir, subitem)
        if not os.path.isdir(subdir):
            continue
        extract(metrics, subdir, params_filename, metrics_filename)

def summary(experiment_dir, params_filename, metrics_filename):
    assert os.path.isdir(experiment_dir)
    metrics = Table()
    extract(metrics, experiment_dir, params_filename, metrics_filename)
    return metrics