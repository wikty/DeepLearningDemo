import json

from .checkpoint import Checkpoint
from .counter import Counter
from .logger import Logger
from .params import Params
from .progress_bar_wrapper import ProgressBarWrapper
from .running_avg import RunningAvg
from .table import Table
from .vocab import Vocab


def load_from_json(json_file, encoding='utf8'):
    with open(json_file, 'r', encoding=encoding) as f:
        return json.load(f)


def dump_to_json(data, json_file, encoding='utf8', indent=4):
    with open(json_file, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)