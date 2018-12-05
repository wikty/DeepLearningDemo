import os
import argparse

import torch

import config
from model import model_factory
from lib.utils import Checkpoint, Params, Logger, Vocab
from lib.experiment import DatasetCfg, ExperimentCfg
from lib.training import get_parser


def lines_align(lines, padding=' '):
    """
    Args:
        lines (list): a list of list of string.

    For Example:
    ```
    align_print([['hello', 'world'], ['O', 'B-org']])
    ```
    """
    data = {}
    outputs = []
    for line in lines:
        for i, item in enumerate(line):
            if i not in data:
                data[i] = []
            data[i].append(item)
        outputs.append([])
    for i in sorted(data.keys()):
        max_width = len(max(data[i], key=lambda s: len(s)))
        fmt = '{' + ':{}^{}'.format(padding, max_width) + '}'
        for row, item in enumerate(data[i]):
            outputs[row].append(fmt.format(item))
    return outputs


class SampleEncoder(object):

    def __init__(self):
        pass

    def encode(self, tokens, word_vocab, unk_word, to_cuda=None):
        """
        Args:
            tokens (list): a list of token string.
        """
        # to vocab index
        tokens_index = []
        for token in tokens:
            i = word_vocab.encode(token, default=None)
            if i is None:
                i = word_vocab.encode(unk_word)
            tokens_index.append(i)
        # to tensor
        inputs = torch.tensor(tokens_index, dtype=torch.long)
        if to_cuda:
            inputs = inputs.cuda()
        # to a mini-batch
        return inputs.unsqueeze(0)

    def decode(self, predictions, tag_vocab):
        """
        Args:
            tags (Tensor): the shape is (tokens_len, tag_vocab_size).
        """
        predictions = torch.argmax(predictions, dim=1)
        tags = [tag_vocab.decode(p.item()) for p in predictions]
        return tags


class SampleReader(object):

    def __init__(self, path, encoding):
        self.path = path
        self.encoding = encoding
        self.file = None

    def __enter__(self):
        self.file = open(self.path, 'r', encoding=self.encoding)
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def __iter__(self):
        for line in self.file:
            line = line.strip()
            if not line:
                continue
            yield line


def predict(model, word_vocab, tag_vocab, inputs_file, 
            outputs_file, unk_word, to_cuda, encoding):
    encoder = SampleEncoder()
    with open(outputs_file, 'w', encoding=encoding) as f:
        with SampleReader(inputs_file, encoding) as reader:
            for sample in reader:
                tokens = sample.split()
                inputs = encoder.encode(tokens, word_vocab, unk_word, to_cuda)
                outputs = model(inputs)
                tags = encoder.decode(outputs, tag_vocab)
                result = lines_align([tokens, tags])
                f.write(' '.join(result[0]) + '\n')
                f.write(' '.join(result[1]) + '\n')
               

if __name__ == '__main__':
    inputs_file = os.path.join(config.data_dir, config.inputs_filename)
    outputs_file = os.path.join(config.data_dir, config.outputs_filename)
    words_file = os.path.join(config.data_dir, config.words_filename)
    tags_file = os.path.join(config.data_dir, config.tags_filename)

    parser = get_parser(
        data_dir=config.data_dir, 
        exp_dir=config.base_model_dir,
        restore_checkpoint='best'
    )
    parser.add_argument('--inputs-file',
                        default=inputs_file,
                        help="The input file for prediction.")
    parser.add_argument('--outputs-file',
                        default=outputs_file,
                        help="The output file to save predictions.")
    parser.add_argument('--encoding',
                        default='utf8',
                        help="The encoding for input and output file.")

    args = parser.parse_args()
    dataset_cfg = DatasetCfg(args.data_dir)
    exp_cfg = ExperimentCfg(args.exp_dir)
    inputs_file = args.inputs_file
    outputs_file = args.outputs_file
    restore_checkpoint = args.restore_checkpoint
    encoding = args.encoding

    msg = "Inputs file not exists: {}"
    assert os.path.isfile(inputs_file), msg.format(inputs_file)

    logger = Logger.set(os.path.join(exp_cfg.experiment_dir(), 
                                     'predict.log'))

    checkpoint = Checkpoint(
        checkpoint_dir=exp_cfg.experiment_dir(),
        filename=exp_cfg.checkpoint_filename(),
        best_checkpoint=exp_cfg.best_checkpoint(),
        latest_checkpoint=exp_cfg.latest_checkpoint(),
        logger=logger)

    # load params
    word_vocab = Vocab(words_file)
    tag_vocab = Vocab(tags_file)

    params = Params(exp_cfg.params_file())
    params.update(Params(dataset_cfg.params_file()))
    params.set('cuda', torch.cuda.is_available())

    # restore model
    items = model_factory(params)
    model = items['model']
    checkpoint.restore(model, None, restore_checkpoint)
    
    # predict
    predict(model, word_vocab, tag_vocab, inputs_file, 
            outputs_file, params.unk_word, params.cuda, encoding)

    print("It's done! Please check the output file:")
    print(outputs_file)