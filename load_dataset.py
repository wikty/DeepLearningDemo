import os

import torch
import numpy as np

from lib.utils import Params, Vocab
from lib.dataset.transform import ComposeTransform
from lib.dataset.loader import DatasetHandler, BaseLoader


class VocabTransform(object):
    """Transform sample from string to integer."""

    def __init__(self, word_vocab, tag_vocab, unk_word):
        self.unk_word = unk_word
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

    def __call__(self, sample):
        if sample is None:
            return None
        tokens, labels = sample
        tokens_encoding, labels_encoding = [], []
        for token in tokens:
            encoding = self.word_vocab.encode(token, default=None)
            if encoding is None:
                encoding = self.word_vocab.encode(self.unk_word)
                assert encoding is not None  # unknow word must be in vocab
            tokens_encoding.append(encoding)
        for label in labels:
            encoding = self.tag_vocab.encode(label, default=None)
            assert encoding is not None  # tag must be in vocab
            labels_encoding.append(encoding)
        return tokens_encoding, labels_encoding


class PaddingTransform(object):
    """Pad the each sample in the batch to the same length."""

    def __init__(self, word_padding, tag_padding):
        self.word_padding = word_padding
        self.tag_padding = tag_padding

    def __call__(self, batch):
        sample_max_len = 0
        for sample in batch:
            tokens, labels = sample
            assert len(tokens) == len(labels)
            sample_len = len(tokens)
            if sample_len > sample_max_len:
                sample_max_len = sample_len
        new_batch = []
        for sample in batch:
            tokens, labels = sample
            padding_len = sample_max_len - len(tokens)
            tokens.extend([self.word_padding]*padding_len)
            labels.extend([self.tag_padding]*padding_len)
            new_batch.append((tokens, labels))
        return new_batch


class SplitTransform(object):
    """Split batch into input and output batch."""

    def __call__(self, batch):
        batch_input, batch_output = [], []
        for sample in batch:
            batch_input.append(sample[0])
            batch_output.append(sample[1])
        return batch_input, batch_output


class ToArrayTransform(object):
    """Convert list to numpy array."""

    def __init__(self, dtype=None):
        """
        Args:
            dtype (np.dtype): numpy dtype.
        """
        self.dtype = dtype

    def __call__(self, batch):
        batch_input, batch_target = batch
        return (np.array(batch_input, dtype=self.dtype), 
            np.array(batch_target, dtype=self.dtype))


class ToTensorTransform(object):

    def __init__(self, dtype=None, to_cuda=None):
        """
        Args:
            dtype (torch.dtype): torch dtype.
            to_cude (bool|device): flag to enable cude or the destination
                GPU device instance.
        """
        self.dtype = dtype
        self.to_cuda = to_cuda

    def __call__(self, batch):
        batch_input, batch_target = batch
        if not self.to_cuda:
            return (torch.tensor(batch_input, dtype=self.dtype), 
                torch.tensor(batch_target, dtype=self.dtype))
        elif isinstance(self.to_cuda, bool):
            return (torch.tensor(batch_input, dtype=self.dtype).cuda(),
                torch.tensor(batch_target, dtype=self.dtype).cuda())
        else:
            return (
                torch.tensor(batch_input, 
                             dtype=self.dtype).cuda(self.to_cuda),
                torch.tensor(batch_target, 
                             dtype=self.dtype).cuda(self.to_cuda)
            )


class NERHandler(DatasetHandler):
    """The handler to read samples."""

    def __init__(self, input_file, target_file, encoding='utf8'):
        self.input_file = input_file
        self.target_file = target_file
        self.encoding = encoding
        self.input_fh = None
        self.target_fh = None

    def __enter__(self):
        self.input_fh = open(self.input_file, 'r', encoding=self.encoding)
        self.target_fh = open(self.target_file, 'r', encoding=self.encoding)
        return self

    def __exit__(self, type, value, trace):
        if self.input_fh:
            self.input_fh.close()
        if self.target_fh:
            self.target_fh.close()
        self.input_fh = None
        self.target_fh = None

    def read(self):
        """Return a sample or None"""
        # handler not open
        if self.input_fh is None or self.target_fh is None:
            return None
        # each line is a sample
        input_line = self.input_fh.readline()
        target_line = self.target_fh.readline()
        # inputs and targets must be with the same size
        assert ((len(input_line) > 0 and len(target_line) > 0) or 
            (len(input_line) == 0 and len(target_line) == 0))
        if len(input_line) == 0:
            return None
        # split only by space ASCII 32, not include ASCII 160
        tokens = input_line.rstrip('\r\n').split(' ')
        labels = target_line.rstrip('\r\n').split(' ')
        msg = "tokens[{}] and labels[{}] of sentence {} don't match in file {}"
        assert len(tokens) == len(labels), \
            msg.format(len(tokens), len(labels), input_line, self.input_file)
        return (tokens, labels)


class Loader(BaseLoader):

    def __init__(self, data_dir, params, encoding):
        # check arguments
        assert os.path.isdir(data_dir)
        assert isinstance(params, Params)
        assert params.check_all(['unk_word', 'pad_word'])
        # info about dataset
        self.data_dir = data_dir
        self.params = params.copy()
        self.word_vocab_filename = 'words.txt'
        self.tag_vocab_filename = 'tags.txt'
        self.sentences_filename = 'sentences.txt'
        self.labels_filename = 'labels.txt'
        # sample transform
        words_file=os.path.join(self.data_dir, 
                                self.word_vocab_filename)
        tags_file=os.path.join(self.data_dir, 
                               self.tag_vocab_filename)
        word_vocab = Vocab(words_file, encoding=encoding)
        tag_vocab = Vocab(tags_file, encoding=encoding)
        self.sample_transform = VocabTransform(
            word_vocab=word_vocab,
            tag_vocab=tag_vocab,    
            unk_word=self.params.unk_word
        )
        # batch transform
        word_padding = word_vocab.encode(self.params.pad_word)
        tag_padding = -1  # don't use self.params.pad_tag
        self.batch_transform = ComposeTransform([
            PaddingTransform(word_padding, tag_padding),
            SplitTransform(), 
            ToArrayTransform()
        ])
        # init base handler
        super().__init__(sample_transform=self.sample_transform,
                         batch_transform=self.batch_transform)

    def parameters(self):
        """A copy of dataset parameters."""
        return self.params.copy()

    def load(self, name, size, encoding='utf8', batch_size=None, 
             shuffle=False, to_tensor=True, to_cuda=None):
        """
        Args:
            to_tensor (bool): enable numpy array to torch tensor.
            to_cuda (bool|device): `True` means use the current GPU device.
                `Flase` or `None` means use CPU device. `device` specifies 
                the destination GPU device.
        """
        dataset_dir = os.path.join(self.data_dir, name)
        msg = 'dataset directory {} not exists'.format(dataset_dir)
        assert os.path.isdir(dataset_dir), msg
        # dataset handler to read sample
        sentences_file = os.path.join(dataset_dir, 
                                      self.sentences_filename)
        labels_file = os.path.join(dataset_dir, 
                                   self.labels_filename)
        handler = NERHandler(
            input_file=sentences_file,
            target_file=labels_file,
            encoding=encoding
        )
        # update batch transform
        batch_transform = self.batch_transform.copy()
        if to_tensor:
            # Note: model's word-embedding requires `LongTensor`
            batch_transform.append(ToTensorTransform(dtype=torch.long,
                                                     to_cuda=to_cuda))
        return super().load(handler, name, size,
                            batch_size=batch_size, shuffle=shuffle,
                            batch_transform=batch_transform)