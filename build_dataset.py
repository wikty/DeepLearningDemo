import os
import csv
import argparse

import config
from lib.utils import Logger, Params, Counter
from lib.dataset import BaseBuilder


class Builder(BaseBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.PAD_WORD = '<pad>'  # pad word
        self.UNK_WORD = 'UNK'  # unknow word
        self.PAD_TAG = 'O'  # pad tag

    def load(self, csv_file, encoding='utf8'):
        self.logger.info('Loading dataset from csv file...')
        with open(csv_file, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            words, tags = [], []
            for row in reader:
                if row['Sentence #'].strip() and len(words):
                    assert len(words) == len(tags)
                    self.add_sample(words=words, tags=tags)
                    words, tags = [], []
                try:
                    word, tag = str(row['Word']), str(row['Tag'])
                except UnicodeDecodeError as e:
                    msg = "An exception was raised, skipping a word: {}"
                    self.logger.warning(msg.format(e))
                    pass
                else:
                    words.append(word)
                    tags.append(tag)
            if len(words) > 0:
                assert len(words) == len(tags)
                self.add_sample(words=words, tags=tags)
        self.logger.info('The number of samples is {}'.format(
            len(self.samples)))
        self.logger.info('- done!')

    def dump(self, data_dir, params_file,
            sentences_filename='sentences.txt', labels_filename='labels.txt', 
            words_filename='words.txt', tags_filename='tags.txt',
            encoding='utf8', shuffle=True, min_count_word=1, min_count_tag=1):
        # datasets params
        params = Params(data={
            'word_vocab_size': 0,
            'tag_vocab_size': 0,
            'pad_word': self.PAD_WORD,
            'unk_word': self.UNK_WORD,
            'pad_tag': self.PAD_TAG
        })
        # dataset and vocab
        datasets = self.build(shuffle=shuffle)
        tag_vocab = Counter([self.PAD_TAG])
        word_vocab = Counter([self.PAD_WORD, self.UNK_WORD])
        # save train/val/test dataset
        for dataset in datasets:
            name = dataset.name
            size = len(dataset)
            self.logger.info('Saving {} dataset...'.format(name))
            params.set('{}_size'.format(name), size)  # set dataset size
            dirpath = os.path.join(data_dir, name)
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            sentences_file = os.path.join(dirpath, sentences_filename)
            labels_file = os.path.join(dirpath, labels_filename)
            with open(sentences_file, 'w', encoding=encoding) as fs, \
                open(labels_file, 'w', encoding=encoding) as fl:
                for sample in dataset:
                    words, tags = sample.words, sample.tags
                    fs.write('{}\n'.format(' '.join(words)))
                    fl.write('{}\n'.format(' '.join(tags)))
                    tag_vocab.update(tags)
                    word_vocab.update(words)          
            self.logger.info('- done!')
        # save word vocab
        self.logger.info('Saving word vocab...')
        word_vocab_file = os.path.join(data_dir, 
                                       words_filename)
        with open(word_vocab_file, 'w', encoding=encoding) as f:
            for word in word_vocab.get(min_count=min_count_word):
                f.write('{}\n'.format(word))
        params.word_vocab_size = word_vocab.size(min_count=min_count_word)
        self.logger.info('- done!')
        # save tag vocab
        self.logger.info('Saving tag vocab...')
        tag_vocab_file = os.path.join(data_dir, 
                                      tags_filename)
        with open(tag_vocab_file, 'w', encoding=encoding) as f:
            for tag in tag_vocab.get(min_count=min_count_tag):
                f.write('{}\n'.format(tag))
        params.tag_vocab_size = tag_vocab.size(min_count=min_count_tag)
        self.logger.info('- done!')
        # save datasets parameters
        self.logger.info('Saving datasets parameters...')
        params.dump(params_file, encoding=encoding)
        self.logger.info('- done!')
        # print dataset characteristics
        self.logger.info("Characteristics of the dataset:")
        for key, value in params:
            self.logger.info("- {}: {}".format(key, value))


if __name__ == '__main__':
    # default settings
    data_dir = config.data_dir
    data_file = os.path.join(data_dir, 'ner_dataset.csv')
    datasets_log_file = config.datasets_log_file
    datasets_params_file = config.datasets_params_file
    min_count_word, min_count_tag = 1, 1
    train_factor, val_factor, test_factor = (config.train_factor,
                                             config.val_factor,
                                             config.test_factor)
    train_name, val_name, test_name = (config.train_name,
                                       config.val_name,
                                       config.test_name)

    # cmd parser
    def tofloat(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in [0.0, 1.0]" % x)
        return x

    def toint(x):
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError("%r must be greater then 0" % x)
        return x

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', 
        default=data_file,
        help="File of data source")
    parser.add_argument('--data-dir', 
        default=data_dir,
        help="Directory for the dataset to save")
    parser.add_argument('--data-factor',
        default=1.0,
        help="The proportion of dataset to be builded(default: %(default)s)",
        type=tofloat)
    parser.add_argument('--train-factor', 
        default=train_factor,
        help="The factor of train dataset", 
        type=tofloat)
    parser.add_argument('--val-factor', 
        default=val_factor,
        help="The factor of validation dataset", 
        type=tofloat)
    parser.add_argument('--test-factor', 
        default=test_factor,
        help="The factor of test dataset", 
        type=tofloat)
    parser.add_argument('--min-count-word', 
        default=min_count_word, 
        help="Minimum count for words in the dataset(default: %(default)s)", 
        type=toint)
    parser.add_argument('--min-count-tag', 
        default=min_count_tag, 
        help="Minimum count for tags in the dataset(default: %(default)s)", 
        type=toint)

    args = parser.parse_args()

    data_file, data_dir = args.data_file, args.data_dir
    data_factor = args.data_factor
    train_factor = args.train_factor
    val_factor = args.val_factor
    test_factor = args.test_factor
    min_count_word = args.min_count_word
    min_count_tag = args.min_count_tag
    msg = '{} file not found. Make sure you have downloaded it.'
    assert os.path.isfile(data_file), msg.format(data_file)
    msg = '{} directory not found. Please create it first.'
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = 'the proportion of dataset to builded must in (0.0, 1.0]'
    assert (data_factor > 0.0) and (data_factor <= 1.0), msg
    msg = 'train factor + val factor + test factor must be equal to 1.0'
    assert (1.0 == (train_factor + val_factor + test_factor)), msg

    # set and get logger
    logger = Logger.set(datasets_log_file)

    # build, load and dump datasets
    builder = Builder(data_factor=data_factor,
                      train_factor=train_factor, 
                      val_factor=val_factor,
                      test_factor=test_factor,
                      train_name=train_name,
                      val_name=val_name,
                      test_name=test_name,
                      logger=logger)
    builder.load(data_file, 
                 encoding='windows-1252')
    builder.dump(data_dir, datasets_params_file,
                 encoding='utf8',
                 min_count_word=min_count_word,
                 min_count_tag=min_count_tag)


