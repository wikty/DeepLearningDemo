import os
import csv
import argparse

import config
from lib.utils import Logger, Params, Counter
from lib.dataset.builder import get_parser, BaseBuilder
from lib.experiment import DatasetCfg


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
    min_count_word = config.min_count_word
    min_count_tag = config.min_count_tag
    dataset_cfg = DatasetCfg(config.data_dir)
    data_file = os.path.join(dataset_cfg.data_dir(), 'ner_dataset.csv')
    
    # cmd parser
    def toint(x):
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError("%r must be greater then 0" % x)
        return x

    parser = get_parser(
        data_dir=dataset_cfg.data_dir(),
        train_factor=dataset_cfg.train_factor(),
        val_factor=dataset_cfg.val_factor(),
        test_factor=dataset_cfg.test_factor(),
        train_name=dataset_cfg.train_name(),
        val_name=dataset_cfg.val_name(),
        test_name=dataset_cfg.test_name()
    )

    parser.add_argument('--data-file', 
        default=data_file,
        help="File of data source")
    parser.add_argument('--min-count-word', 
        default=min_count_word, 
        help="Minimum count for words in the dataset(default: %(default)s)", 
        type=toint)
    parser.add_argument('--min-count-tag', 
        default=min_count_tag, 
        help="Minimum count for tags in the dataset(default: %(default)s)", 
        type=toint)

    args = parser.parse_args()

    msg = 'Data file {} not found.'
    assert os.path.isfile(args.data_file), msg.format(args.data_file)
    msg = '{} directory not found. Please create it first.'
    assert os.path.isdir(args.data_dir), msg.format(args.data_dir)
    msg = 'the proportion of dataset to builded must in (0.0, 1.0]'
    assert (args.data_factor > 0.0) and (args.data_factor <= 1.0), msg
    msg = 'train factor + val factor + test factor must be equal to 1.0'
    total = args.train_factor + args.val_factor + args.test_factor
    assert (1.0 == total), msg

    dataset_cfg.set_data_dir(args.data_dir)

    # set and get logger
    logger = Logger.set(dataset_cfg.log_file())

    # build, load and dump datasets
    builder = Builder(data_factor=args.data_factor,
                      train_factor=args.train_factor, 
                      val_factor=args.val_factor,
                      test_factor=args.test_factor,
                      train_name=args.train_name,
                      val_name=args.val_name,
                      test_name=args.test_name,
                      logger=logger)
    builder.load(args.data_file, 
                 encoding='windows-1252')
    builder.dump(dataset_cfg.data_dir(), dataset_cfg.params_file(),
                 min_count_word=args.min_count_word,
                 min_count_tag=args.min_count_tag,
                 sentences_filename=config.sentences_filename,
                 labels_filename=config.labels_filename,
                 words_filename=config.words_filename,
                 tags_filename=config.tags_filename,
                 encoding=config.data_file_encoding)


