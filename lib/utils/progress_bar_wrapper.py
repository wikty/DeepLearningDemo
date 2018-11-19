class ProgressBarWrapper(object):
    """A wrapper of iterator, shows the progress bar of iteration."""

    def __init__(self, iterator, iterator_len, with_bar=True, 
                 with_index=True, prefix='', suffix='', decimals=1, 
                 bar_len=50, fill='@', padding='-'):
        """
        Args:
            iterator (iterator): the data iterator.
            iterator_len (int): the length of the iterator.
            prefix (str): the prefix string for progress bar.
            suffix (str): the suffix string for progress bar.
            decimals (int): the number of decimals in percent.
            bar_len (int): the length of progress bar.
            fill (str): bar fill character.
            padding (str): bar padding character.
        """
        self.iterator = iterator
        self.iterator_len = iterator_len
        self.with_index = with_index
        self.with_bar = with_bar
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.bar_len = bar_len
        self.fill = fill
        self.padding = padding

    def set_prefix(self, prefix=''):
        self.prefix = prefix

    def set_suffix(self, suffix=''):
        self.suffix = suffix

    def print_bar(self, i):
        total = self.iterator_len
        percent = (i / float(total)) if total != 0 else 1.0
        fill_len = int(self.bar_len * percent)
        padding_len = self.bar_len - fill_len
        percent_fmt = ''.join([
            '{', '0:.{}f'.format(self.decimals), '}'
        ])
        percent_str = percent_fmt.format(100 * percent)
        bar_str = ''.join([self.fill]*fill_len + [self.padding]*padding_len)
        print()
        print('{} [{}] {}%% {}'.format(
            self.prefix, bar_str, percent_str, self.suffix))

    def __iter__(self):
        """Wrap the original iterator.

        Returns: a tuple of (index, bar, item)
        """
        for i, item in enumerate(self.iterator, 1):
            self.print_bar(i)
            if self.with_index and self.with_bar:
                yield i-1, self, item
            elif self.with_index and (not self.with_bar):
                yield i-1, item
            elif self.with_bar and (not self.with_index):
                yield self, item
            else:
                yield item
