import os

import numpy as np


class Vocab(object):
    """Vocab maps a string to integer or one-hot vector.
    
    Note: The index of vocab start from zero, and auto-increased when new
    term is added into vocab.
    """

    def __init__(self, raw_txt=None, encoding='utf8', 
                 one_hot=False, case_sensitive=True):
        """
        Args:
            raw_txt (path or list): a path for text file or a list of tokens.
        """
        self.one_hot = one_hot
        self.case_sensitive = case_sensitive
        self.terms = {}
        self.rterms = {}
        if raw_txt:
            self.build(raw_txt, encoding)

    def __len__(self):
        return len(self.terms)

    def size(self):
        return len(self)

    def process(self, term):
        """The term processor."""
        # to string
        term = str(term)
        # to lower string
        if not self.case_sensitive:
            term = term.lower()
        return term

    def clear(self):
        """Clear vocab.
        
        Returns: return self
        """
        self.terms = {}
        self.rterms = {}
        return self

    def build(self, raw_txt, encoding='utf8'):
        """Build a new vocab.

        Args:
            raw_txt (path or list): if it's a text file, the tokens split by 
                space or end-of-line. Or it's a list of tokens. 

        Returns: return self
        """
        assert (isinstance(raw_txt, list) or os.path.isfile(raw_txt))
        self.clear()
        terms = raw_txt
        if os.path.isfile(raw_txt):
            with open(raw_txt, 'r', encoding=encoding) as f:
                terms = f.read().split()
        for term in terms:
            self.add(term)
        return self

    def add(self, term):
        """
        Args:
            term (str): the term string.

        Return: return self
        """
        term = self.process(term)
        i = self.terms.setdefault(term, len(self.terms))
        self.rterms[i] = term
        return self

    def encode(self, term, default=None):
        """Term to integer or one-hot encoding."""
        term = self.process(term)
        i = self.terms.get(term, default)
        if i == default:
            return default
        if self.one_hot:
            a = np.zeros(self.size())
            a[i] = 1.0
            return a
        return i

    def decode(self, i, default=None):
        """Integer or one-hot encoding to the term."""
        if self.one_hot:
            i = np.argmax(i)
        return self.rterms.get(i, default)