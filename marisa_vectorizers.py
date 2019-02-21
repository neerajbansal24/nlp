# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import marisa_trie
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# hack to store vocabulary in MARISA Trie
class _MarisaVocabularyMixin(object):

    def fit_transform(self, raw_documents, y=None):
        super(_MarisaVocabularyMixin, self).fit_transform(raw_documents)
        self._freeze_vocabulary()
        return super(_MarisaVocabularyMixin, self).fit_transform(raw_documents, y)
        
    def _freeze_vocabulary(self):
        if not self.fixed_vocabulary_:
            self.vocabulary_ = marisa_trie.Trie(self.vocabulary_.keys())
            self.fixed_vocabulary_ = True
            del self.stop_words_
            

class MarisaCountVectorizer(_MarisaVocabularyMixin, CountVectorizer):
    pass


class MarisaTfidfVectorizer(_MarisaVocabularyMixin, TfidfVectorizer):
    def fit(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit(raw_documents, y)