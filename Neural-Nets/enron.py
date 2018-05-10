# Use this file to read the enron email data

import os
import pickle
from itertools import chain
import random

import nltk
from nltk import tokenize
import numpy as np

ENRON_PATH = 'data/Enron/'
CACHE_PATH = 'data/cache/'

# Create these folders if they do not exist
os.makedirs(ENRON_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)


def load_text(enron_id, cache=True):
    """
    Reads all the files for every `enron_id data set` and
    returns a tuple of the form ([ham, spam]) where
    `ham` is a list of strings containing one ham email
    and `spam` is a list of strings each containing
    one spam email

    params
    enron_id : an integer from 1 to 6
    cache : A bool. If true, try loading the data from cache.
    If data is not stored in cache, load it and save it
    to the cache. If False, don't cache or load cache

    returns:
    (ham, spam) : list of strings of ham and spam
    """

    if not (isinstance(enron_id, int) and 1 <= enron_id <= 6):
        raise ValueError('enron_id must be an int from 1 to 6')

    if not (isinstance(cache, bool)):
        raise ValueError('cache must be True or False')

    cache_path = CACHE_PATH + 'enron{}'.format(enron_id)
    if cache and os.path.isfile(cache_path):
        with open(cache_path, 'rb') as fd:
            ham, spam = pickle.load(fd)
        return ham, spam

    ham = []
    spam = []

    ham_path = ENRON_PATH + 'enron{}/ham/'.format(enron_id)
    try:
        for filepath in os.listdir(ham_path):
            try:
                with open(ham_path + filepath, 'r') as fd:
                    txt = fd.read()
                    ham.append(txt)
            except UnicodeDecodeError:
                print('utf-8 error. Skipping file {}'.format(ham_path + filepath))
    except FileNotFoundError:
        print('No enron ham dataset found for id {}'.format(enron_id))
        raise

    spam_path = ENRON_PATH + 'enron{}/spam/'.format(enron_id)
    try:
        for filepath in os.listdir(spam_path):
            try:
                with open(spam_path + filepath, 'r') as fd:
                    txt = fd.read()
                    spam.append(txt)
            except UnicodeDecodeError:
                print('utf-8 error. Skipping file {}'.format(spam_path + filepath))
    except FileNotFoundError:
        print('No enron spam dataset found for id {}'.format(enron_id))
        raise

    if cache:
        with open(cache_path, 'wb') as fd:
            pickle.dump((ham, spam), fd)

    return (ham, spam)


def most_freq_words(all_text, n):
    """
    Given a iterable (or list) of strings, this functions returns a list
    of the `n` most frequent words. Using `itertools.chain` to combine
    multiple text sources. This function converts all words to lowercase
    before returning them.

    params:
    all_text : an iterable of strings
    n : An integer denoting the number of words to return

    returns:
    A list containing `n` words: the most frequent ones in the text.
    If `n` is lesser than the number of unique words in the text, all
    the words in the text are returned. The returned list is sorted.
    """

    freq_dist = nltk.FreqDist()

    for i, text in enumerate(all_text):
        words = tokenize.word_tokenize(text)

        l_words = [w.lower() for w in words
                   if w.isalnum() and not w.isnumeric()]

        freq_dist.update(l_words)

    top_n = [w[0] for w in freq_dist.most_common(n)]
    top_n.sort()
    return top_n


def vectorize_text(all_text, most_freq_words):
    """
    params:
    all_text : A list of strings; each string is a message that
           will be vectorized
    most_freq_words : An iterable of words which will be considered
           for vectorization

    returns:
    `V`, an n*m ndarray where n is `len(text)` and m is `len(most_freq_words)`.
    `V[i][j]==1.0` iff `most_freq_words[j]` is present in `all_text[i]`. Else,
    `V[i][j]==0.0`
    """

    n = len(all_text)
    m = len(most_freq_words)

    V = np.zeros([n, m], dtype='float32')
    for i, text in enumerate(all_text):
        words = tokenize.word_tokenize(text)
        l_words = {w.lower() for w in words}

        for j, mfw in enumerate(most_freq_words):
            if mfw in l_words:
                V[i][j] = 1.0

    return V


def vectorize_class(all_text, is_spam):
    """
    Takes a list of strings, and whether or not it is spam.
    Returns a vectorized numpy version

    params:
    all_text : A list of strings
    is_spam : Either True or False

    returns:
    `V`, a n*2 ndarray where n=len(all_text)
    `V[i][0]`==1.0 iff `all_text[i]` is NOT spam
    `V[i][1]`==1.0 iff `all_text[i]` is spam
    """

    n = len(all_text)
    V = np.zeros([n, 2])

    if is_spam:
        V[:, 1] = 1.0
    else:
        V[:, 0] = 1.0
    return V


def shared(batch):
    """
    Takes a tuple of the form (X, Y), where both X and Y
    are numpy ndarrays, and converts them into a form 
    compatible with theano

    params:
    batch : (X,Y) tuple where both X and Y are ndarrays

    returns
    (shared_X, shared_Y) equivalent theano-shared arrays
    """
    import theano
    # np.asarray : Convert the input to an array
    # a : data of any data type which can be converted to arrays
    # dtype : data type
    shared_X = theano.shared(
        np.asarray(a=batch[0], dtype=theano.config.floatX),
        borrow=True)
    shared_Y = theano.shared(
        np.asarray(batch[1], dtype=theano.config.floatX),
        borrow=True)
    return shared_X, shared_Y


def shuffle_xy(X, Y):
    """
    Shuffles the rows of X and Y, ensuring that every row
    points to its actual class label in Y. Shuffling is done
    in place.

    params:
    X : An ndarray
    Y : An ndarray
    X and Y must have the same number of rows

    returns:
    X,Y : Shuffled ndarrays
    """
    n = X.shape[0]
    for i in range(n):
        j = random.randint(0, n - 1)

        # Swapping two rows in numpy
        X[[i, j]] = X[[j, i]]
        Y[[i, j]] = Y[[j, i]]
    return X, Y


def load_enron_vectorized(enron_ids, n, shuffle=False, cache=True, theano_shared=True):
    """
    Completely load the email dataset for all the ids in `enron_ids`.
    `n` is the number of words to consider for `most_freq_words`.
    Concretely, this function does the following:

    * Call `load_text` for every id in `enron_ids`
    * Call `most_freq_words` on the COMBINED text collected
    * Vectorize the text and classes
    * OPTIONALLY cache the loaded data
    * OPTIONALLY shuffle the rows around
    * OPTIONALLY return a theano shared version of the data instead of
      numpy ndarrays

    params:
    enron_ids : A list of ints from 1 to 6. The order is preserved
                as well as used for caching
    n         : Number of words to consider for most_freq_words
    shuffle   : Whether or not to shuffle the rows. Shuffling is done
                after loading the data, so the cached data is always
                in order
    cache     : Whether or not to cache/load from cache
    theano_shared : Whether or not to return a theano-shared version
                    of the data

    returns:
    A tuple of the form (X, Y) where X is the vectorized text and Y
    is the vectorized classes. If theano_shared is True, a theano
    shared version of the tuple is returned
    """

    cache_path = CACHE_PATH + 'enron_vec_{}_{}'.format(
        ''.join(map(str, enron_ids)), n)

    if cache and os.path.isfile(cache_path):
        with open(cache_path, 'rb') as fd:
            X, Y = pickle.load(fd)
    else:
        enrons = [load_text(enron_id, cache=cache) for enron_id in enron_ids]
        flattened_enron = chain.from_iterable(chain.from_iterable(enrons))
        mfw = most_freq_words(flattened_enron, n)

        count = sum((len(h) + len(s) for h, s in enrons))  # number of emails

        X = np.ndarray([count, n])
        Y = np.ndarray([count, 2])

        i = 0
        for ham, spam in enrons:
            X[i:i + len(ham), :] = vectorize_text(ham, mfw)
            Y[i:i + len(ham), :] = vectorize_class(ham, is_spam=False)
            i += len(ham)

            X[i:i + len(spam), :] = vectorize_text(spam, mfw)
            Y[i:i + len(spam), :] = vectorize_class(spam, is_spam=True)
            i += len(spam)

        if cache:
            with open(cache_path, 'wb') as fd:
                pickle.dump((X, Y), fd)

    if shuffle:
        shuffle_xy(X, Y)

    if theano_shared:
        X, Y = shared((X, Y))

    return X, Y
