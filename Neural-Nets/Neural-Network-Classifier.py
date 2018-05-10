import os
import pickle
from itertools import chain

from neural import *
import enron


def block_input():
    """
    Reads user input into a string until the user
    presses Ctrl+C

    returns:
    The string read
    """

    text = ''
    try:
        while True:
            print(' >', end=' ')
            line = input().strip()
            text = text + '\n' + line
    except KeyboardInterrupt:
        return text


def load_mfw(enron_ids, n):
    """
    load_mfw = load most frequent words
    :param enron_ids: [1, 2, 3, 4]
    :param n: 150

    :return: mfw
    """
    cache_path = 'data/cache/enron_mfw_{}_{}'.format(''.join(map(str, enron_ids)), n)
    if os.path.isfile(cache_path):
        print('Found most frequent words cached. Unpickling...')
        with open(cache_path, 'rb') as fd:
            mfw = pickle.load(fd)
        return mfw

    print('Finding {} most frequent words...'.format(n))
    enrons = [enron.load_text(enron_id, cache=True) for enron_id in enron_ids]
    flattened_enron = chain.from_iterable(chain.from_iterable(enrons))
    mfw = enron.most_freq_words(flattened_enron, n)

    with open(cache_path, 'wb') as fd:
        pickle.dump(mfw, fd)

    return mfw


enron_ids = [1, 2, 3, 4]  # List of enron datasets to load
mini_batch_size = 20  # Mini batch size for NeuralNetwork
n = 150  # Number of words to consider
train_fraction = 0.75  # fraction of enron data used for testing

print('enron_ids: ', enron_ids)
print('Considering top-{} words'.format(n))

print('Loading enron data')
ed = enron.load_enron_vectorized(enron_ids=enron_ids, n=n,
                                 shuffle=True, theano_shared=False)
N = ed[0].shape[0]
print('Loaded {} messages from enron data'.format(N))

train_fraction = round(N * train_fraction)

training_data = enron.shared((ed[0][:train_fraction, :], ed[1][:train_fraction, :]))
test_data = enron.shared((ed[0][train_fraction:, :], ed[1][train_fraction:, :]))

print('Using {} messages as training data'.format(train_fraction))

net = NeuralNetwork(
    layers=[FullyConnectedLayer(n, 80, activation_fn=softmax),
            FullyConnectedLayer(80, 2, activation_fn=softmax),
            CrossEntropyCostLayer(2)])

print('Initialized Network:')
print(net, '\n')


def activate_session():
    mfw = None
    try:
        while True:
            print('Neural Nets Spam Classifer')
            print('')
            print('Enter 1 for train and 2 for testing the classifier')
            print('1] Train')
            print('2] Test ')
            print(' >', end=' ')

            try:
                choice = int(input())
            except ValueError:
                print('Invalid choice!')
                continue

            if choice == 1:
                print('\nEnter epochs and eta separated by a space.')
                print('(Leave blank for epochs=60, eta=0.3)')
                print(' >', end=' ')
                i = input().strip()
                if i == '':
                    epochs, eta = 60, 0.3
                else:
                    i = i.split()
                    epochs = int(i[0])
                    eta = float(i[1])

                net.SGD(training_data, epochs, eta, mini_batch_size, test_data)
                print()
            elif choice == 2:
                print('\nEnter text for testing as Spam/Ham. Press Ctrl+C to end the message')
                text = block_input()
                print('\n')

                if mfw is None:
                    mfw = load_mfw(enron_ids, n)

                X = enron.vectorize_text([text], mfw)
                a = np.argmax(net.feedforward(X), axis=1)
                if a[0] == 0:
                    print('Net says: HAM')
                else:
                    print('Net says: SPAM')
            else:
                print('Invalid choice!')

    except KeyboardInterrupt:
        print('terminating...')

    print('Session terminated')


if __name__ == '__main__':
    activate_session()
