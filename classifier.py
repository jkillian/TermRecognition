import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit

from gmm import gmmhmm
from features import logfbank


class Classifier(object):

    def __init__(self):
        pass

    def read_train_files(self):
        """Return
            - a list of filenames from the data directory
            - a list of labels for those filenames
            - a list of ints representing those labels
            in a tuple"""

        fpaths = []
        labels = []

        for fname in glob('audio/*'):
            fpaths.append(fname)
            label = fname.split('/')[1][:-6]
            labels.append(label)

        # int_labels is an array of numbers with the correct class of each row in feat, as an int
        int_labels = np.zeros(len(labels))
        for ind, label in enumerate(set(labels)):
            int_labels[np.array([i for i, _ in enumerate(labels) if _ == label])] = ind

        print("Read {num_files} files".format(num_files=len(int_labels)))
        return (fpaths, labels, int_labels)

    def make_features(self, fpaths):
        """Given a list of filenames, return an np array of the features for those filenames"""

        feats = []
        for fname in fpaths:
            mfccs = []
            rate, sig = wav.read(fname)
            sig = np.array(sig)
            sig = np.pad(sig, (0, 2.1*rate - len(sig)), mode='constant')
            fb = np.array(logfbank(sig, rate))
            feats.append(np.transpose(fb))

        # feats[i] = ith audio file
        # feats[i][j] = jth fbank feature of frame
        # feats[i][j][k] = kth frame of audio file
        feats = np.atleast_3d(feats)
        
        # The HMM expects probabilities, so we normalize.
        # Without it, we get random guessing
        for index, val in enumerate(feats):
            feats[index] /= feats[index].sum(axis=0)

        return feats

    def make_models(self, X_train, y_train, word_set):
        """Train self.models on X_train with y_train as the correct labels (ints) for X_train.
            word_set (ints) is the list of potential words to train on."""
        num_states = 3
        self.models = [gmmhmm(num_states) for word in word_set]
        self.models = [model.fit(X_train[y_train == word, :, :]) for model, word in zip(self.models, word_set)]

    def predict(self, X_test, y_test):

        ps = [model.transform(X_test) for model in self.models]
        res = np.vstack(ps)
        predicted_labels = np.argmax(res, axis=0)
        return predicted_labels
        

    def train(self, plot=False):
        
        fnames, labels, int_labels = self.read_train_files()

        feats = self.make_features(fnames)        

        sss = StratifiedShuffleSplit(int_labels, test_size=0.8, random_state=0)

        for train_index, test_index in sss:
            # train_index: list of indices, len = (1-test_size)*num_audio_files
            # test_index: inverse of train_index

            X_train, X_test = feats[train_index, ...], feats[test_index, ...]
            y_train, y_test = int_labels[train_index], int_labels[test_index]

            self.make_models(X_train, y_train, set(int_labels))
            print("models trained!")
            predicted_labels = self.predict(X_test, y_test)
            missed = (predicted_labels != y_test)
            print('Test accuracy:%.2f percent' % (100 * (1 - np.mean(missed))))
            if plot:
                self.plot(y_test, predicted_labels, list(set(labels)))
            break


    def plot(self, y_test, predicted_labels, word_set):
        cm = confusion_matrix(y_test, predicted_labels)
        plt.matshow(cm, cmap='gray')
        ax = plt.gca()
        _ = ax.set_xticklabels([" "] + [l[4:6] for l in word_set])
        _ = ax.set_yticklabels([" "] + word_set)
        plt.title('Confusion matrix, single speaker')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()