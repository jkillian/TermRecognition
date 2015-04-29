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
        self.fpaths = []
        labels = []

        for fname in glob('data/*'):
            self.fpaths.append(fname)
            label = fname.split('/')[1][:-6]
            labels.append(label)

        # self.int_labels is an array of numbers with the correct class of each row in feat, as an int
        self.int_labels = np.zeros(len(labels))
        for ind, label in enumerate(set(labels)):
            self.int_labels[np.array([i for i, _ in enumerate(labels) if _ == label])] = ind

        print "Read {num_files} files".format(num_files=len(self.int_labels))

    def make_features(self):

        self.feat = []
        for fname in self.fpaths:
            mfccs = []
            rate, sig = wav.read(fname)
            sig = np.array(sig)
            sig = np.pad(sig, (0, 2.1*rate - len(sig)), mode='constant')
            fb = np.array(logfbank(sig, rate))
            self.feat.append(np.transpose(fb))

        # self.feat[i] = ith audio file
        # self.feat[i][j] = jth fbank feature of frame
        # self.feat[i][j][k] = kth frame of audio file
        self.feat = np.atleast_3d(self.feat)
        
        # The HMM expects probabilities, so we normalize.
        # Without it, we get random guessing
        for index, val in enumerate(self.feat):
            self.feat[index] /= self.feat[index].sum(axis=0)


    def train(self):
        
        self.read_train_files()

        self.make_features()        

        sss = StratifiedShuffleSplit(self.int_labels, test_size=0.2, random_state=0)

        for num_states in range(2, 10):
            for train_index, test_index in sss:
                # train_index: list of indices, len = (1-test_size)*num_audio_files
                # test_index: inverse of train_index
                X_train, X_test = self.feat[train_index, ...], self.feat[test_index, ...]
                y_train, y_test = self.int_labels[train_index], self.int_labels[test_index]

            ys = set(self.int_labels)
            models = [gmmhmm(num_states) for y in ys]
            _ = [model.fit(X_train[y_train == y, :, :]) for model, y in zip(models, ys)]
            ps = [model.transform(X_test) for model in models]
            res = np.vstack(ps)
            predicted_labels = np.argmax(res, axis=0)
            missed = (predicted_labels != y_test)
            print '%d Test accuracy:%.2f percent' % (num_states, (100 * (1 - np.mean(missed))))


    def plot(self):
        cm = confusion_matrix(y_test, predicted_labels)
        plt.matshow(cm, cmap='gray')
        ax = plt.gca()
        _ = ax.set_xticklabels([" "] + [l[:2] for l in spoken])
        _ = ax.set_yticklabels([" "] + spoken)
        plt.title('Confusion matrix, single speaker')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()