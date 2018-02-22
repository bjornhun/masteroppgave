import os
import random
from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
import pickle

wakeword = "seven"
data_path = "data/"

def get_positives(wakeword):
    return [(wakeword + "/" + p) for p in os.listdir(data_path + wakeword)]

def get_negatives(n_negatives):
    classes = [e for e in os.listdir(data_path) if os.path.isdir(data_path + e) and not e.startswith("_") and e!=wakeword]
    n_classes = len(classes)
    n_negatives_per_class = int(n_negatives/n_classes)
    negatives = []
    for c in classes:
        [negatives.append(c + "/" + n) for n in random.sample(os.listdir(data_path + c), n_negatives_per_class)]
    return negatives

def get_subsample(positives, negatives, n_pos, n_neg):
    subsample = []

    for i in range(n_pos):
        subsample.append(positives.pop(0))
    for i in range(n_neg):
        subsample.append(negatives.pop(0))

    return positives, negatives, subsample

def train_test_val_split(positives, negatives, test_ratio, val_ratio):
    n_test_pos = int(len(positives) * test_ratio)
    n_val_pos = int(len(positives) * val_ratio)
    n_train_pos = len(positives) - (n_test_pos + n_val_pos)

    n_test_neg = int(len(negatives) * test_ratio)
    n_val_neg = int(len(negatives) * val_ratio)
    n_train_neg = len(negatives) - (n_test_neg + n_val_neg)

    positives, negatives, test = get_subsample(positives, negatives, n_test_pos, n_test_neg)
    positives, negatives, val = get_subsample(positives, negatives, n_val_pos, n_val_neg)
    positives, negatives, train = get_subsample(positives, negatives, n_train_pos, n_train_neg)

    return train, test, val

def get_data_paths(neg_per_pos=1, test_ratio=.1, val_ratio=.1):
    positives = get_positives(wakeword)
    negatives = get_negatives(len(positives) * neg_per_pos)
    return train_test_val_split(positives, negatives, test_ratio, val_ratio)

def set_length(x, cutoff):
    num_samples = len(x)
    if num_samples > cutoff:
        return x[:cutoff]
    else:
        zeros = cutoff - num_samples
        return np.append(x, [0]*zeros)

def read_wav(path):
    fs, x = wavfile.read(data_path + path)
    if len(x) != fs:
        x = set_length(x, fs)
    return x

def normalize(coeff):
    coeff += np.abs(coeff.min())
    coeff /= coeff.max()
    return coeff

def get_features(path):
    wav_data = read_wav(path)
    return normalize(mfcc(wav_data))

def get_label(path):
    if path.startswith(wakeword):
        return 1
    return 0

def get_features_and_labels(paths):
    X = []
    y = []
    for path in paths:
        X.append(get_features(path))
        y.append(get_label(path))
    return X, y


def add_noise(noise_ratio=.5):
    pass

if __name__ == "__main__":
    train_paths, test_paths, val_paths = get_data_paths()
    X_train, y_train = get_features_and_labels(train_paths)
    X_test, y_test = get_features_and_labels(test_paths)
    X_val, y_val = get_features_and_labels(val_paths)

    pickle.dump((X_train, y_train), open("data/train.p", "wb"))
    pickle.dump((X_test, y_test), open("data/test.p", "wb"))
    pickle.dump((X_val, y_val), open("data/val.p", "wb"))