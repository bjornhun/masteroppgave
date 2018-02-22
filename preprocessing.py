import os
import random

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
        subsample.append((positives.pop(), 1))
    for i in range(n_neg):
        subsample.append((negatives.pop(), 0))

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

def preprocess(neg_per_pos=1, test_ratio=.1, val_ratio=.1, noise_ratio=.5):
    positives = get_positives(wakeword)
    negatives = get_negatives(len(positives) * neg_per_pos)
    train, test, val = train_test_val_split(positives, negatives, test_ratio, val_ratio)

if __name__ == "__main__":
    preprocess()