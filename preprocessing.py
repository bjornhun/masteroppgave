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

def preprocess(neg_per_pos=1, test_ratio=.1, val_ratio=.1, noise_ratio=.5):
    positives = get_positives(wakeword)
    negatives = get_negatives(len(positives) * neg_per_pos)

if __name__ == "__main__":
    preprocess()