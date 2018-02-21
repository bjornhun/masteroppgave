# make list of paths


import os
import random

data_path = "data/"

def preprocess(neg_per_pos = 1, test_ratio=.1, val_ratio=.1, noise_ratio=.5, wakeword="seven"):
    wakeword_path = data_path + wakeword
    classes = [e for e in os.listdir(data_path) if os.path.isdir(data_path + e) and not e.startswith("_")]
    positives = [(wakeword + "/" + p) for p in os.listdir(wakeword_path)]
    
    n_classes = len(classes)

    n_positives = len(positives)
    n_negatives = n_positives * neg_per_pos
    n_samples = n_positives + n_negatives

    n_train = int(n_samples * (1-(test_ratio + val_ratio)))
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)

    n_negatives_per_class = int(n_negatives/n_classes)
    negatives = []
    for c in classes:
        [negatives.append(c + "/" + n) for n in random.sample(os.listdir(data_path + c), n_negatives_per_class)]

if __name__ == "__main__":
    preprocess()