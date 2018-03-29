import os
from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
import pickle
from preprocessing import read_wav, normalize
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__)) + "\\"
data_path = path + "data\\"

n_mfccs = 26

if __name__ == "__main__":
    files = ["dog\\0ac15fe9_nohash_0.wav",
             "seven\\0b77ee66_nohash_0.wav",
             "go\\0ac15fe9_nohash_1.wav",
             "on\\0c5027de_nohash_0.wav",
             "four\\0b77ee66_nohash_0.wav",
             "one\\0c40e715_nohash_0.wav",
             "stop\\0cd323ec_nohash_0.wav",
             "seven\\0c2d2ffa_nohash_0.wav",
             "tree\\0d53e045_nohash_0.wav",
             "cat\\0c5027de_nohash_0.wav",
             "dog\\0ac15fe9_nohash_0.wav",
             "seven\\0b77ee66_nohash_0.wav",
             "go\\0ac15fe9_nohash_1.wav",
             "on\\0c5027de_nohash_0.wav",
             "four\\0b77ee66_nohash_0.wav",
             "one\\0c40e715_nohash_0.wav",
             "stop\\0cd323ec_nohash_0.wav",
             "seven\\0c2d2ffa_nohash_0.wav",
             "tree\\0d53e045_nohash_0.wav",
             "cat\\0c5027de_nohash_0.wav"]
    
    speech = []
    for f in files:
        x = read_wav(f, False)
        for val in x:
            speech.append(val)
    speech = np.asarray(speech, dtype=np.int16)
    out_file = normalize(mfcc(speech, numcep=n_mfccs))
    print(speech)

    pickle.dump((out_file), open(data_path + "concat_file.p", "wb"))
    plt.plot(speech, range(len(speech)))
    plt.show()
    wavfile.write("banana.wav", 16000, speech)