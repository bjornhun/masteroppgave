from model import *
import pyaudio
import winsound 
import time

fs = 16000
framesize = 4000
seq_len = int(float(framesize)/float(fs)*100-1)
threshold = 0.9999

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=framesize)

def record():
    data = stream.read(framesize)
    raw_speech = np.fromstring(data, dtype=np.int16)
    return normalize(mfcc(raw_speech, numcep=26))

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")
    cooldown = 0
    count = 0
    probs = np.array([])
    st = np.zeros((n_layers, 1, n_neurons))
    while True:
        rec_start = time.time()
        coeff = record()
        rec_end = time.time()
        #print("Recording:", rec_end-rec_start)
        start = time.time()
        st, wp = sess.run([states, wakeword_probs], feed_dict={X: [coeff], initial_state: st, seq_length: seq_len})
        for val in wp:
            probs = np.append(probs, val)

        probs = probs[-seq_len:]
        detections = probs > threshold
        if True in detections:
            count+=1
            print("Wake word detected #", count)
            #winsound.Beep(1000, 300)
            st = np.zeros((n_layers, 1, n_neurons))
        end = time.time()
        #print("Processing:", end-start)

stream.stop_stream()
stream.close()
p.terminate()