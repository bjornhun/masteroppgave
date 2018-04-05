from model import *
import pyaudio
import winsound

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=16000)
    data = stream.read(16000)
    raw_speech = np.fromstring(data, dtype=np.int16)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return normalize(mfcc(raw_speech, numcep=26))

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")
    cooldown = 0
    count = 0
    probs = np.array([])
    st = np.zeros((n_layers, 1, n_neurons))
    while True:
        coeff = record()
        st, wp = sess.run([states, wakeword_probs], feed_dict={X: [coeff], initial_state: st, seq_length : 99})
        for val in wp:
            probs = np.append(probs, val)

        probs = probs[-99:]
        threshold = 0.99
        detections = probs > threshold
        if cooldown == 0:
            if True in detections:
                count+=1
                print("Wake word detected #", count)
                winsound.Beep(1000, 300)
                cooldown = 3
        else:
            cooldown -= 1