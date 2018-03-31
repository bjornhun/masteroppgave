import tensorflow as tf
import numpy as np
import pickle
import os
from scipy.io import wavfile
from python_speech_features import mfcc
from preprocessing import normalize
import matplotlib.pyplot as plt

model_name = "single_layer_gru_10-90"
train_model = False

data_path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"

X_train, y_train = pickle.load(open(data_path + "train.p", "rb"))
X_test, y_test = pickle.load(open(data_path + "test.p", "rb"))
X_val, y_val = pickle.load(open(data_path + "val.p", "rb"))

n_train = len(y_train)
n_val = len(y_val)
n_test = len(y_test)

n_steps = 99
n_inputs = 26
n_neurons = 128
n_outputs = 2

learning_rate = 0.001
n_epochs = 100
batch_size = 128
n_batches = int(len(X_train)/batch_size)

X_batches = np.array_split(X_train[:-(n_train%batch_size)], n_batches)
Y_batches = np.array_split(y_train[:-(n_train%batch_size)], n_batches)

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
initial_state = tf.placeholder(tf.float32, [None, n_neurons])
keep_prob = tf.placeholder_with_default(1.0, shape=())

cell = tf.contrib.rnn.GRUCell(n_neurons)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32, initial_state=initial_state)

stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
logits = outputs[:,-1,:]

wakeword_probs = tf.nn.softmax(outputs)
last_prob = wakeword_probs[:,-1,:]

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

if train_model == True:
    with tf.Session() as sess:
        init.run()
        first_state = np.zeros((batch_size, n_neurons))
        val_state = np.zeros((n_val, n_neurons))
        for epoch in range(n_epochs):
            #initial_state = [[0.0]*100]*128
            avg_loss = 0.
            for i in range(n_batches):
                #print(first_state)
                X_batch, y_batch = X_batches[i], Y_batches[i]
                _, c, first_state = sess.run([training_op, loss, states], feed_dict={X: X_batch,
                                                                y: y_batch,
                                                                keep_prob: 0.75,
                                                                initial_state: first_state})
                avg_loss += c / n_batches

            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "Val accuracy:", accuracy.eval({X: X_val[:batch_size],
                                                                                                                    y: y_val[:batch_size],
                                                                                                                    keep_prob: 1.0,
                                                                                                                    initial_state: first_state}))

        
        print("Optimization Finished!")
        print("Test accuracy:", accuracy.eval({X: X_test, y: y_test, keep_prob: 1.0, initial_state: val_state}))
        save_path = saver.save(sess, "./models/" + model_name + ".ckpt")

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")

    rec_dict = {"recording":  [170, 740, 930],
                "recording2": [620],
                "recording3": [790],
                "recording4": [495, 940],
                "recording5": [363],
                "recording6": [795, 1684],
                "recording7": [1180],
                "concat_file": [200, 800, 1200, 1800],
                }

    for fname in rec_dict.keys():
        if not os.path.exists("plots/" + model_name):
            os.makedirs("plots/" + model_name)
        fs, x = wavfile.read(fname + ".wav")
        recording = normalize(mfcc(x, numcep=26))


        ww_times = rec_dict[fname]

        probs = []
        st = np.zeros((1, n_neurons))
        for i in range(0, len(recording)-99, 100):
            st, wp = sess.run([states, wakeword_probs], feed_dict={X: [recording[i:i+99]], keep_prob: 1.0, initial_state: st})
            #print(wp.shape)
            for val in wp[0,:,0]:
                probs.append(val)

        probs = np.asarray(probs[100:])
        threshold = 0.99
        detections = probs > threshold

        x = np.linspace(0, len(probs)-1, len(probs))

        plt.plot(x[detections], probs[detections], 'go')

        plt.plot(x, probs)

        for t in ww_times:
            plt.axvline(x=t-100, color="red")

        plt.savefig("plots/" + model_name + "/" + fname)
        plt.clf()