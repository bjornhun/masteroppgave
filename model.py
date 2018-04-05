import tensorflow as tf
import numpy as np
import pickle
import os
from scipy.io import wavfile
from python_speech_features import mfcc
from preprocessing import normalize
import matplotlib.pyplot as plt

data_path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
model_name = "one_layer_gru_10-90_var_len_random_state"
train_model = False
plot_preds = False

# Load data
X_train, y_train = pickle.load(open(data_path + "train.p", "rb"))
X_test, y_test = pickle.load(open(data_path + "test.p", "rb"))
X_val, y_val = pickle.load(open(data_path + "val.p", "rb"))

n_train = len(y_train)
n_val = len(y_val)
n_test = len(y_test)

# Set parameters
n_inputs = 26
n_timesteps = 99
n_neurons = 128
n_outputs = 2
n_layers = 1

learning_rate = 0.001
n_epochs = 100
batch_size = 128
n_batches = int(len(X_train)/batch_size)

# Divide training data into batches
X_batches = np.array_split(X_train[:-(n_train%batch_size)], n_batches)
Y_batches = np.array_split(y_train[:-(n_train%batch_size)], n_batches)

# Define placeholders
X = tf.placeholder(tf.float32, [None, None, n_inputs])
y = tf.placeholder(tf.int32, [None])
initial_state = tf.placeholder(tf.float32, [n_layers, None, n_neurons])
seq_length = tf.placeholder(tf.int32)

# Make initial state tuple
initial_state_list = []
for i in range(n_layers):
    initial_state_list.append(initial_state[i])
initial_state_tuple = tuple(initial_state_list)

# Define network
cells = []
for _ in range(n_layers):
    grucell = tf.contrib.rnn.GRUCell(n_neurons)
    cells.append(grucell)
cell = tf.contrib.rnn.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                    inputs=X,
                                    dtype=tf.float32,
                                    initial_state=initial_state_tuple,
                                    sequence_length=seq_length)

# Compute wake word probabilities from RNN outputs
stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, seq_length, n_outputs])
logits = outputs[:,-1,:] # Used for optimization
wakeword_probs = tf.nn.softmax(outputs)[0,:,0] # Wake word probabilities for each timestep

# Optimize
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Compute accuracy
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize global variables and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Train model
if train_model == True:
    with tf.Session() as sess:
        init.run()
        test_state = np.random.uniform(low=-1.0, high=1.0, size=((n_layers, n_test, n_neurons)))
        val_state = np.random.uniform(low=-1.0, high=1.0, size=((n_layers, n_val, n_neurons)))
        for epoch in range(n_epochs):
            avg_loss = 0.
            for i in range(n_batches):
                init_state = np.random.uniform(low=-1.0, high=1.0, size=((n_layers, batch_size, n_neurons)))
                X_batch, y_batch = X_batches[i], Y_batches[i]
                _, c = sess.run([training_op, loss],
                                            feed_dict={X: X_batch,
                                                       seq_length: n_timesteps,
                                                       y: y_batch,
                                                       initial_state: init_state})
                avg_loss += c / n_batches

            print("Epoch:", '%04d' % (epoch+1),
                  "loss=", "{:.9f}".format(avg_loss),
                  "Val accuracy:", accuracy.eval({X: X_val,
                                                  y: y_val,
                                                  seq_length: n_timesteps,
                                                  initial_state: val_state}))

        
        print("Optimization Finished!")
        print("Test accuracy:", accuracy.eval({X: X_test,
                                               y: y_test,
                                               seq_length: n_timesteps,
                                               initial_state: test_state}))
        save_path = saver.save(sess, "./models/" + model_name + ".ckpt")

if plot_preds == True:
    with tf.Session() as sess:
        saver.restore(sess, "./models/" + model_name + ".ckpt")

        rec_dict = {"recording":  [170, 740, 930],
                    "recording2": [620],
                    "recording3": [790],
                    "recording4": [495, 940],
                    "recording5": [363],
                    "recording6": [795, 1684],
                    "recording7": [1180],
                    "concat_file": [200, 800],
                    }

        for fname in rec_dict.keys():
            if not os.path.exists("plots/" + model_name):
                os.makedirs("plots/" + model_name)
            fs, x = wavfile.read(fname + ".wav")
            recording = normalize(mfcc(x, numcep=26))

            init_state = np.random.uniform(low=-1.0, high=1.0, size=((n_layers, 1, n_neurons)))

            ww_times = rec_dict[fname]

            probs=[]
            for i in range(0, len(recording)-9, 10):
                init_state, wp = sess.run([states, wakeword_probs], feed_dict={X: [recording[i:i+10]], seq_length: 10, initial_state: init_state})
                for val in wp:
                    probs.append(val)
            probs = np.asarray(probs)

            threshold = 0.99
            detections = probs > threshold

            x = np.linspace(0, len(probs)-1, len(probs))
            plt.plot(x[detections], probs[detections], 'go')
            plt.plot(x, probs)
            for t in ww_times:
                plt.axvline(x=t, color="red")
            plt.savefig("plots/" + model_name + "/" + fname)
            plt.clf()