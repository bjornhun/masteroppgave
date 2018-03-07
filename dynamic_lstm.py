# Simple LSTM based on the following example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb
# 20 epochs, 25 % noise: val acc 85.71 %, test acc 86.62 %

import tensorflow as tf
import numpy as np
import pickle
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"

# Training Parameters
learning_rate = 0.001
n_epochs = 20
batch_size = 128

# Network Parameters
n_inputs = 26
timesteps = 99
n_hidden = 128
n_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, timesteps, n_inputs])
y = tf.placeholder("float", [None, n_classes])

# Read data
X_train, y_train = pickle.load(open(data_path + "train.p", "rb"))
X_test, y_test = pickle.load(open(data_path + "test.p", "rb"))
X_val, y_val = pickle.load(open(data_path + "val.p", "rb"))

# Define weights and biases
weights = {
    'out': tf.get_variable('W_o', shape=[n_hidden, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'out': tf.get_variable('b_o', shape=[n_classes])
}

# Create model
def lstm(X, weights, biases):
    lstmcell = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True)
    _, last = tf.nn.dynamic_rnn(cell=lstmcell, inputs=X, dtype=tf.float32)
    flow = last[-1]
    return tf.matmul(flow, weights['out']) + biases['out']

# Construct model
logits = lstm(X, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Define accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(n_epochs):
        avg_cost = 0.
        n_batches = int(len(X_train)/batch_size)
        X_batches = np.array_split(X_train, n_batches)
        Y_batches = np.array_split(y_train, n_batches)
        # Loop over all batches
        for i in range(n_batches):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / n_batches
        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "Val accuracy:", accuracy.eval({X: X_val, y: y_val}))
    print("Optimization Finished!")

    print("Test accuracy:", accuracy.eval({X: X_test, y: y_test}))