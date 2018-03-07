# Bidirectional LSTM based on the following example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
# 20 epochs, 25 % noise: val acc 85.98 %, test acc 86.41 %

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
    'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Construct model
logits = BiRNN(X, weights, biases)

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