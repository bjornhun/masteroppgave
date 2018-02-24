# MLP based on the following example: https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb

import tensorflow as tf
import numpy as np
import pickle

# Parameters
learning_rate = 0.001
n_epochs = 100
batch_size = 100

# Network Parameters
n_inputs = 26*99 # Number of feature
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_hidden_3 = 512 # 2nd layer number of features
n_classes = 2 # Number of classes to predict

X_train, y_train = pickle.load(open("data/train.p", "rb"))
X_test, y_test = pickle.load(open("data/test.p", "rb"))
X_val, y_val = pickle.load(open("data/val.p", "rb"))

n_train = len(X_train)
n_test = len(X_test)
n_val = len(X_val)

X_train = X_train.reshape(n_train, n_inputs)
X_test = X_test.reshape(n_test, n_inputs)
X_val = X_val.reshape(n_val, n_inputs)

# tf Graph input
X = tf.placeholder("float", [None, n_inputs])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(X, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    logits = tf.matmul(layer_3, weights['out']) + biases['out']
    return logits

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
logits = multilayer_perceptron(X, weights, biases)

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