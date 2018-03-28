import tensorflow as tf
import numpy as np
import pickle
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"

X_train, y_train = pickle.load(open(data_path + "train.p", "rb"))
X_test, y_test = pickle.load(open(data_path + "test.p", "rb"))
X_val, y_val = pickle.load(open(data_path + "val.p", "rb"))

n_steps = 99
n_inputs = 26
n_neurons = 128
n_outputs = 2
n_layers = 3

learning_rate = 0.001
n_epochs = 20
batch_size = 128
n_batches = int(len(X_train)/batch_size)

X_batches = np.array_split(X_train, n_batches)
Y_batches = np.array_split(y_train, n_batches)

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None, n_outputs])
keep_prob = tf.placeholder_with_default(1.0, shape=())

cells = []
for _ in range(n_layers):
    grucell = tf.contrib.rnn.GRUCell(n_neurons)
    grucell = tf.contrib.rnn.DropoutWrapper(grucell, output_keep_prob=keep_prob)
    cells.append(grucell)
cell = tf.contrib.rnn.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
'''cells = [tf.contrib.rnn.GRUCell(n_neurons) for layer in range(n_layers)]
#cells_proj = [tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=n_outputs, activation=tf.nn.softmax) for cell in cells]
cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)'''

stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

logits = outputs[:,-1,:]
#logits = tf.layers.dense(last, n_outputs)



#logits = tf.layers.dense(outputs[:,-1,:], n_outputs, activation=tf.nn.softmax)
xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
#correct = tf.nn.in_top_k(logits, y, 1)
#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        avg_loss = 0.
        for i in range(n_batches):
            X_batch, y_batch = X_batches[i], Y_batches[i]
            _, c = sess.run([training_op, loss], feed_dict={X: X_batch,
                                                            y: y_batch,
                                                            keep_prob: 0.5})
            avg_loss += c / n_batches            
        #log = logits.eval(feed_dict={X: X_test, y: y_test})
        #y_pred = y.eval(feed_dict={X: X_test, y: y_test})
        #print(['%.4f' % elem for elem in log[:,1]][10:20])
        #print(y_pred[10:20])
        print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "Val accuracy:", accuracy.eval({X: X_val, y: y_val}))
    print("Optimization Finished!")

    print("Test accuracy:", accuracy.eval({X: X_test, y: y_test}))