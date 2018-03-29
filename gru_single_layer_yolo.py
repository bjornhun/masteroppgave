import tensorflow as tf
import numpy as np
import pickle
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"

X_train, y_train = pickle.load(open(data_path + "train.p", "rb"))
X_test, y_test = pickle.load(open(data_path + "test.p", "rb"))
X_val, y_val = pickle.load(open(data_path + "val.p", "rb"))
X_concat = pickle.load(open(data_path + "concat_file.p", "rb"))

n_val = len(y_val)
n_test = len(y_test)

n_steps = 99
n_inputs = 26
n_neurons = 100
n_outputs = 2

learning_rate = 0.001
n_epochs = 100
batch_size = 128
n_batches = int(len(X_train)/batch_size)

X_batches = np.array_split(X_train[:-72], n_batches)
Y_batches = np.array_split(y_train[:-72], n_batches)

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
initial_state = tf.placeholder(tf.float32, [None, n_neurons])

cell = tf.contrib.rnn.GRUCell(n_neurons)
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

'''with tf.Session() as sess:
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
                                                            initial_state: first_state})
            avg_loss += c / n_batches
            #out, sta = sess.run([outputs, states], feed_dict={X: X_batch, 
            #                                                  y: y_batch,
            #                                                  initial_state: first_state})
        #log = logits.eval(feed_dict={X: X_test, y: y_test})
        #y_pred = y.eval(feed_dict={X: X_test, y: y_test})
        #y_prob, y_true = sess.run([last_prob, y], feed_dict={X: X_val, y: y_val, initial_state: val_state})
        #print(['%.4f' % elem for elem in y_prob[:,1]][10:20])
        #print(y_true[10:20])
        print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "Val accuracy:", accuracy.eval({X: X_val[:batch_size],
                                                                                                                y: y_val[:batch_size],
                                                                                                                initial_state: first_state}))
        #probs_pos = wakeword_probs.eval(feed_dict={X: [X_test[11]], y: [y_test[11]], initial_state: [first_state[0]]})
        #print(probs_pos)
    
    print("Optimization Finished!")
    #probs_pos = wakeword_probs.eval(feed_dict={X: [X_test[11]], y: [y_test[11]]})
    #probs_neg = wakeword_probs.eval(feed_dict={X: [X_test[12]], y: [y_test[12]]})
    #print(probs_pos)
    #print(probs_neg)
    print("Test accuracy:", accuracy.eval({X: X_test, y: y_test, initial_state: val_state}))
    save_path = saver.save(sess, "./single_layer_gru.ckpt")'''

with tf.Session() as sess:
    saver.restore(sess, "./single_layer_gru.ckpt")
    #probs = []
    #st = np.zeros((1, n_neurons))
    #for i in range(10):
    #    st, wp = sess.run([states, wakeword_probs], feed_dict={X: [X_test[i]], y: [y_test[i]], initial_state: st})
    #    print(wp)
    #    print(y_test[i])
    probs = []
    st = np.zeros((1, n_neurons))
    for i in range(0, 1999, 100):
        st, wp = sess.run([states, wakeword_probs], feed_dict={X: [X_concat[i:i+99]], initial_state: st})
        print(wp.shape)
        for val in wp[0,:,0]:
            probs.append(val)
    import matplotlib.pyplot as plt
    plt.plot(range(len(probs)), probs)
    plt.show()