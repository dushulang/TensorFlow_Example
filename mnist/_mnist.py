import tensorflow as tf 
import numpy as np 


from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print ("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print ("Done!")
    print ("Accuracy:", accuracy)






















'''

from numpy.random import RandomState 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

batch_size = 8


w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))


x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')


a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

cross_entropy=tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
#get train step while minimize(cross_entropy)
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)



rdm = RandomState(1)

dataset_size=128

X=rdm.rand(dataset_size,2)

Y=[[int(x1+x2<1)] for (x1,x2) in X]

STEPS = 5000











with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)  
    print(sess.run(w1))
    print(sess.run(w2))

    for i in range(STEPS):
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
        print("Afeter %d traing seteps(s), croos entropy on all data is %g"%(i,total_cross_entropy))

        print(sess.run(w1))
        print(sess.run(w2))
'''



 