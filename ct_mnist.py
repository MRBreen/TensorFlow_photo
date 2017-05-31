# https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt

"""fig = plt.figure()
ax = fig.add_subplot(111)

#x = np.random.normal(0,1,1000)
x = mnist.test.labels
numBins = 10
ax.hist(x,numBins,color='green',alpha=0.8)
plt.show()"""

# Create symbolic variables
x = tf.placeholder(tf.float32, [None, 784])

# Set variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)
#y = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)
#y = tf.nn.softmax(tf.softmax_cross_entropy_with_logits(x, W) + b)

# To implement cross-entropy we need to first add a new placeholder
# to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# results:
# range |  .next_batch( )
# 5000      100 -> 0.9239
# same          -> 0.9232
# 1000     100  -> .9139
#  500      100 -> .9094
#  100          -> .8943
#   50          -> .8609
#   20      100 -> .7874
