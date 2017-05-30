# https://www.tensorflow.org/get_started/mnist/beginners

import numpy as np
import tensorflow as tf

# Model parapeters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to incorrect defaults
#sess.run(init) # reset values to incorrect defaults.
for i in range(10000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

#evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
