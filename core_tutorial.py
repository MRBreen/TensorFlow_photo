import numpy as np
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#adder_node = a + b  # + provides a shortcut for tf.add(a, b)

#print(sess.run(adder_node, {a: 3, b:4.5}))
#print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#init = tf.global_variables_initializer()
#sess.run(init)

#print(sess.run(linear_model, {x:[1,2,3,4]}))

#loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to incorrect defaults
#fixW = tf.assign(W, [-1.1])
#fixb = tf.assign(b, [1.1])
#sess.run([fixW, fixb])
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#optimizer = tf.train.GradientDescentOptimizer(0.01)
#train = optimizer.minimize(loss)

#sess.run(init) # reset values to incorrect defaults.
for i in range(1000000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

#evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
