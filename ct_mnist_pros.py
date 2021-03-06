# https://www.tensorflow.org/get_started/mnist/pros

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Create symbolic variables
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Set variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

# Predicted Class and Loss Function
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train Model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# First Convolutional Layer
out1 = 50
W_conv1 = weight_variable([5, 5, 1, out1])  # 5x5 patch, 1 input channel, 32 output
b_conv1 = bias_variable([out1])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
out2 = 100
W_conv2 = weight_variable([5, 5, out1, out2])  # 5x5 patch, 32 input, 64 output
b_conv2 = bias_variable([out2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * out2, 1024]) #size is 7x7 with 64 inputs
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*out2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(1000): #20000
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# results:
# range |  .next_batch( )
# 5000      100 ->  .9206
# same          ->
# 1000     500  ->  .9078
# 1000     100  ->  .9066
#  500      100 ->
#  100          ->
#   50          ->
#   20      100 ->
"""
Results for a typical default settings - ie 32 and 64:
step 0, training accuracy 0.02
step 100, training accuracy 0.84
step 200, training accuracy 0.94
step 300, training accuracy 0.9
step 400, training accuracy 0.98
step 500, training accuracy 0.96
step 600, training accuracy 1
step 700, training accuracy 0.98
step 800, training accuracy 0.84
step 900, training accuracy 1
test accuracy 0.9646"""

"""
Results for a changing default settings to 30 and 60:
step 0, training accuracy 0.12
step 100, training accuracy 0.86
step 200, training accuracy 0.92
step 300, training accuracy 0.86
step 400, training accuracy 0.96
step 500, training accuracy 0.92
step 600, training accuracy 1
step 700, training accuracy 0.94
step 800, training accuracy 0.88
step 900, training accuracy 1
test accuracy 0.9576 """
