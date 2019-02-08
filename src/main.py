# This code is inspired by https://www.youtube.com/watch?v=yX8KuPZCAMo



import tensorflow as tf

# There are two step to tensor, setup which sets up the operators
# and an execution which is where we change the data.

print('****** Tensors *******')
# Creating tensors
# Tensors is like kind of like arrays... but can have dimensions
# Tensors have different dimentions aka ranks
# Tensor of rank 0 = A constant
# Tensor of rank 1 = An array/ A vector
# Tensor of rank 2 = A matrix
# Tensor of rank 3 = An Array of arrays/ a Cube of numbers
# Tensors cant do anything they only holds te value

node1 = tf.constant(3, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(4.123, tf.float64)

print("******node1******")
print(node1)
print("******node2******")
print(node2)
print("******node3******")
print(node3)

# To execute tensors it needs to be run in a session

sess = tf.Session()

#one way to run a session
print(sess.run([node1,node2,node3]))

sess.close()

# output should be [3, 4.0, 4,123]
# Another way to run a session is:
with tf.Session() as session:
    output = session.run([node1, node2, node3])
    print(output)

# auto close the session, same output ;)

# You can use this to setup the variables to an with operations to run on them and then
# run the session
# Eg if we wanted to multiply oure nodes then


sess = tf.Session()

c = node1 * node2

# Use tensorboard to watch the computation graph
File_Writer = tf.summary.FileWriter("graph", sess.graph)

print("Results")
print(sess.run(c))
sess.close()

# Placeholders
# is a promise to provide a value

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b

sess = tf.Session()
print(sess.run(adder_node, {a:[1,4],b:[3,4]}))

# Values
# you use variables to hold and upgrade variables, variable ar e constant changeing
# must be explicitly initialized

a = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = a*x+b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

# making a Model and
# Calculating the Loss
# model Parameters
a = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# input
x = tf.placeholder(tf.float32)

# The Model
linear_model = a*x+b

# desired output
y = tf.placeholder(tf.float32)

#Loss function
squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("The Loss:")
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Optimizers
# an optimizer updates the variables an examin the output
# if the a change i a variable is decressing the loss the optimizer
# will continue to change the value in that direction

# making a Model and
# Calculating the Loss
# model Parameters
a = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# input
x = tf.placeholder(tf.float32)

# The Model
linear_model = a*x+b

# desired output
y = tf.placeholder(tf.float32)

#Loss function
squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)

#Optimizer
LearningRate = 0.01
optimizer = tf.train.GradientDescentOptimizer(LearningRate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

numberOfEpochs = 1000
for i in range(numberOfEpochs):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    print("calculating: ",i)
print(sess.run([a,b]))