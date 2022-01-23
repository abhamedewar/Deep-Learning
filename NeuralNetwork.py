""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import time
import datetime
import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 800
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
if os.path.isdir(mnist_folder) != True:
    os.mkdir('data')
    os.mkdir(mnist_folder)
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)


# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.compat.v1.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.compat.v1.data.Dataset.from_tensor_slices(test)
#shuffling of test data not required
test_data = test_data.batch(batch_size)


# Step 3: create weights and bias

# #each image is of 28X28
input_shape = 784
hidden_layer1 = 128
hidden_layer2 = 40
#there are 10 digits 0-9
output_layer = 10

def get_weight( shape , name ):
    return tf.Variable(tf.random.normal(shape), name=name, trainable=True, dtype=tf.float32)

class Model():

    # Step 4: build model
    # the model that returns the logits.
    # this logits will be later passed through softmax layer

    #initialize the weights and bias
    def __init__(self):
        weight_shapes = [
            [input_shape, hidden_layer1] ,
            [hidden_layer1, hidden_layer2],
            [hidden_layer2, output_layer],
        ]

        self.weights = []
        for i in range(len(weight_shapes)):
            self.weights.append(get_weight(weight_shapes[i], 'weight{}'.format(i)))

        bias_shape = [
            [hidden_layer1],
            [hidden_layer2],
            [output_layer]
        ]

        self.bias = []
        for i in range(len(bias_shape)):
            self.bias.append(get_weight(bias_shape[i], 'bias{}'.format(i)))

    #define the network architecture
    def compute_logits(self, img):
        layer_one = tf.add(tf.matmul(img, self.weights[0]), self.bias[0])
        Z1 = tf.nn.relu(layer_one)
        second_layer = tf.add(tf.matmul(Z1, self.weights[1]), self.bias[1])
        output_layer = tf.matmul(second_layer, self.weights[2]) + self.bias[2]

        return output_layer

    #this will be used in the training loop
    def trainable_variables(self):

        return self.weights + self.bias


model = Model()

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def compute_loss(logits, labels):
    entropy =  tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits, name='entropy')
    loss = tf.reduce_mean(entropy)
    return loss

# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
#default learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08
optimizer = tf.optimizers.Adam(learning_rate)

# Step 7: calculate accuracy
def compute_accuracy(dataset):
    n_total = 0
    n_total_correct_prediction = 0
    for _, (imgs, labels) in enumerate(dataset):
        predicts = tf.nn.softmax(model.compute_logits(imgs))
        n_correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1)), tf.float32))
        n_total_correct_prediction += n_correct_prediction
        n_total += len(imgs)
    return n_total_correct_prediction / n_total

# TensorBoard writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_train = 'dnn/' + current_time + '/train'
log_dir_test = 'dnn/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(log_dir_train)
test_summary_writer = tf.summary.create_file_writer(log_dir_test)

# Step 8: train the model
start_time = time.time()
# train the model n_epochs times
for i in range(n_epochs):
    total_loss = 0
    n_batches = 0
    # draw samples from training data

    for j, (imgs, labels) in enumerate(train_data):
        with tf.GradientTape() as g:
            # logits = compute_logits(imgs)
            logits = model.compute_logits(imgs)
            loss = compute_loss(logits, labels)
        gradients = g.gradient(loss, model.trainable_variables())
        optimizer.apply_gradients(zip(gradients, model.trainable_variables()))
        total_loss += loss
        n_batches += 1

    
    # compute loss and accuracy for training data and testing data
    # after each epoch
    average_loss = total_loss / n_batches
    train_accuracy = compute_accuracy(train_data)
    test_accuracy = compute_accuracy(test_data)
    
    with train_summary_writer.as_default():
        tf.summary.scalar('Loss',  average_loss, step=i)
        tf.summary.scalar('Accuracy', train_accuracy, step=i)
    with test_summary_writer.as_default():
        tf.summary.scalar('Accuracy', test_accuracy, step=i)
    
    template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Accuracy: {}'
    print(template.format(i+1, average_loss, train_accuracy*100, test_accuracy*100))


end_time = time.time()   

print('Total Time {} ms.'.format(end_time- start_time))