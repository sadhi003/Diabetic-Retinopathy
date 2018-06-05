
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = '/Users/shankar/Dropbox/Kaggle/Diabetes/inputdata/retinopathy.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  del save
  
image_size = train_dataset.shape[1]
print (image_size)
print (train_dataset.shape)
print (train_labels.shape)


num_labels = 5
num_channels = 3 # grayscale

X = 768

train = train_dataset[0:X, :, :, :]
label = train_labels[0:X]

validation = train_dataset[X:, :, :, :]
val_label = train_labels[X:]

print ('Training:', train.shape, label.shape)
print ('Validation:', validation.shape, val_label.shape)




def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size,image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train, label = reformat(train, label)
validation, val_label = reformat(validation, val_label)
print ('Training:', train.shape, label.shape)
print ('Validation:', validation.shape, val_label.shape)
#test_dataset, test_labels = reformat(test_dataset, test_labels)
#print('Test set', test_dataset.shape, test_labels.shape)
#print (train)


# train the different models here before implimenting on neural network



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  
  tf_valid_dataset = tf.constant(validation)
  #tf_test_dataset = tf.constant(test_dataset)
  
  ''' Variables.
   1. here the filter define with the weight size define by pathches
   2. main idea of filter is here, where filter will update based on loss function 
   through optimization, 
   3. here depth specify the number of weights(filters)
   4. random number weight generate here with dimension of (5*5*3 + 1)*16
   5. giving total number of weight = 16 for first layer'''

  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  # here patch_size are 2dim flat space, num_channels are the dimension of image and 
  #  depth is the what is our output after convolve the filter on the patch. 

  layer1_biases = tf.Variable(tf.zeros([depth]))  # add biases with the filter value

# this is for 2nd convolution filter

  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  ''' next convolution 
  layer21_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer21_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  next convolution
  layer31_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer31_biases = tf.Variable(tf.constant(1.0, shape=[depth]))'''
# weight initialized for fully connected layer, where total number of weights are 
#     size = 8*8*16, output = 

  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(.5, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  

  # Model.// here [1,2,2,1], applies the filter on every other patches

  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    print ('first conv. dimension ',  hidden.get_shape().as_list())
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    print ('2nd conv. dimension ',  hidden.get_shape().as_list())
    '''conv = tf.nn.conv2d(hidden, layer21_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer21_biases)
    print ('third conv. dimension ',  hidden.get_shape().as_list())
    conv = tf.nn.conv2d(hidden, layer31_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer31_biases)
    print ('fourth conv. dimension ',  hidden.get_shape().as_list())'''
    shape = hidden.get_shape().as_list()
    print ('2nd conv. dimension  ', shape)
    # start fully connected layer, where total number of weights are 
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    print (reshape)
    hidden = tf.nn.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  #valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  #test_prediction = tf.nn.softmax(model(tf_test_dataset))
num_steps = 501

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (label.shape[0] - batch_size)
    #print (offset)
    batch_data = train[offset:(offset + batch_size), :, :, :]
    batch_labels = label[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      #print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), val_label))
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))





