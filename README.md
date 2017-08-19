# DL_tuning
Deep Learning model tuning experience. Some of them are summarized from Andrew Ng's course.

# Data:
1. The more, the better. 
2. Always normalize the data before feeding into the neural network. May not be helpful in increasing the performance but it would not hurt as well. Almost always helpful in increasing the learning speed.
3. The ratio of training data and testing data should be the legend 8:2 or 7:3. The world is changing. Let's go for 100:1 or even more for big data era.

# Normalization:
1. Always normalize the data before feeding into the neural network.
2. Batch normalization. It would speed the training process and have some regularization effect.

# Basic Strategy:
1. Set one and only one single number evaluation metric related to the final result of the model.
2. Train a model on the training dataset, optimizing the metric, and test on the testing dataset.
3. Oversee the bias (how much does the model fit the training dataset) and variance (how well does the model generalize on the testing data set) during the training process.
4. Carry on error analysis. Check the wrong labeled data to see why they are labeled incorrectly
4. Tune hyperparameter and iterate over the previous steps.

# Hyperparameter Tuning Principle:
1. Orthogonalization. When trying to reduce bias, do not affect variance. When trying to reduce variance, do not affect bias.
2. Ways to reduce bias: bigger network, alternative optimizer (related hyperparameter, like learning rate), different network structure.
3. Ways to reduce variance: more train data, weight decay, dropout.
4. Random search instead of grid search.
5. Crude and refine search.

# Hyperparameters List:
Personally, I think the importance would go down as the list goes down.
1. Model structure. The fancier model would indeed give better performance. We can try from the legend CNN or RNN to make sure the model would work and then turn to the more advanced structure.
2. Optimizer and related learning rate. Under most circumstance, SGD with Momentum should serve as the baseline optimizer, and Adam should be better than SGD. However, the assumption does not always hold. The learning rate should be carefully chosen in both cases.
3. Momentum coefficient. Under most circumstance, 0.9 should give a good result.
4. The number of hidden nodes. No golden standard to follow. The one should be chosen based on experience and intuition. Since we have the following regularization methods, we can try big network.
5. Weight decay coefficient. Increase it if the variance is high. However, this should be increased too much otherwise it would affect bias.
6. Dropout ratio. Try from 0.2 to 0.5. Using dropout in each layer if possible.
7. Batch size. Use as big batch as possible. Personally, I think the main reason that we have to use mini-batch is that we can not feed all data to GPU due to the relatively small memory size. I did not see the disadvantage of using big batch.
8. Number of layers. No golden standard. For fully connected layers, no need to have ten layers and it would hurt in both speed and performance.
9. Learning rate decay. We can have a try.
10. Early stopping. Not recommended. It hurts the Orthogonalization rule.

# CNN Models
1. AlexNet
2. SqueezeNet (Small net but can reach AlexNet performance)
3. VGG
4. Google Inception v1, v2, v3, v4(Inception-ResNet)
5. ResNet v1, v2
6. FCN
7. ResNeXt
8. DenseNets


# RNN Models
Not very familiar with the evolution, only know the most advanced ones
1. LSTM
2. Bi-LSTM
3. WaveNet
4. Google transformer
5. Convolutional Seq2Seq


# Code in Tensorflow
I would thank Tensorflow team to make the software open source, but I have to say the documentation, expecially the API, is not very clear and for each command I would have to search the Internet to see how to use it. I would appreciate if the documentation of Tensorflow would be something like the documentation of Numpy or Scipy, which would have several examples for each operation.
Anyway, the current documentation is much better than that of Tensorflow v0.7. You can not image how crude it is. 

Weight Initailization
```
def weight_variable(shape):
    import math
    if len(shape)>2:
    # For 2d conv weights
        weight_std=math.sqrt(2.0/(shape[0]*shape[1]*shape[2]))
    else:
    # For fully connected weights
        weight_std=math.sqrt(2.0/shape[0])
    initial=tf.truncated_normal(shape,stddev=weight_std)
    return tf.Variable(initial,name='weights')
```
Dropout
```
# For fully connected layers
fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout)

# For RNN
cell = tf.contrib.rnn.DropoutWrapper(
    cell, output_keep_prob=1.0 - dropout)
```
Weight decay
```
# theta is the weight decay coefficient. We ignore the bias here.
weight_collection=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_collection])
cross_entropy_with_weight_decay=tf.add(cross_entropy,theta*l2_loss)
```
Learning rate decay
```
# Here we use it in Momentum, but we can try to use it in Adam. No theory proves that we can't use it in Adam
global_step=tf.Variable(0,trainable=False)
starter_learning_rate=0.01
learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,200,0.96,staircase=True)
train_step=tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(cross_entropy_with_weight_decay,global_step=global_step)
```
Batch normalization
From: http://ruishu.io/2016/12/27/batchnorm/
```
# For graph define, do batch norm before the activation function
# phase need to be a placeholder which accpet value to indicate whether we are
# in train phase or in testing phase
h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase,scope='bn')

# For train op define
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
```
