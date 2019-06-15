import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop('label', axis=1)
Y_train = train['label'].copy()

X_train = X_train / 255.0
test = test / 255.0

# to values/ array
X_train = X_train.values
test = test.values

# 減少feature可以快一些，但會犧牲準確度
# from sklearn.decomposition import PCA
# pca = PCA(.95)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# test = pca.transform(test)

# Y_train要做oneHot
Y_train = pd.get_dummies(Y_train)
Y_train = Y_train.values
test_dump = Y_train[:28000]

import os
import time

import tensorflow as tf

from functools import reduce

from sklearn.metrics import accuracy_score

import collections

#evaluation
from sklearn.metrics import confusion_matrix
class TreeProperties(object):
    '''
    :param max_leafs: maximum number of leafs
    :param n_features: maximum number of feature available within the data
    :param n_classes: number of classes
    '''
    def __init__(self,max_depth,max_leafs,n_features,n_classes,regularisation_penality=10.,decay_penality=0.9):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.n_features = n_features
        self.n_classes = n_classes

        self.epsilon = 1e-8

        self.decay_penality = decay_penality
        self.regularisation_penality = regularisation_penality


class Node(object):
    def __init__(self,id,depth,pathprob,tree):
        self.id = id
        self.depth = depth

        self.prune(tree)

        if self.isLeaf:
            self.W = tf.get_variable(name='weight_' + self.id,
                                     shape=(tree.params.n_features,tree.params.n_classes),
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer)
            self.b = tf.get_variable(name='bias_' + self.id, shape=(tree.params.n_classes,), dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer)
        else:
            self.W = tf.get_variable(name='weight_' + self.id, shape=(tree.params.n_features,1), dtype=tf.float32,
                                     initializer=tf.random_normal_initializer)
            self.b = tf.get_variable(name='bias_' + self.id, shape=(1,), dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer)


        self.leftChild = None
        self.rightChild = None



        self.pathprob = pathprob

        self.epsilon = 1e-8 #this is a correction to avoid log(0)

    def prune(self,tree):
        '''
        prunes the leaf by setting isLeaf to True if the pruning condition applies.
        :param tree:
        '''
        self.isLeaf = (self.depth>=tree.params.max_depth)
    def build(self,x,tree):
        '''
        define the output probability of the node and build the children
        :param x:
        :return:
        '''
        self.prob = self.forward(x)

        if not(self.isLeaf):
            self.leftChild = Node(id=self.id + str(0), depth=self.depth + 1, pathprob=self.pathprob * self.prob,
                                  tree=tree)
            self.rightChild = Node(self.id + str(1), depth=self.depth + 1, pathprob=self.pathprob * (1. - self.prob),
                                   tree=tree)

    def forward(self,x):
        '''
        defines the output probability
        :param x:
        :return:
        '''
        if self.isLeaf:
            # TODO: replace by logsoft max for improved stability
            return tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        else:
            return tf.keras.backend.hard_sigmoid(tf.matmul(x, self.W) + self.b)

    def regularise(self,tree):
        if self.isLeaf:
            return 0.0
        else:
            alpha = tf.reduce_mean(self.pathprob * self.prob) / (
                        self.epsilon + tf.reduce_mean(self.pathprob))
            return (-0.5 * tf.log(alpha + self.epsilon) - 0.5 * tf.log(
                1. - alpha + self.epsilon)) * (tree.params.decay_penality** self.depth)

    def get_loss(self,y,tree):
        if self.isLeaf:
            return -tf.reduce_mean( tf.log( self.epsilon+tf.reduce_sum(y *self.prob, axis=1) )*self.pathprob )
        else:
            return tree.params.regularisation_penality * self.regularise(tree)

class SoftDecisionTree(object):
    def __init__(self, *args,**kwargs):
        self.params = TreeProperties(*args,**kwargs)

        self.n_leafs = 1
        self.loss = 0.0

        self.output = list()
        self.leafs_distribution = list()

        self._n_nodes = 0
        self._n_leafs = 0

    @property
    def n_leafs(self):
        return self._n_leafs

    @n_leafs.setter
    def n_leafs(self,n):
        self._n_leafs = n

    def add_leaf(self,node):
        self.n_leafs += int(node.isLeaf)

    @property
    def n_nodes(self):
        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self,n):
        self._n_nodes = n

    def add_node(self):
        self.n_leafs += 1

    def build_tree(self):
        self.tf_X = tf.placeholder(tf.float32, [None, self.params.n_features])
        self.tf_y = tf.placeholder(tf.float32, [None, self.params.n_classes])

        leafs = list()
        self.root = Node(id='0',depth=0,pathprob=tf.constant(1.0,shape=(1,)),tree=self)
        leafs.append(self.root )

        for node in leafs:
            self.n_nodes+=1
            node.build(x=self.tf_X,tree=self)
            self.loss += node.get_loss(y=self.tf_y, tree=self)

            self.add_node()
            self.add_leaf(node)
            if node.isLeaf:
                #self.n_leafs+=1
                self.output.append(node.prob)
                self.leafs_distribution.append(node.pathprob)
            else:
                leafs.append(node.leftChild)
                leafs.append(node.rightChild)


        self.output = tf.concat(self.output,axis=1)
        self.leafs_distribution = tf.concat(self.leafs_distribution,axis=1)

        print('Tree has {} leafs and {} nodes'.format(self.n_leafs,self.n_nodes))

    def boost(self,X,y,sess):
        _,c = sess.run(
            [
                optimizer,tree.loss
            ],
            feed_dict={
                self.tf_X: X,
                self.tf_y: y
            }
        )
        return c
        
    def predict(self,X,y,sess):
        leafs_distribution, leaf_probs = sess.run(
            [
                self.leafs_distribution,
                self.output
            ],
            feed_dict={
                self.tf_X: X,
                self.tf_y:y
            }
        )
        return self.get_prediction_target(leafs_distribution,leaf_probs)
        
    def get_prediction_target(self,leafs_distribution,leaf_probs):
        indices = np.argmax(leafs_distribution, axis=1)
        return [np.argmax(leaf_probs[nn, indices[nn] * self.params.n_classes:\
                                         indices[nn] * self.params.n_classes\
                                         + self.params.n_classes]) for nn in
                range(leaf_probs.shape[0])]
                
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    # numpy-array to list
    # idx = list(idx)
    
    # data_shuffle = [data[i] for i in idx]
    # labels_shuffle = [labels[i] for i in idx]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# 如果要做pca，要改feature數量
n_features = 784
n_classes = 10
batch_size = 32
# 驗證時用
val_batch_size = 256

# 建構高度不超過6的樹
tree = SoftDecisionTree(max_depth=6,n_features=n_features,n_classes=n_classes,max_leafs=None)
tree.build_tree()

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(tree.loss)

# Saving the model
#saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

EPOCHS = 1000
TOTAL_BATCH = 16
display_step = 100
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCHS):

        avg_cost = 0.
        # Loop over all batches
        acc =0.0
        val_acc = 0.0
        for i in range(TOTAL_BATCH):
            batch_xs, batch_ys = next_batch(batch_size, X_train, Y_train)

            c = tree.boost(X=batch_xs,y=batch_ys,sess=sess)

            target = np.argmax(batch_ys,axis=1)
            preds = tree.predict(X=batch_xs,y=batch_ys,sess=sess)
            acc += accuracy_score(y_pred=preds,y_true=target)/TOTAL_BATCH

            # Compute average loss
            avg_cost+= acc/TOTAL_BATCH
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            batch_val_xs, batch_val_ys = next_batch(val_batch_size, X_train, Y_train)

            val_target = np.argmax(batch_val_ys, axis=1)
            val_preds = tree.predict(X=batch_val_xs,y=batch_val_ys,sess=sess)
            val_acc = accuracy_score(y_pred=val_preds, y_true=val_target)
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(avg_cost),"training_accuracy=","{:.4f}".format(acc),
                  "validation_accuracy=","{:.4f}".format(val_acc)  )
            
            #print(collections.Counter(np.argmax(path_probs,axis=1)))

            #print(confusion_matrix(y_true=val_target,y_pred=val_preds) )
            
    # 做預測
    predictions = tree.predict(X=test,y=test_dump,sess=sess)
    sub = pd.read_csv("sample_submission.csv")
    sub['Label'] = predictions
    sub.to_csv('CART_model_sub.csv',index=False)