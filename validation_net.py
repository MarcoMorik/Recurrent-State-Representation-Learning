# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:29:54 2018

@author: marco
"""

import numpy as np
import sonnet as snt
import tensorflow as tf
import os
from collections import Counter
from sklearn.neighbors import NearestNeighbors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class validation_net:
    
    def __init__(self, inp_dim, out_dim, seed=1, learning_rate=0.001, l1_reg = 0.001, one_hot = False):
        ##############Variables:        
        hidden_nodes = 64
        self.one_hot = one_hot
        self.inp_var = tf.placeholder(tf.float32, shape=[None,inp_dim], name="input")
        self.is_training = tf.placeholder(tf.bool, shape=[],  name="train_cond")
        self.weights = tf.placeholder(tf.float32, shape=[None],  name="train_cond")

        if(one_hot):
            self.learning_goal =  tf.placeholder(tf.int32, shape=[None], name="pose")
            self.learning_goal_one_hot = tf.one_hot(self.learning_goal,out_dim)
        else:
            self.learning_goal =  tf.placeholder(tf.float32, shape=[None,out_dim], name="pose")
            
        
        regularizers_in = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                        "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}        
        regularizers_out = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                        "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}        

        inp_layer = snt.Linear(output_size=hidden_nodes, name='inp_to_hidden', regularizers=regularizers_in)
        middle_layer = snt.Linear(output_size=hidden_nodes, name='hidden_to_hidden')
        out_layer = snt.Linear(output_size=out_dim, name='hidden_to_out', regularizers=regularizers_out)

        # to keep loss terms differentiable, states should never be equal
        noise_layer = lambda x: tf.cond(self.is_training, lambda: x + tf.random_normal(shape=tf.shape(x), stddev=1e-6), lambda: x)
        seq_input_layer = snt.Sequential([inp_layer,noise_layer,tf.nn.relu,middle_layer,tf.nn.relu,out_layer])
        
        out_state = seq_input_layer(self.inp_var)
        
        if(one_hot):
            self.state = tf.argmax(out_state,axis=1)
        else:
            self.state = out_state
            
        ########## Losses: ###########
        graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_regularization_loss = tf.reduce_sum(graph_regularizers)
                            

        distance = lambda x, y: tf.norm((x-y),ord=2, axis=1)**2
        
        if(one_hot):
            self.loss = tf.losses.softmax_cross_entropy(self.learning_goal_one_hot,out_state, weights=self.weights) + total_regularization_loss
        else:
            self.loss = tf.reduce_mean(distance(out_state,self.learning_goal)) + total_regularization_loss        
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradient = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(gradient)
        self.saver = tf.train.Saver()
            

    def learn(self, inp_data, pose, num_epochs= 2000, batch_size= 1000, num_batches = 20, weights=[], validation_data=[], validaton_pose=[], validation_weight=[]):

        inp_size = np.shape(inp_data)[0]
        
        if(len(validation_data)!=0):
            optimize_with_validation = True
            validation_losses = []
            
        else:
            optimize_with_validation = False
        # TRAINING -----------------------------------------------------------------------------------------------------

        num_batches= int(inp_size/batch_size)
        losses = []
        min_loss = np.inf
        min_loss_count=0
        if(self.one_hot and len(weights)==0):
            pose_counter = Counter(pose)
            weights = np.asarray([inp_size / pose_counter[x] for x in pose])

        with tf.Session() as sess:
            init = tf.global_variables_initializer()      
            sess.run(init)  
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                epoch_permutation = np.random.permutation(inp_size)
                for i in range(num_batches):
                    curr_batch = epoch_permutation[i*batch_size:(i+1)*batch_size]
                    curr_inp = inp_data[curr_batch]
                    curr_pose = pose[curr_batch]
                    curr_weights = weights[curr_batch]
                    
                    _ , tmp_loss  = sess.run([self.train_op,self.loss], feed_dict={
                                    self.inp_var: curr_inp,
                                    self.learning_goal: curr_pose,
                                    self.is_training: True,
                                    self.weights: curr_weights})
                                    
                    
                    epoch_loss += tmp_loss
                losses.append(epoch_loss)
                if(optimize_with_validation):
                    validation_loss = sess.run(self.loss, feed_dict={self.inp_var: validation_data, self.learning_goal: validaton_pose, self.is_training: False, self.weights: validation_weight})
                    validation_losses.append(validation_loss)
                    epoch_loss = validation_loss
                    
                if(epoch_loss <= min_loss):
                    min_loss = epoch_loss
                    min_loss_count = 0
                else:
                    min_loss_count += 1

                if(min_loss_count >=30):
                    print("Validation network learned after {:3} epochs".format(epoch))
                    break
                if( epoch % 50 == 0):
                    print("After {:5} epochs, training loss: {:.4f} ".format(epoch,epoch_loss))
            
            
            if(self.one_hot):
                self.saver.save(sess, "saved_models/predict_topomodel")
            else:
                self.saver.save(sess, "saved_models/predict_model")
                    
    def predict(self, inp_data):
        with tf.Session() as sess:
            #Loads the model and calculate the new states
            if(self.one_hot):
                self.saver.restore(sess, "saved_models/predict_topomodel")
            else:
                self.saver.restore(sess, "saved_models/predict_model")
            states = sess.run(self.state, feed_dict = {self.inp_var: inp_data, self.is_training: False})
        
        return states


    def calculate_score(self, inp_data, ground_truth ):
        states = self.predict(inp_data)
        if(self.one_hot):
            score = np.average(np.equal(states,ground_truth))
            return score
        else:
            diff = np.linalg.norm(states-ground_truth, axis=1)**2
            score = np.average(diff)
            std_error = np.std(diff) / len(diff)**0.5
        
            return score, std_error


    """Calculates the KNN-MSE score as in Unsupervised state representation learning with robotic priors: a
robustness benchmark ( Lesort 2017) 

    Assumes the input is flatten"""
    def knn_mse(self, inp_data, ground_truth, k = 20):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(inp_data)
        distances, indices = nbrs.kneighbors(inp_data)
        scores = [np.mean([np.sum((ground_truth[i] - ground_truth[y])**2) for y in idx[1:]]) for i, idx in enumerate(indices)]
        print("The mean KNN-MSE score is : ", np.mean(scores))
        return np.mean(scores)
