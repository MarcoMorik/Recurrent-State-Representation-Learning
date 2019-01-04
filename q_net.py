# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:56:12 2018

@author: marco
"""


import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib
import itertools
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import warnings
import datetime
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class q_net:
    def __init__(self, inp_dim, out_dim, learning_rate = 0.005, hidden_nodes = 512, middle_layer = True, regularization=1, reg_par = 0.001):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        self.x = tf.placeholder(tf.float32, shape=[None,self.inp_dim], name="inp_var")
        self.y = tf.placeholder(tf.float32, shape=[None], name="out_var")
        self.actions = tf.placeholder(tf.int32, shape=[None], name= "actions")
        self.q_net = self._make_model("_q_net",hidden_nodes=hidden_nodes, middle_layer=middle_layer, regu=regularization, l1_reg=reg_par)
        self.t_net = self._make_model("_t_net",hidden_nodes=hidden_nodes, middle_layer=middle_layer, regu=regularization, l1_reg=reg_par)
        
        self.q_val = self.q_net(self.x)
        self.t_val = self.t_net(self.x)
        
        self.collected = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        one_hot_actions = tf.one_hot(self.actions,self.out_dim)
        #distance = lambda x, y: tf.norm((x-y),ord=2, axis=1)**2
        q_action = tf.reduce_sum(tf.multiply(self.q_val,one_hot_actions), axis=1)
        graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_regularization_loss = tf.reduce_sum(graph_regularizers)
        
        self.loss = tf.reduce_mean( tf.square(q_action-self.y)) + total_regularization_loss
        
        tf.summary.histogram("Q-Values", self.q_val)
        tf.summary.scalar('Loss',self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradient = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(gradient)
        tf.summary.scalar('gradients',tf.norm(tf.concat([tf.reshape(gv[0],[-1]) for gv in gradient if gv[0]!= None],0),ord=2))
        
        
        var_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_save = []
        for var in var_col:
            if var.name.find("q_net") >= 0 or var.name.find("t_net") >= 0:
                var_save.append(var)
        self.saver = tf.train.Saver(var_save, max_to_keep=None)


        self.make_copy_op()
        self.merge = tf.summary.merge_all() 
        
    def _make_model(self, name_extra, hidden_nodes = 512, l1_reg = 0.001,middle_layer=True,regu=0):
        if(regu==1):        
            regularizers_inp = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}        
            regularizers_mid = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}        
            regularizers_out = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}        
        elif(regu==2):
            regularizers_inp = {"w": tf.contrib.layers.l2_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l2_regularizer(scale=l1_reg)}        
            regularizers_mid = {"w": tf.contrib.layers.l2_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l2_regularizer(scale=l1_reg)}        
            regularizers_out = {"w": tf.contrib.layers.l2_regularizer(scale=l1_reg),
                            "b": tf.contrib.layers.l2_regularizer(scale=l1_reg)}        
        
        if(regu==0):
            
            inp_layer = snt.Linear(hidden_nodes, name='1_layer'+name_extra)#, regularizers=regularizers_inp)
            out_layer = snt.Linear(self.out_dim, name='3_layer'+name_extra)#, regularizers= regularizers_out)        
        else:
                        
            inp_layer = snt.Linear(hidden_nodes, name='1_layer'+name_extra, regularizers=regularizers_inp)
            out_layer = snt.Linear(self.out_dim, name='3_layer'+name_extra, regularizers= regularizers_out)        
        noise_layer = lambda x:  x + tf.random_normal(shape=tf.shape(x), stddev=1e-9)
        if(middle_layer):
            if(regu==0):
                middle_layer = snt.Linear(hidden_nodes, name='2_layer'+name_extra)#, regularizers=regularizers_mid)
            else:
                middle_layer = snt.Linear(hidden_nodes, name='2_layer'+name_extra, regularizers=regularizers_mid)

            q_net = snt.Sequential([inp_layer,noise_layer, tf.nn.tanh,middle_layer, tf.nn.tanh, out_layer])
        else:
            q_net = snt.Sequential([inp_layer, tf.nn.tanh, out_layer])

        return q_net
    
    def save_model(self,sess, model_name = "saved_models/Q_model"):
        self.saver.save(sess,  model_name)
        
    def load_model(self,sess,model_name):
        self.saver.restore(sess,model_name)

    def learn(self, X, Y, actions, sess, summary=False):
        
        if(summary):
            _, loss, summary = sess.run([self.train_op, self.loss, self.merge], feed_dict={self.x: X, self.y: Y, self.actions: actions})
            return loss, summary
        else:
            _, loss = sess.run([self.train_op, self.loss], feed_dict={self.x: X, self.y:Y, self.actions: actions})
            return loss

    def predict(self, X, sess, target_net = False):
        if(target_net):
            return sess.run(self.t_val, feed_dict={self.x: X})
        else:
            return sess.run(self.q_val, feed_dict={self.x: X})

    def make_copy_op(self):
        var_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_t = []
        var_q = []
        copy_ops = []
        for var in var_col:
            if var.name.find("q_net") >= 0:
                var_q.append(var)
            if var.name.find("t_net") >= 0:
                var_t.append(var)

        for var in var_t:
            target = [v for v in var_q if v.name == var.name.replace("t_net", "q_net")][0]
            copy_ops.append(var.assign(target.value()))
        assert (len(var_t) == len(var_q) == len(copy_ops))
        self.copy_operation = tf.group(*copy_ops)

    def copy_to_t_net(self, sess):
        sess.run(self.copy_operation)
