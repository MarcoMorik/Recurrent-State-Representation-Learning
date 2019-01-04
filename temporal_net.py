# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:34:30 2017

@author: marco
"""
import numpy as np
import sonnet as snt
import tensorflow as tf
# import matplotlib
import itertools
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
# import warnings
from sklearn.decomposition import PCA

from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pandas.tools.plotting import scatter_matrix
# import Tensorflow
import datetime
import time
import coloring
import data_utils
# import validation_net
import os
import sys
import plotting
from skimage.measure import compare_ssim as ssim # Needs scikit-image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# BATCH_SIZE = 35
# SEQ_LEN = 25
SEQ_VARIATION = 10
# After how much iterations stop
EARLY_STOP = 150
SAVE_EPOCHS = True
# NUM_SIMILAR = 5
# NUM_BATCHES = 30
# NUM_EPOCHS = 2500
EXCLUDE = 10
INCLUDE_REPEAT = False
SEQ_VARIATION_BOOL = True
DEBUG = False
USE_DISSIMILAR = False
TOPOLOGICAL_SAMPLE = True
DATA_PATH = "/home/marco/Documents/recurrent_neural_baysian_filter"
# DATA_PATH = "/home/marcomorik/Documents/recurrent_neural_baysian_filter"
# DATA_PATH = "/home/marco/recurrent_neural_baysian_filter"
MAZE_DATA_PATH = DATA_PATH + "/maze_data/"
LANDMARKS_TOPONEIGHBORS = False
LANDMARK_POINTS = 20
class temporal_net:
    def __init__(self, obs_dim, state_dim, groundtruth_dim=3, seed=1, learning_rate=0.001, l1_reg=0.001,
                 num_epochs=1500, plot=False,
                 model_file="new_model", similar_points=20, maze=1, use_landmark=False, supervised=False,
                 hidden_nodes=512, dropout_prob=0.5, use_depth=True, landmark_error=0.0):
        """ Recurrent Neural Network for State Learning"""

        # Initialize Class Parameter
        self.LRANGEMODUS = 0
        self.mean_pose = np.zeros((1, 1, groundtruth_dim))
        self.std_pose = np.ones((1, 1, groundtruth_dim))
        self.state_step_size = np.ones(groundtruth_dim)
        self.img_width = int((obs_dim / (3 + use_depth)) ** 0.5)
        self.landmark_error = landmark_error
        print("THis run, we have {} Landmark error".format(self.landmark_error))
        assert (self.img_width ** 2 * (3 + use_depth)) == obs_dim, "Expected Square Image"
        self.mean_obs = np.zeros((1, 1, self.img_width, self.img_width, 3 + use_depth))
        self.std_obs = np.ones((1, 1, self.img_width, self.img_width, 3 + use_depth))
        self.state_step_size_n = self.state_step_size
        self.supervised = supervised
        self.maze = maze
        self.model_file = model_file
        self.plot = plot
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_similar = similar_points
        self.num_epochs = num_epochs
        self.use_landmarks = use_landmark
        self.droput_prob = dropout_prob
        self.mean_obs = np.zeros(self.obs_dim)
        self.std_obs = 1

        # seed random number generator
        self.rand = np.random.RandomState(seed)

        self.obs_var = tf.placeholder(tf.float32, shape=[None, None, self.obs_dim], name="obs_var")
        self.is_training = tf.placeholder(tf.bool, shape=[], name="train_cond")
        self.random_state_pair = tf.placeholder(tf.int32, shape=[None, 2])
        self.similar_state_pair = tf.placeholder(tf.int32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.action_var = tf.placeholder(tf.float32, shape=[None, None, groundtruth_dim], name="actions")
        # The position in the form x,y, sin(orientation),cos(orientation)
        self.real_pos = tf.placeholder(tf.float32, shape=[None, groundtruth_dim + 1], name="pose")
        self.supervised_learning = tf.placeholder(tf.bool, shape=[], name="supervised_learning")
        input_shape = tf.shape(self.obs_var)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        ######## DEFINE OBSERVATION STATE MAPPING ########
        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=l1_reg),
                        "b": tf.contrib.layers.l2_regularizer(scale=l1_reg)}
        """Conv-Net"""
        obs_encoder = snt.Sequential([
            snt.nets.ConvNet2D([16, 32, 64], [[3, 3]], [2], [snt.SAME], activate_final=True),
            snt.BatchFlatten(preserve_dims=1),
            lambda x: tf.nn.dropout(x, self.keep_prob),
            snt.Linear(hidden_nodes, name='Conv_to_LSTM'),
            tf.nn.relu
        ])
        lstm1_layer = snt.LSTM(hidden_nodes, name="LSTM1")
        lstm2_layer = snt.LSTM(hidden_nodes, name="LSTM2")
        skips = []
        # hidden_shape = tf.TensorShape([hidden_nodes+3])
        hidden_shape = tf.TensorShape([hidden_nodes + 3])
        skips.append(snt.SkipConnectionCore(lstm1_layer, input_shape=hidden_shape, name="skip_1"))
        # hidden_shape = tf.TensorShape([hidden_nodes+hidden_nodes+3])
        hidden_shape = tf.TensorShape([hidden_nodes + hidden_nodes + 3])
        skips.append(snt.SkipConnectionCore(lstm2_layer, input_shape=hidden_shape, name="skip_2"))
        lstm_core = snt.DeepRNN(skips, skip_connections=False, name="deep_lstm")
        output_layer = snt.Linear(output_size=state_dim, name="LSTM_to_out")

        ############ Apply Layers: ##########

        batch_input_layer = snt.BatchApply(obs_encoder)
        obs_reshaped = tf.reshape(self.obs_var, (input_shape[0], input_shape[1], 32, 32, 4))
        input_sequence = batch_input_layer(obs_reshaped)

        # Include the action vector
        input_sequence = tf.concat([input_sequence, self.action_var], axis=2)
        output_sequence, final_state = tf.nn.dynamic_rnn(
            cell=lstm_core,
            inputs=input_sequence,
            time_major=False,
            dtype=tf.float32)

        batch_output_layer = snt.BatchApply(output_layer)
        state_var = batch_output_layer(output_sequence)
        self.state = state_var

        # Possibility of a single run while Providing the hidden states of the LSTM (Used for RL)
        self.state_placeholder = tf.placeholder(tf.float32, [2, 2, 1, hidden_nodes])
        s = tf.unstack(self.state_placeholder, axis=0)
        rnn_tuple_state = tuple([tuple([s[i][0], s[i][1]]) for i in range(2)])
        single_out_seq, self.single_state = lstm_core(input_sequence[0], rnn_tuple_state)
        self.single_out = output_layer(single_out_seq)

        # DEFINE LOSS FUNCTION -----------------------------------------------------------------------------------------
        state_diff = state_var[:, EXCLUDE:-1, :] - state_var[:, EXCLUDE + 1:, :]
        action_exclude = tf.reshape(tf.reshape(self.action_var, [-1, seq_len, 3])[:, EXCLUDE + 1:, :], [-1, 3])

        state_diff_norm = tf.norm(state_diff, ord=2, axis=2)

        state_diff_norm_flatten = tf.reshape(state_diff_norm, [-1])
        # later_state = state_var[:,10:,:]
        state_flatten = tf.reshape(state_var, [-1, state_dim])
        similarity = lambda x, y: tf.exp(- tf.norm((x - y), ord=2, axis=1) ** 2)
        distance = lambda x, y: tf.norm((x - y), ord=2, axis=1) ** 2

        # Alpha depending on the loaded network
        # alpha = tf.constant(0.0)
        # alpha = tf.get_variable("alpha", dtype=tf.float32, initializer=tf.constant(2.0))
        alpha = tf.Variable(tf.constant(2.0), name="alpha", dtype=tf.float32)

        temp_loss_factor = 2
        variation_loss_factor = 25  # 9
        landmark_loss_factor = 1
        proportional_loss_factor = 0.04
        temp_coherence_loss = temp_loss_factor * tf.reduce_mean(state_diff_norm ** 2)
        variation_loss = variation_loss_factor * tf.reduce_mean(
            similarity(tf.gather(state_flatten, self.random_state_pair[:, 0]),
                       tf.gather(state_flatten, self.random_state_pair[:, 1])))

        reference_loss = tf.cond(tf.equal(tf.shape(self.similar_state_pair)[0], 0), lambda: 0.0,
                                 lambda: landmark_loss_factor * tf.reduce_mean(
                                     distance(tf.gather(state_flatten, self.similar_state_pair[:, 0]),
                                              tf.gather(state_flatten, self.similar_state_pair[:, 1]))))
        proportional_loss = proportional_loss_factor * tf.reduce_mean(
            (state_diff_norm_flatten - tf.scalar_mul(tf.exp(alpha), tf.norm(action_exclude, ord=2, axis=1))) ** 2)

        graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_regularization_loss = tf.reduce_sum(graph_regularizers)

        ############ Put losses together ############

        # compute a weighted sum of the loss terms
        self.supervised_loss = (
                tf.reduce_mean(distance(state_flatten[:, :4], self.real_pos)) + total_regularization_loss)

        self.unsupervised_loss = (
                proportional_loss + reference_loss + temp_coherence_loss + variation_loss + total_regularization_loss)

        self.loss = tf.cond(self.supervised_learning,
                            lambda: self.supervised_loss,
                            lambda: self.unsupervised_loss
                            )

        # TRAINING FUNCTIONS --------------------------------------------------------------------------------------------

        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradient = optimizer.compute_gradients(self.loss)

        reduce_gradient = lambda x: tf.norm(tf.concat([tf.reshape(gv[0], [-1]) for gv in x if gv[0] != None], 0), ord=2)

        self.gradient_temp = reduce_gradient(optimizer.compute_gradients(temp_coherence_loss))
        self.gradient_variation = reduce_gradient(optimizer.compute_gradients(variation_loss))
        self.gradient_reference = reduce_gradient(optimizer.compute_gradients(reference_loss))
        self.gradient_prop = reduce_gradient(optimizer.compute_gradients(proportional_loss))

        # Gradient clipping:
        splitted_gradients, variables = zip(*gradient)
        splitted_gradients, _ = tf.clip_by_global_norm(splitted_gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(splitted_gradients, variables))

        list_of_variables = []
        var_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in var_col:
            if (var.name.find("LSTM") >= 0 or var.name.find("alpha") >= 0 or var.name.find(
                    "conv_net_2d") >= 0 or var.name.find("_power") >= 0):
                list_of_variables.append(var)
        self.saver = tf.train.Saver(list_of_variables, max_to_keep=None)

    def learn(self, state_step_sizes, batch_size, num_batches, seq_len, pose, observations, actions, rewards):
        """Train the initalized temporal Network.
        The data must have the form: [num_episodes,episode_length,obs_dim]
        """

        self.seq_len = seq_len
        # Get the coloring of the states
        color_codex, color_codex_hex = get_positional_coloring(pose, maze=self.maze)

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we normalize the observations, organize the data into minibatches
        # and find pairs for the respective loss terms
        print(np.shape(observations), np.shape(actions), np.shape(rewards))

        self.mean_pose = np.mean(pose, axis=(0, 1), keepdims=True)
        self.std_pose = np.std(pose, axis=(0, 1), keepdims=True)
        self.state_step_size = np.copy(state_step_sizes)
        self.mean_obs = np.mean(observations, axis=(0, 1), keepdims=True)
        self.std_obs = np.std(observations, ddof=1)

        observations, actions, pose = self.normalize_data(observations, actions, pose)
        self.state_step_size_n = state_step_sizes
        self.state_step_size_n[:2] = (state_step_sizes[:2]) / np.std(pose, axis=(0, 1))[:2]

        # Save Mean file:
        np.savez(self.model_file+"_mean.npz", stepsize=self.state_step_size,
                 mean_obs=self.mean_obs, std_obs=self.std_obs)

        self.mean_obs = loaded_data['mean_obs']
        self.std_obs = loaded_data['std_obs']
        self.state_step_size = loaded_data['stepsize']

        # Initialize new landmarks, either at random or at fixed 8 points
        if (self.use_landmarks):
            if (self.num_similar == 8):
                self.init_selected_landmarks(self.mean_pose[0, 0, :])
            else:
                self.init_positional_landmarks(pose, self.num_similar, TOPOLOGICAL_SAMPLE)

            plot_points = np.concatenate(
                (np.reshape(pose, (pose.shape[0] * pose.shape[1], 3))[:, :2], self.landmarks[:, :2]))
            color_codex_landmark = np.concatenate((color_codex, np.zeros((self.num_similar, 3))))
            with open(("plots/" + self.model_file + "/landmarks.txt"), 'wb') as write_file:
                np.savetxt(write_file, self.landmarks, delimiter=",")

        else:
            plot_points = np.reshape(pose, (pose.shape[0] * pose.shape[1], 3))[:, :2]
            color_codex_landmark = color_codex

        if (self.plot):
            plot_representation(plot_points, color_codex_landmark, add_colorbar=False, name="Ground truth poses",
                                filename="plots/" + self.model_file + "/colored_poses.png",
                                axis_labels=["X-Cordinate", "Y-Cordinate"])

        # Split in Validation and Training Set
        n_val = int(np.shape(observations)[0] * 0.1)
        vali_index = np.random.permutation(np.arange(0, np.shape(observations)[0]))

        validation_orig_set = {'obs': observations[vali_index[-n_val:]], 'pose': pose[vali_index[-n_val:]],
                               'action': actions[vali_index[-n_val:]]}

        batch_iterator = make_batch_iterator(observations[vali_index[:-n_val]], pose[vali_index[:-n_val]],
                                             actions[vali_index[:-n_val]], batch_size, seq_len,
                                             fixed_seq=SEQ_VARIATION_BOOL, seq_variation=SEQ_VARIATION)

        # Preprocess the Reference Point loss
        validation_orig_batch = self.batch_preperation(
            validation_orig_set['obs'][:min(n_val, 30), :50], validation_orig_set['pose'][:min(n_val, 30), :50],
            validation_orig_set['action'][:min(n_val, 30), :50])

        # TRAINING
        #EARLY_STOP
        gradients = []
        losses = []
        min_loss = np.inf
        min_loss_count = 0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            epoch_train_time = 0
            epoch_prep_time = 0
            saved_landmarks = 0
            for epoch in range(self.num_epochs):
                epoch_loss = 0
                epoch_batches = 0
                grad = [0, 0, 0, 0]
                loss = [0, 0]
                average_dist = 0
                average_number = 0
                for i in range(num_batches):
                    time_before_prep = time.time()
                    obs_batch, pose_batch, action_batches = next(batch_iterator)
                    obs_batch, pose_extended, action_batches, random_variation, landmarks = self.batch_preperation(
                        obs_batch, pose_batch, action_batches)
                    if (np.shape(landmarks)[0]):
                        average_dist += np.mean(
                            np.linalg.norm(pose_extended[landmarks[:, 0], :] - pose_extended[landmarks[:, 1], :],
                                           axis=1))
                        average_number += np.shape(landmarks)[0]

                    epoch_prep_time += time.time() - time_before_prep
                    time_before_train = time.time()
                    _, tmp_loss, sup_loss, unsup_loss, grad_land, grad_temp, grad_vari, grad_prop, tmp_state = sess.run(
                        [
                            self.train_op, self.loss, self.supervised_loss, self.unsupervised_loss,
                            self.gradient_reference, self.gradient_temp, self.gradient_variation, self.gradient_prop,
                            self.state],
                        feed_dict={
                            self.obs_var: obs_batch,
                            self.random_state_pair: random_variation,
                            self.similar_state_pair: landmarks,
                            self.is_training: True,
                            self.keep_prob: self.droput_prob,
                            self.action_var: action_batches,
                            self.supervised_learning: self.supervised,
                            self.real_pos: pose_extended})

                    epoch_train_time += time.time() - time_before_train
                    grad = [grad[0] + grad_land, grad[1] + grad_temp, grad[2] + grad_vari, grad[3] + grad_prop]
                    loss = [loss[0] + sup_loss, loss[1] + unsup_loss]
                    if (np.isnan(epoch_loss)):
                        print("There is a nan in the loss")
                        return

                    epoch_loss += tmp_loss
                    epoch_batches += 1

                    #Save the model every 20 Epochs
                    if(epoch % 20==0 and SAVE_EPOCHS):
                        self.saver.save(sess, "saved_models/" + self.model_file.replace("/training", "Epoch{}".format(epoch)))

                    # Plot the Position and Representation of a single Batch during Training
                    if (epoch % 50 == 0 and False and self.plot and i % 1 == 0):

                        pose_plot = pose_batch[:, :, :2] * self.std_pose[:, :, :2] + self.mean_pose[:, :, :2]
                        color_codex_batch, _ = get_sequential_coloring(pose_plot, flatten_inp=False)
                        pose_plot = flatten_seq(pose_plot[:, EXCLUDE:, :])
                        landmarks_plot = [[flatten_ind_remove_10(x, seq_len), flatten_ind_remove_10(y, seq_len)] for
                                          x, y in landmarks]
                        landmarks_plot = np.asarray([[x, y] for x, y in landmarks_plot if x != None and y != None])
                        if (len(landmarks_plot > 0)):
                            print("average distance ", np.mean(
                                np.linalg.norm(pose_plot[landmarks_plot[:, 0], :] - pose_plot[landmarks_plot[:, 1], :],
                                               axis=1)), "with ", len(landmarks_plot), " Landmarks")


                        wallplotts = lambda figname: plotting.plot_maze(figure_name=figname)
                        if(self.use_landmarks):
                            nonnormedlandmarks = self.landmarks[:, :2] * self.std_pose[0, :, :2] + self.mean_pose[0, :, :2]
                        else:
                            nonnormedlandmarks = None
                        plot_representation(pose_plot[:, :2], color_codex_batch,
                                            add_colorbar=False,
                                            name="Position and landmarks (Training Data), \n Epoch {}".format(epoch + 1),
                                            axis_labels=["X - Coordinates", "Y - Coordinates"],
                                            filename="plots/" + self.model_file + "/PoseLandmarksEpoch{:3}Batch{:3}FirstTwoDimensions.png".format(
                                                epoch + 1, i), seq_connect=seq_len - EXCLUDE, landmarks=landmarks_plot,
                                            plotting_walls=wallplotts, landmark_points=nonnormedlandmarks, alpha=0.6)

                        plot_representation(flatten_seq(tmp_state[:, EXCLUDE:, :2]), color_codex_batch,
                                            add_colorbar=False,
                                            name="Learned State Representation (Training Data), \n Epoch {}".format(epoch + 1),
                                            filename="plots/" + self.model_file + "/StateLandmarksEpoch{:3}Batch{:3}FirstTwoDimensions.png".format(
                                                epoch + 1, i), seq_connect=seq_len - EXCLUDE, landmarks=landmarks_plot,
                                            alpha=0.6)

                        """
                        plot_representation(flatten_seq(tmp_state[:, EXCLUDE:, :2]), color_codex_batch, add_colorbar=False,
                                            name="Learned State Representation (Training Data)".format(epoch + 1),
                                            filename="plots/" + self.model_file + "/LandmarksEpoch{:3}Batch{:3}FirstTwoDimensions.png".format(
                                                epoch + 1, i), seq_connect=-EXCLUDE, landmarks=landmarks_plot)
                        plot_representation(flatten_seq(tmp_state[:, EXCLUDE:, :2]), color_codex_batch, add_colorbar=False,
                                            name="Learned State Representation (Training Data)".format(epoch + 1),
                                            filename="plots/" + self.model_file + "/SequencesEpoch{:3}Batch{:3}FirstTwoDimensions.png".format(
                                                epoch + 1, i), seq_connect=seq_len - EXCLUDE)"""
                gradients.append(grad)

                # print("This epoch, there were ", saved_landmarks, " saved landmarks " )
                # print("Gradients : " , grad)
                saved_landmarks += average_number
                if (epoch % 10 == 0):
                    print("Average distance to a landmark: ", average_dist / num_batches)
                    print(
                        "Epoch {:3}, Preperation time: {:.2f}, Training time: {:.2f}".format(epoch + 1, epoch_prep_time,
                                                                                             epoch_train_time))

                    epoch_train_time = 0
                    epoch_prep_time = 0

                # Representation of the Validation Dataset
                obs_batch, pose_extended, action_batches, random_variation, landmarks = validation_orig_batch
                epoch_loss = sess.run(self.loss, feed_dict={
                    self.obs_var: obs_batch,
                    self.random_state_pair: random_variation,
                    self.similar_state_pair: landmarks,
                    self.is_training: False,
                    self.keep_prob: 1,
                    self.action_var: action_batches,
                    self.supervised_learning: self.supervised,
                    self.real_pos: pose_extended})
                loss.append(epoch_loss)

                if (epoch_loss <= min_loss):
                    min_loss = epoch_loss
                    min_loss_count = 0
                else:
                    min_loss_count += 1

                ######### Print epoch Statistics ###########
                losses.append(loss)
                if (epoch + 1) % 25 == 0:
                    if (self.supervised):
                        filename = "SupervisedLearning"
                    else:
                        filename = "Unsupervised"
                    print("Epoch {:3}/{}, loss:{:.4f}".format(epoch + 1, self.num_epochs, epoch_loss / epoch_batches))
                    print(
                        "Gradients: landmark: {:.4f}, Temporal: {:.4f}, Variation: {:.4f}, Proportion: {:.4f}".format(
                            grad[0], grad[1], grad[2], grad[3]))
                    if (self.plot):
                        self.epoch_statistics(observations, actions, sess, color_codex, filename, epoch)

                # Check if the validation loss is not decreasing, if so break
                if (min_loss_count >= EARLY_STOP):  # and epoch >= 750):
                    print("Training finished after {} Epochs".format(epoch))
                    break
            # Save Results
            self.saver.save(sess, "tmp/model.ckpt")
            self.saver.save(sess, "saved_models/" + self.model_file.replace("/training", ""))
            if (self.plot):
                plot_gradients(gradients, "plots/" + self.model_file + "/gradients.png")
                plt.close('all')
                plt.plot(list(range(len(losses))), losses)
                if (self.supervised):
                    plt.legend(["supervised", "unsupervised", "supervised validation"], loc="upper right")
                else:
                    plt.legend(["supervised", "unsupervised", "unsupervised validation"], loc="upper right")
                plt.savefig("plots/" + self.model_file + "/losses.png", bbox_inches="tight")

            np.savetxt("plots/" + self.model_file + "/gradients.txt", np.asarray(gradients), delimiter=",")
            np.savetxt("plots/" + self.model_file + "/losses.txt", np.asarray(losses), delimiter=",")

            predicted_state = self.phi_and_plot(observations, actions, sess, color_codex,
                                                "FinalEpoch{}".format(epoch + 1))
        return predicted_state

    def epoch_statistics(self, observations, actions, sess, color_codex, filename, epoch):
        # Optionally plot the current state space
        n = np.shape(observations)[0]
        # Splitting the computation reduces the GPU memory requirements
        states1 = sess.run(self.state, feed_dict={self.obs_var: observations[:int(n / 2), :, :],
                                                  self.action_var: actions[:int(n / 2), :, :], self.keep_prob: 1.0,
                                                  self.is_training: False})
        states2 = sess.run(self.state, feed_dict={self.obs_var: observations[int(n / 2):, :, :],
                                                  self.action_var: actions[int(n / 2):, :, :], self.keep_prob: 1.0,
                                                  self.is_training: False})
        current_state = np.concatenate((states1, states2), axis=0)

        color_codex = remove_first_10(color_codex, flatten_inp=True, seq_len=np.shape(current_state)[1])
        current_state = remove_first_10(current_state)
        current_state = np.reshape(current_state, (current_state.shape[0] * current_state.shape[1], self.state_dim))
        plot_representation_pca(current_state, color_codex, add_colorbar=False,
                                name="Learned State Representation (Training Data)".format(epoch + 1),
                                filename="plots/" + self.model_file + "/" + filename + "Learningepoch{:3}.png".format(
                                    epoch + 1))
        plot_representation(current_state[:, :2], color_codex, add_colorbar=False,
                            name="Learned State Representation (Training Data), \n Epoch {}".format(epoch + 1),
                            filename="plots/" + self.model_file + "/FirstTwoDim" + filename + "Learningepoch{:3}.png".format(
                                epoch + 1), axis_labels=['State dimension 1', 'State dimension 2'], axis_range=[-3,3,-3,3])
        plot_representation(current_state[:, 2:4], color_codex, add_colorbar=False,
                            name="Learned State Representation (Training Data), \n Epoch {}".format(epoch + 1),
                            filename="plots/" + self.model_file + "/LastTwoDim" + filename + "Learningepoch{:3}.png".format(
                                epoch + 1), axis_labels=['State dimension 3', 'State dimension 4'], axis_range=[-3,3,-3,3])

    def phi_and_plot(self, observations, actions, sess, color_codex, filename, exclude=EXCLUDE):
        """ Calculates states and plot those with already normalized data"""

        n = np.shape(observations)[0]
        states1 = sess.run(self.state, feed_dict={self.obs_var: observations[:int(n / 2), :, :],
                                                  self.action_var: actions[:int(n / 2), :, :], self.keep_prob: 1.0,
                                                  self.is_training: False})
        states2 = sess.run(self.state, feed_dict={self.obs_var: observations[int(n / 2):, :, :],
                                                  self.action_var: actions[int(n / 2):, :, :], self.keep_prob: 1.0,
                                                  self.is_training: False})
        predicted_state = np.concatenate((states1, states2), axis=0)
        if (exclude):
            color_codex = remove_first_10(color_codex, flatten_inp=True, seq_len=np.shape(predicted_state)[1])
            predicted_state = remove_first_10(predicted_state)
        predicted_state = flatten_seq(predicted_state)
        if (self.plot):
            plot_representation_matrix(predicted_state, color_codex,
                                       filename=r"plots/" + self.model_file + "/" + filename + "scatter_matrix" + datetime.datetime.today().strftime(
                                           '%Y-%m-%d') + ".png")
        return predicted_state

    def phi(self, observations, actions, modelpath="tmp/model.ckpt", exclude=EXCLUDE,
            meanfile="saved_models/mixinfo.npz"):
        """Loads a model and normalize data, then calculates the state"""

        loaded_data = np.load(meanfile)
        self.mean_obs = loaded_data['mean_obs']
        self.std_obs = loaded_data['std_obs']
        self.state_step_size = loaded_data['stepsize']
        observations_n, actions_n = self.normalize_data(observations, actions)
        """print(self.mean_obs, " Mean")
        print(self.std_obs, "std")
        print(self.state_step_size, "stepsize")
        observations_n = (observations - self.mean_obs) / self.std_obs
        #       observations = (observations - self.mean_obs) / self.std_obs
        actions_n = actions / self.state_step_size"""
        n, s = np.shape(observations)[0:2]
        print("number of sequences : ", n)
        states_single = np.zeros((n, s, 4))
        with tf.Session() as sess:
            # Loads the model and calculate the new states
            self.saver.restore(sess, modelpath)
            states = sess.run(self.state,
                              feed_dict={self.obs_var: observations_n[0:1, :, :], self.action_var: actions_n[0:1, :, :],
                                         self.keep_prob: 1.0, self.is_training: False})
            for i in range(1, n):
                states1 = sess.run(self.state, feed_dict={self.obs_var: observations_n[i:i + 1, :, :],
                                                          self.action_var: actions_n[i:i + 1, :, :],
                                                          self.keep_prob: 1.0, self.is_training: False})
                # states1 = sess.run(self.state, feed_dict = {self.obs_var: observations_n[:int(n/2),:,:],self.action_var: actions_n[:int(n/2),:,:], self.keep_prob:  1.0, self.is_training: False})
                # states2 = sess.run(self.state, feed_dict = {self.obs_var: observations_n[int(n/2):,:,:],self.action_var: actions_n[int(n/2):,:,:], self.keep_prob:  1.0, self.is_training: False})
                states = np.concatenate((states, states1), axis=0)
        if (exclude):
            states = states[:, exclude:, :]
        return states

    def phi_step_with_session(self, observation, action, sess, lstm_state=np.zeros((2, 2, 1, 128))):
        # print(np.mean(self.mean_obs))
        observation = (observation[np.newaxis, np.newaxis, :] - self.mean_obs) / self.std_obs
        """plt.imshow(np.reshape(observation,(32,32,4))[:,:,:3]/255.)
        print(np.reshape(observation,(32,32,4))[10,15,:])        
        sys.stdout.flush()
        plt.pause(.01)
        plt.draw()
        """
        action = action[np.newaxis, np.newaxis, :] / self.state_step_size
        states, lstm_state = sess.run([self.single_out, self.single_state],
                                      feed_dict={self.obs_var: observation, self.action_var: action,
                                                 self.keep_prob: 1.0, self.is_training: False,
                                                 self.state_placeholder: lstm_state})
        # print(states)
        return states, lstm_state

    def phi_with_session(self, observations, actions, sess, exclude=EXCLUDE):
        observations, actions = self.normalize_data(observations, actions)
        """observations = (observations - self.mean_obs) / self.std_obs
        actions = actions / self.state_step_size"""
        states = sess.run(self.state,
                          feed_dict={self.obs_var: observations, self.action_var: actions, self.keep_prob: 1.0,
                                     self.is_training: False})
        if (exclude):
            states = states[:, exclude:, :]
        return states

    def restore_model(self, sess, modelpath, meanfile):
        self.saver.restore(sess, modelpath)
        loaded_data = np.load(meanfile)
        self.mean_obs = loaded_data['mean_obs']
        self.std_obs = loaded_data['std_obs']
        self.state_step_size = loaded_data['stepsize']

    # get Random Landmarks
    def init_positional_landmarks(self, inp_pose, num_landmarks, topological_sampling=True):
        real_pose = (inp_pose * self.std_pose) + self.mean_pose
        pose = np.reshape(inp_pose, (np.prod(np.shape(inp_pose)[:2]), np.shape(inp_pose)[2]))
        real_pose = np.reshape(real_pose, (np.prod(np.shape(real_pose)[:2]), np.shape(real_pose)[2]))
        if (topological_sampling):
            topo_color = coloring.Coloring(self.maze)

            print(np.shape(real_pose))
            topo_indieces = topo_color.get_topo_pose_index(real_pose)
            n_tiles = np.max(topo_indieces) + 1
            print("There are {} Tiles to select for landmarks ".format(n_tiles))
            topo_dict = dict.fromkeys(np.arange(n_tiles))
            for key in topo_dict.keys():
                topo_dict[key] = []
            for i in range(len(pose)):
                topo_dict[topo_indieces[i]].append(pose[i, :])

            topo_indieces = np.random.choice(n_tiles, size=num_landmarks % n_tiles, replace=False)
            landmarks = []
            landmark_per_tile = int(num_landmarks / n_tiles)
            for i in range(n_tiles):
                if (len(topo_dict[i]) < landmark_per_tile + 1):
                    print("to less points in this area")
                    continue
                landmark_indices = []
                if i in topo_indieces:
                    landmark_indices = np.random.choice(len(topo_dict[i]), size=landmark_per_tile + 1, replace=False)
                elif landmark_per_tile >= 1:
                    landmark_indices = np.random.choice(len(topo_dict[i]), size=landmark_per_tile, replace=False)

                for j in landmark_indices:
                    landmarks.append(topo_dict[i][j])
            self.landmarks = np.asarray(landmarks)

        else:
            landmark_indices = np.random.choice(len(pose), size=num_landmarks, replace=False)
            self.landmarks = pose[landmark_indices]
        # self.landmark_range = np.asarray([80./mean_pose[0],80./mean_pose[1],0.5*np.pi]) # Before 40, 40, 0.25 pi

        self.topological_distances = coloring.Coloring(maze=self.maze)
        self.topological_distances.vectorize_topo_map()

        if (self.LRANGEMODUS == 0):
            self.landmark_range = np.asarray([40. / self.std_pose[0, 0, 0], 40. / self.std_pose[0, 0, 1], 0.5 * np.pi])
        elif (self.LRANGEMODUS == 1):
            self.landmark_range = np.asarray(
                [self.state_step_size_n[0], self.state_step_size_n[1], self.state_step_size_n[2]])
        elif (self.LRANGEMODUS == 2):
            self.landmark_range = np.asarray([1000. / self.std_pose[0, 0, 0], 500. / self.std_pose[0, 0, 1], 2 * np.pi])
        elif (self.LRANGEMODUS == 3):
            self.landmark_range = np.asarray([40. / self.std_pose[0, 0, 0], 40. / self.std_pose[0, 0, 1], 2 * np.pi])
        print(self.landmarks)

    def init_selected_landmarks(self, mean_pose):

        # all_landmarks = np.asarray([[x,y,np.random.rand()*np.pi*2 - np.pi]  for x in range(50,1000,50) for y in range(50,600,50)])
        # all_landmarks[:,:2] = ( all_landmarks[:,:2] - self.mean_pose[0,0,:] ) / self.std_pose[0,0,:]
        self.landmarks = np.asarray([[-1.5, -0.1, -1.5],
                                     [-0.5, -1.5, 0.0],
                                     [-1.0, 1.3, 1.5],
                                     [0.3, 1.3, 3.0],
                                     [0.0, 0.0, -1.5],
                                     [0.9, -0.8, -3],
                                     [1.5, -1.5, 2.0],
                                     [1.3, 1.2, 0.0]
                                     ])
        # self.landmark_range = np.asarray([80./mean_pose[0],80./mean_pose[1],0.5*np.pi])
        self.landmark_range = np.asarray([60. / self.std_pose[0, 0, 0], 60. / self.std_pose[0, 0, 1], 0.5 * np.pi])
        print(self.landmarks)

        self.topological_distances = coloring.Coloring(maze=self.maze)
        self.topological_distances.vectorize_topo_map()

    def get_positional_landmarks(self, pose, num_points=20):
        """ Input: an array of poses
        Output: All pairs of points which are looking at the same landmark"""

        points_with_landmark = set([])
        _, seq_len = np.shape(pose)[:2]
        if EXCLUDE:
            pose = remove_first_10(pose, flatten_inp=False, seq_len=seq_len)
            seq_len -= EXCLUDE
        pose = np.reshape(pose, (np.prod(np.shape(pose)[:2]), np.shape(pose)[2]))
        n = np.shape(pose)[0]
        m = np.shape(self.landmarks)[0]
        dist_to_landmark = np.zeros((n, m))
        diff = pose[:, np.newaxis, :] - self.landmarks[np.newaxis, :, :]
        diff[:, :, 2] = ((diff[:, :, 2] + np.pi) % (2 * np.pi)) - np.pi
        dist_to_landmark = np.max((diff / self.landmark_range[np.newaxis, np.newaxis, :]) ** 2, axis=2)

        # self.save_landmarks = []
        for i in range(m):
            indices = np.where(dist_to_landmark[:, i] <= 1)
            points_with_landmark = points_with_landmark.union(set(itertools.combinations(indices[0], 2)))
            # print(list(itertools.combinations(indices[0],2)))
            # self.save_landmarks.extend(pose[indices])
        # print(np.shape(self.save_landmarks))
        points_with_landmark = list(set([tuple(sorted((x, y))) for x, y in points_with_landmark if
                                         int(x / seq_len) != int(y / seq_len)]))
        dissimilar_points = []
        if (len(points_with_landmark) == 0):
            points_with_landmark = np.empty((0, 2))
        elif (num_points < len(points_with_landmark)):
            # distances = [distance_with_angles(pose[x, :], pose[y, :], self.landmark_range) for x, y in
            #             points_with_landmark]
            distances = [distance_with_angles(pose[x, :], pose[y, :], self.state_step_size_n) for x, y in
                         points_with_landmark]
            indices = np.argsort(distances)

            sorted_points = np.asarray(points_with_landmark)[indices, :]
            # all pairs of points from the selected ones, which belong to different landmarks
            #dissimilar_points = np.asarray(
            #    [[x, y] for i in range(2) for j in range(i, 2) for x in sorted_points[:num_points, i] for y in
            #     sorted_points[:num_points, j] if
            #     sum((dist_to_landmark[x, :] <= 1) != (dist_to_landmark[y, :] <= 1)) >= 2])
            if (LANDMARKS_TOPONEIGHBORS):
                points_with_landmark = []
                for i in range(np.shape(sorted_points)[0]):
                    a = pose[sorted_points[i, 0]] * self.std_pose[0, 0, :] + self.mean_pose[0, 0, :]
                    b = pose[sorted_points[i, 1]] * self.std_pose[0, 0, :] + self.mean_pose[0, 0, :]
                    if (self.topological_distances.get_topo_distance_by_pose(a, b) < 2):
                        points_with_landmark.append(sorted_points[i])
                    if (len(points_with_landmark) >= num_points):
                        break
                points_with_landmark = np.asarray(points_with_landmark)
            else:
                points_with_landmark = np.asarray(sorted_points[:num_points, :])

        if self.landmark_error > 0:
            rand_land = np.random.rand(num_points)
            print("There are {} points selected random".format(np.sum(rand_land < self.landmark_error)))
            points_with_landmark = [points_with_landmark[i] if rand_land[i] >= self.landmark_error else tuple(
                np.random.choice(n, 2, replace=False)) for i in range(len(points_with_landmark))]

        if len(points_with_landmark) and EXCLUDE:
            points_with_landmark = [(int(i / (seq_len) + 1) * EXCLUDE + i, int(j / (seq_len) + 1) * EXCLUDE + j) for
                                    i, j in points_with_landmark]
            dissimilar_points = [(int(i / (seq_len) + 1) * EXCLUDE + i, int(j / (seq_len) + 1) * EXCLUDE + j) for
                                 i, j in dissimilar_points]
        return np.asarray(points_with_landmark), dissimilar_points

    def normalize_data(self, obs, actions, pose=[]):
        obs_n = (obs - self.mean_obs) / self.std_obs
        actions_n = actions / self.state_step_size
        if (len(pose) >= 1):
            pose[:, :, :2] = (pose[:, :, :2] - self.mean_pose[:, :, :2]) / self.std_pose[:, :, :2]
            return obs_n, actions_n, pose
        return obs_n, actions_n

    def batch_preperation(self, obs_batch, pose_batch, action_batches):
        landmark_points = []
        # Setting the first action to zeros, since there was no last state
        action_batches[:, 0] = np.zeros(np.shape(action_batches)[2])
        batch_size, seq_len = np.shape(obs_batch)[:2]

        pose_flatten = np.reshape(pose_batch, (batch_size * seq_len, 3))
        pose_extended = np.concatenate(
            (pose_flatten[:, :2], [[np.cos(x), np.sin(x)] for x in pose_flatten[:, 2]]), axis=1)

        non_exclude_indices = [i for i in range(0, (batch_size * seq_len)) if i % seq_len >= EXCLUDE or (not EXCLUDE)]

        random_variation = np.array(
            [[i, np.random.choice([j for j in non_exclude_indices if j != i])] for i in non_exclude_indices],
            dtype='int32')
        # random_variation = np.array(
        #    [[i, np.random.choice(list(range(0, i)) + list(range(i + 1, batch_size * seq_len)))] for i in
        #     range(0, (batch_size * seq_len)) if i % seq_len >= EXCLUDE or (not EXCLUDE)],
        #    dtype='int32')
        same_time_variation = np.array([[i, np.random.choice(
            list(range(i % seq_len, i, seq_len)) + list(range(i + seq_len, batch_size * seq_len, seq_len)))]
                                        for i in non_exclude_indices],
                                       dtype='int32')

        if (self.use_landmarks):
            # New Landmarks
            landmarks, dissimilar_points = self.get_positional_landmarks(pose_batch, num_points=LANDMARK_POINTS)
        else:
            # Old landmarks
            # landmarks = np.asarray(get_similar_points(pose_batch,self.num_similar, self.state_step_size_n))
            landmarks = np.asarray(self.get_similar_point_knn(pose_batch, self.num_similar, self.state_step_size_n))
            #landmarks = np.asarray(self.get_similar_point_ssim(obs_batch, self.num_similar)) # TODO testwise test image comparison

            print("Mean Distance of points with similar images", np.mean([ np.linalg.norm(pose_extended[i] - pose_extended[j]) for i,j in landmarks]))
            dissimilar_points = []

        if (len(dissimilar_points) >= 1 and USE_DISSIMILAR):
            random_variation = np.concatenate((random_variation, same_time_variation, dissimilar_points),
                                              axis=0)
        else:
            random_variation = np.concatenate((random_variation, same_time_variation), axis=0)
        return obs_batch, pose_extended, action_batches, random_variation, landmarks


    def get_similar_point_ssim(self, inp_img, num_points):
        ssim_time = time.time()
        seq_len = np.shape(inp_img)[1]
        batch_size = np.shape(inp_img)[0]

        if (EXCLUDE):
            img = inp_img[:, EXCLUDE:, :]
            seq_len = seq_len - EXCLUDE

        else:
            img = inp_img

        img = np.reshape(img, (np.prod(np.shape(img)[:2]), 32,32,4))

        n = np.shape(img)[0]
        assert (num_points <= n * (n - 1) / 2)
        dist_matrix = np.ones((n, n)) * -1.
        for i in range(n):
            for j in range(i, n):
                if( int(i/seq_len) != int(j /seq_len)):
                    dist_matrix[i, j] = ssim(img[i], img[j], multichannel=True)



        dist_matrix_flatten = np.reshape(dist_matrix, n * n)

        sorted_indices = np.argsort(dist_matrix_flatten)[-num_points:]
        closest = list(zip(*np.unravel_index(sorted_indices, (n, n))))

        #Retransform the indices back to full length
        if (EXCLUDE):
            closest = [(int(i / (seq_len) + 1) * EXCLUDE + i, int(j / (seq_len) + 1) * EXCLUDE + j) for i, j in closest]


        #print("One Batch SSIM took: ", time.time()-ssim_time)
        return closest

    def get_similar_point_knn(self, inp_pose, num_points, state_step_sizes):
        seq_len = np.shape(inp_pose)[1]
        batch_size = np.shape(inp_pose)[0]

        if EXCLUDE:
            pose = inp_pose[:, EXCLUDE:, :]
            seq_len = seq_len - EXCLUDE

        else:
            pose = inp_pose
        pose = np.reshape(pose, (np.prod(np.shape(pose)[:2]), np.shape(pose)[2])) / state_step_sizes
        nbrs = NearestNeighbors(n_neighbors=num_points / 2, algorithm='auto').fit(pose)
        distances, indices = nbrs.kneighbors(pose)
        distances = distances[:, :]
        indices = indices[:, :]
        tmp_i = zip(*np.unravel_index(np.argsort(distances, axis=None), np.shape(distances)))

        closest = [(i, indices[i, j]) for i, j in tmp_i if
                   int(i / seq_len) != int(indices[i, j] / seq_len) and indices[i, j] < i][:num_points]

        #Replace a pair with a random pair with landmark_error probability
        if self.landmark_error > 0:
            rand_land = np.random.rand(num_points)
            closest = [closest[i] if rand_land[i] >= self.landmark_error else tuple(np.random.choice(seq_len*batch_size,2,replace=False)) for i in range(num_points)]
        # Need to transfer the closest indices now to the real indice( without Excluded first)
        if EXCLUDE:
            closest = [(int(i / (seq_len) + 1) * EXCLUDE + i, int(j / (seq_len) + 1) * EXCLUDE + j) for i, j in closest]
        # print_pose = np.reshape(inp_pose, (np.prod(np.shape(inp_pose)[:2]),np.shape(inp_pose)[2])) /state_step_sizes
        # print([np.linalg.norm(print_pose[i]-print_pose[j]) for i,j in closest])
        return closest

def distance_with_angles(a, b, normalization=[1.0, 1.0, 1.0]):
    diff = a - b
    diff[2] = ((diff[2] + np.pi) % (2 * np.pi)) - np.pi
    return np.sum((diff / normalization) ** 2)


def distance_between_two_angles(a, b):
    return (((a - b) + np.pi) % (2 * np.pi)) - np.pi



def make_batch_iterator(obs, pose=None, actions=None, batch_size=32, seq_len=20, fixed_seq=True, seq_variation=10):
    # go through data and select a subsequence from each sequence
    # seed list allows to go through a list of repeating batches, a seed list of length 100 will return the first batch again at run 101
    current_seq_len = seq_len
    while True:
        if (not fixed_seq):
            current_seq_len = np.random.randint(seq_len - seq_variation, seq_len + seq_variation)

        # episodes = np.random.random_integers(0, len(obs) - 1, size=batch_size)
        episodes = np.random.choice(len(obs), size=batch_size, replace=False)
        start_steps = np.random.random_integers(0, len(obs[0]) - current_seq_len - 1, size=batch_size)
        # print('episodes', episodes)
        # print('start_steps', start_steps)
        obs_batches = np.concatenate([obs[i:i + 1, j:j + current_seq_len] for i, j in zip(episodes, start_steps)])
        # if( (pose != None).all() and (actions != None).all()):
        pose_batches = np.concatenate([pose[i:i + 1, j:j + current_seq_len] for i, j in zip(episodes, start_steps)])
        action_batches = np.concatenate(
            [actions[i:i + 1, j:j + current_seq_len] for i, j in zip(episodes, start_steps)])
        yield obs_batches, pose_batches, action_batches
        # else:
        #    yield obs_batches


def flatten_seq(data):
    return np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))


def remove_first_10(data, flatten_inp=False, seq_len=20):
    data = np.asarray(data)
    if (flatten_inp):
        n = int(np.shape(data)[0] / seq_len)
        obs_dim = np.shape(data)[1]
        data_expanded = np.reshape(data, (n, seq_len, obs_dim))
        return np.reshape(data_expanded[:, EXCLUDE:, :], (n * (seq_len - EXCLUDE), obs_dim))
    else:
        return data[:, EXCLUDE:, :]


def flatten_ind_remove_10(indice, seq_len):
    if (indice % seq_len < EXCLUDE):
        return None
    seq_num = int(indice / seq_len)
    return int(indice - EXCLUDE * (seq_num + 1))


#Plotting Functions:

def plot_gradients(gradient, filename):
    gradient = np.asarray(gradient)
    x_axis = list(range(np.shape(gradient)[0]))
    plt.close('all')
    plt.plot(x_axis, gradient)
    plt.legend(["landmark", "temporal", "variation", "proportional"], loc="upper right")
    plt.savefig(filename, bbox_inches="tight")


def plot_representation(states, color_codex, name="Learned State Representation", add_colorbar=False, filename=None,
                        seq_connect=-1, landmarks=[], axis_labels=['State dimension 1', 'State dimension 2'],
                        plotting_walls=None, landmark_points=None, alpha=0.4, axis_range=None):
    # plt.ion()
    plt.figure(name)
    plt.hold(True)
    plt.scatter(states[:, 0], states[:, 1], s=4, c=color_codex, cmap='rgb', linewidths=0.1, alpha=alpha)
    if (seq_connect > 0):
        for i in range(int(np.shape(states)[0] / seq_connect)):
            # assume sequential coloring for sequence color
            plt.plot(states[i * seq_connect:(i + 1) * seq_connect, 0], states[i * seq_connect:(i + 1) * seq_connect, 1],
                     "--", linewidth=0.4, color=color_codex[i * seq_connect])

    if (len(landmarks) > 0):
        for x, y in landmarks:
            a = [states[x, 0], states[y, 0]]
            b = [states[x, 1], states[y, 1]]
            plt.plot(a, b, color="black", linewidth=1.3)
    if landmark_points is not None:
        print(np.shape(landmark_points))
        plt.plot(landmark_points[:, 0], landmark_points[:, 1], linestyle='None', color=[0, 0, 0], marker='*',
                 markersize=3)
    if name is not None:
        plt.title(name)
    if axis_range is not None:
        plt.xlim(axis_range[0:2])
        plt.ylim(axis_range[2:4])
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    if plotting_walls is not None:
        plotting_walls(name)
    if add_colorbar:
        plt.colorbar(label='Reward')
    if (filename != None):
        plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.pause(0.0001)
    plt.close('all')


def plot_representation_pca(states, color_codex, name="Learned State Representation", add_colorbar=False,
                            filename=None):
    # plt.ion()
    plt.figure(name)
    plt.hold(False)
    pca = PCA(n_components=2)
    mean = np.mean(states, axis=0)
    std_d = np.std(states, axis=0)
    scaled_states = (states - mean) / std_d
    # scaled_states = preprocessing.scale(states)
    pca.fit(scaled_states)
    principle_states = pca.transform(scaled_states)
    plt.scatter(principle_states[:, 0], principle_states[:, 1], s=4, c=color_codex, cmap='rgb', linewidths=0.1,
                alpha=0.4)
    plt.xlabel('Principle dimension 1')
    plt.ylabel('Principle dimension 2')
    if add_colorbar:
        plt.colorbar(label='Reward')
    if (filename != None):
        plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.pause(0.0001)


def plot_representation_matrix(states, color_codex_string, to_file=True, filename=r"plots/scatter_matrix.png"):
    if (np.shape(states)[0] >= 1500):
        states = states[-1500:]
        color_codex_string = color_codex_string[-1500:]
    data = pd.DataFrame(states)
    plot = scatter_matrix(data, alpha=0.6, figsize=(8, 8), diagonal='kde', c=color_codex_string, cmap='rgb')
    if (to_file):
        # plot[0].get_figure().savefig("plots/scatter_matrix.png")
        plt.savefig(filename, dpi=600, bbox_inches="tight")



def get_sequential_coloring(observations, flatten_inp=False):
    if (flatten_inp):
        print("Flatten Input is not available for sequential coloring")
        return
    num_episode = np.shape(observations)[0]
    episode_size = np.shape(observations)[1]
    diverse_colors = plt.cm.get_cmap('hsv', num_episode)
    color_codex = np.asarray(
        [np.ones((episode_size, 3)) * np.asarray(diverse_colors(i))[np.newaxis, :3] for i in range(num_episode)])
    # print("color shape : " , np.shape(color_codex))
    # color_codex = np.swapaxes(color_codex,0,1)
    color_codex = np.reshape(color_codex, (num_episode * episode_size, 3))
    color_codex_hex = 255 * color_codex
    color_codex_hex = ['#%02x%02x%02x' % tuple(map(int, col[:3])) for col in color_codex_hex]

    return color_codex, color_codex_hex


def get_positional_coloring(pose, flatten_inp=False, maze=1):
    Color_map = coloring.Coloring(maze=maze)
    if (not flatten_inp):
        pose = np.reshape(pose, (pose.shape[0] * pose.shape[1], 3))
    color_codex = Color_map.get_color(pose[:, :2])
    # color_codex = Color_map.get_grid_color(np.reshape(pose,(pose.shape[0]*pose.shape[1],3))[:,:2])
    color_codex_hex = 255 * color_codex
    color_codex_hex = ['#%02x%02x%02x' % tuple(map(int, col[:3])) for col in color_codex_hex]
    return color_codex, color_codex_hex


def get_orientational_coloring(pose, flatten_inp=False, maze=1):
    Color_map = coloring.Coloring(maze=maze)
    if (not flatten_inp):
        pose = np.reshape(pose, (pose.shape[0] * pose.shape[1], 3))
    color_codex = Color_map.get_orientational_color(pose[:, 2])
    # color_codex = Color_map.get_grid_color(np.reshape(pose,(pose.shape[0]*pose.shape[1],3))[:,:2])
    color_codex_hex = 255 * color_codex
    color_codex_hex = ['#%02x%02x%02x' % tuple(map(int, col[:3])) for col in color_codex_hex]
    return color_codex, color_codex_hex

