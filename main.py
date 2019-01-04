# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:56:51 2018

@author: marco
"""

import matplotlib as mpl
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
mpl.rcParams.update(params)
mpl.use('Agg')

import numpy as np
import data_utils
# import Tensorflow
import temporal_net
import time
import plotting
import coloring
import validation_net
import tests
import matplotlib.pyplot as plt
import csv
from collections import Counter
import os
import argparse

TEMP_NET = True
TRAINING = False
EXCLUDE = 10
def main():
    
    
    parser = argparse.ArgumentParser(usage="Train a new network, or evaluate a trained network for state representation Learning with robotic priors.")
    parser.add_argument("-n", "--n", type=int, default=1000, help="Number of Episodes, default is 1000")
    parser.add_argument("-d", "--state_dim", type=int, default=5, help="Number of Dimensions in State-Space")
    parser.add_argument("-bs", "--batch_size", type=int, default=30, help="Number of Sequences in one Batch")
    parser.add_argument("-bn", "--batch_number", type=int, default=30, help="Number of Batches in one Epoch")
    parser.add_argument("-sl", "--seq_length", type=int, default=40, help="Sequence-length in each Batch")
    parser.add_argument("-sp", "--similar_points", type=int, default=10, help="Number of points taken for landmarks")
    parser.add_argument("-cs", "--cut_seq", type=int, default=99, help="Length of the Sequence to take into Account for validation")
    parser.add_argument("-epochs", "--epochs", type=int, default=12000, help="Upper limit of Epochs to train")
    parser.add_argument("-t", "--training", action="store_true", default=False,
                        help="If selected, a new network is trained")
    parser.add_argument("-e", "--evaluate", action="store_true", default=False,
                        help="If selected, a trained network is evaluated")
    parser.add_argument("-plot", "--plotting", action="store_true", default=False,
                        help="If selected, plots are created")
    parser.add_argument("-m", "--model_file", default="prop_loss", help="Model to validate")
    parser.add_argument("--maze", default="nav01", help="Maze data to load")
    parser.add_argument("-l", "--use_landmark", action="store_true", default=False,
                        help="If selected, the new positional landmark is used")
    parser.add_argument( "--supervised", action="store_true", default=False,
                        help="If selected, the training will be supervised")
    parser.add_argument("--mixture", action="store_true", default=False,
                        help="If selected, we combine both the short and normal dataset" )                        
    parser.add_argument("--repeat", action="store_true", default=False,
                        help="If selected, we will also use the repeatability loss" )                        
    parser.add_argument('--gpu', type=str, default='',
                  help='Which Gpu is used, either 0 or 0,1 or 1')
    parser.add_argument("--no_lstm", action="store_false", default=True,
                        help="If selected, we are not using an lstm")
    parser.add_argument("-hn", "--hidden_nodes", type=int, default=512,
                        help="The number of hidden and LSTM nodes")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="The dropout probability during training")
    parser.add_argument("--no_skip", action="store_false", default=True, 
                         help=" If selected, we are not using skip-connections in the LSTM" )
    parser.add_argument("--dropout_conv", action="store_false", default=True,
                        help="If selected, we have the dropout behind the fully connected layer, otherwise behind the conv layer "  )
    parser.add_argument("--landmark_error", type=int, default=0, help="Error percentage when selecting similar points (around a landmark)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    landmark_float = args.landmark_error / 100.0
    if not os.path.exists("plots/"+args.model_file):
        os.makedirs("plots/"+args.model_file)
    #Training a new network
    if(args.training):
        if not os.path.exists("plots/"+args.model_file+"/training"):
            os.makedirs("plots/"+args.model_file+"/training")
        train(args.maze, args.state_dim, args.n, args.epochs, args.batch_number, args.batch_size, args.seq_length, args.plotting, args.model_file, args.similar_points, args.use_landmark, args.supervised, args.mixture, args.no_lstm, args.hidden_nodes, landmark_float)
    #Evaluate a network
    if(args.evaluate):
        load_and_validate(args.maze, args.state_dim, args.model_file, args.n, args.cut_seq, args.no_lstm, args.hidden_nodes, args.mixture, args.no_skip, args.plotting, args.supervised)


def train(task, state_dim, num_episodes, num_epochs, num_batches, batch_size, seq_len, plot, model_file,
          similar_points, new_landmark, supervised, mixture, use_lstm=True, hidden_nodes=128, landmark_error=0.0):
    state_step_sizes, data = load_data(task, mixture, num_episodes)
    print("Step size : ", state_step_sizes)

    time_before_init = time.time()
    dim_img = np.shape(data['observations'])[2]
    model_file = model_file + "/training"

    # data['actions'] = data['actions'] / state_step_sizes Now normalizing everything in the learn function
    srl = temporal_net.temporal_net(dim_img, state_dim, learning_rate=0.0001, l1_reg=0.01, num_epochs=num_epochs,
                                    plot=plot, model_file=model_file, similar_points=similar_points,
                                    maze=int(task[4]), use_landmark=new_landmark, supervised=supervised,
                                    hidden_nodes=hidden_nodes, landmark_error=landmark_error)
    print("time for initializing: {:.4f}".format(time.time() - time_before_init))
    training_states = srl.learn(state_step_sizes, batch_size, num_batches, seq_len, **data)
    # Tensorflow.plot_representation_matrix(training_states, to_file=True, filename= " temporal_scatter_matrix_2")


def load_and_validate(task, state_dim, model, num_episodes, cut_sequence=99, use_lstm=True, hidden_nodes=128,
                      mixture=False, skip=True, plot=True, supervised=False):
    try:

        plotting.plot_loss(model, supervised=model.find("sup") >= 0, gradient=True)
    except IOError:
        print("could not find loss file ")
    # Load Training Data to train the validation Network:
    state_step_sizes_train, data_train = load_data(task, mixture, num_episodes)
    # Initialize Model
    maze = int(task[4])
    """
    if (task == "nav01short"):
        steps_per_e = 400
        meanfile = "saved_models/shortinfo.npz"
    else:
        steps_per_e = 100
        meanfile = "saved_models/longinfo.npz"
    if (model.count("mix")):
        meanfile = "saved_models/mixinfo.npz"
    elif (model.count("short")):
        meanfile = "saved_models/shortinfo.npz"
    elif (model.count("long")):
        meanfile = "saved_models/longinfo.npz"
    print("task : ", task, "maze: ", maze)
    # state_step_sizes_train, data_train = data_utils.load_data_tensorflow(filename=task + '_train', with_depth=True, num_episodes=num_episodes)
    # state_step_sizes_test, data_test = data_utils.load_data_tensorflow(filename=task + '_test', with_depth=True, num_episodes=num_episodes, steps_per_episode=steps_per_e)
    state_step_sizes_test, data_test = data_utils.load_data_tensorflow(filename=task + '_train', with_depth=True,
                                                                       num_episodes=num_episodes,
                                                                       steps_per_episode=steps_per_e)
    print("step size: ", state_step_sizes_test)
    dim_img = np.shape(data_test['observations'])[2]
    """
    dim_img = np.shape(data_train['observations'])[2]
    # data_test['actions'] = data_test['actions'] / state_step_sizes_test
    srl = temporal_net.temporal_net(dim_img, state_dim, learning_rate=0.0001, l1_reg=0.01, num_epochs=1, maze=maze,
                                    hidden_nodes=hidden_nodes)

    model_path = "saved_models/" + model

    # Calculate the mean of the train data and store them in mean file
    meanfile = "plots/" + model + "/mean.npz"
    np.savez(meanfile, stepsize=state_step_sizes_train,
             mean_obs=np.mean(data_train['observations'], axis=(0, 1), keepdims=True),
             std_obs=np.std(data_train['observations'], ddof=1))

    # model_path = "saved_models/repeat_model1"
    # model_path = "saved_models/supervised"
    # model_path = "saved_models/prop_6_states"True
    # model_path = "saved_models/prop_loss"
    # model_path = "saved_models/10_landmarks"
    # model_path = "saved_models/no_landmarks"
    # model_path = "saved_models/model.ckpt"

    # validate_network(srl, data_train,model+"/Train",model_path)

    validate_network(srl, data_train, model + "/" + task, model_path, cut_sequence, maze, meanfile, plot, supervised)

    # temporal_net.plot_representation(flatten_and_normed(data_train['pose']), color_codex, name="Reconstructed Position", add_colorbar=False, filename="plots/Normed_Position.png")


def validate_network(srl, data, file_prefix, model_path, cut_sequence=99, maze=1, meanfile="saved_models/longinfo.npz",
                     plot=True, supervised=False):
    # Predict state space
    print_time = time.time()
    predicted_state = flatten_seq(
        srl.phi(data['observations'][:, :cut_sequence], data['actions'][:, :cut_sequence], model_path, exclude=EXCLUDE,
                meanfile=meanfile))

    print("to  predict the state takes ", time.time() - print_time, " seconds")
    print_time = time.time()
    # Reshape and clean the data

    mean_pose = np.mean(data['pose'], axis=(0, 1), keepdims=True)[0, :, :]
    std_pose = np.std(data['pose'], axis=(0, 1), keepdims=True)[0, :, :]

    without_first_pose = data['pose'][:,
                         EXCLUDE:cut_sequence]  # temporal_net.remove_first_10(data['pose'][:,:cut_sequence])
    flatten_pose = flatten_seq(without_first_pose)

    flatten_pose[:, :2] = (flatten_pose[:, :2] - mean_pose[:, :2]) / std_pose[:, :2]
    pose = np.concatenate((flatten_pose[:, :2], [[np.cos(x), np.sin(x)] for x in flatten_pose[:, 2]]), axis=1)
    not_normed_pose = flatten_seq(without_first_pose)

    if False:  # (np.shape(predicted_state)[1] == 4 and _plot ):
        load_and_plot_validation(srl, cut_sequence, model_path, meanfile, mean_pose, std_pose)
        return

    # Coloring information for plotting
    color_codex, color_codex_hex = temporal_net.get_positional_coloring(without_first_pose, flatten_inp=False,
                                                                        maze=maze)
    color_codex_orient, color_codex_orient_hex = temporal_net.get_orientational_coloring(without_first_pose,

                                                                                         flatten_inp=False, maze=maze)

    filename = "plots/" + file_prefix + "reconstructed_"

    inp_size = np.shape(pose)[0]
    inp_dim = np.shape(predicted_state)[1]
    out_dim = np.shape(pose)[1]

    train_border = int(inp_size * 7 / 10)
    mid = int(inp_size / 2)

    # Calculating Weights for different samples depending occurence in each field
    topo_coloring = coloring.Coloring(maze=maze)
    topo_indices = topo_coloring.get_topo_pose_index(not_normed_pose)

    pose_counter = Counter(topo_indices)
    weights = np.asarray([len(topo_indices) / pose_counter[x] for x in topo_indices[:train_border]])
    vali_weights = np.asarray([len(topo_indices) / pose_counter[x] for x in topo_indices[train_border:]])

    print("Norming and stuff takes ", time.time() - print_time, " seconds")
    print_time = time.time()
    if not supervised:
        # Validation net for pose
        vali_net = validation_net.validation_net(inp_dim, out_dim)
        vali_net.learn(predicted_state[:train_border], pose[:train_border], weights=weights,
                       validation_data=predicted_state[train_border:],
                       validaton_pose=pose[train_border:], validation_weight=vali_weights)

    else:
        vali_net = False

    if False:
        load_and_plot_validation(srl, cut_sequence, model_path, meanfile, mean_pose, std_pose, vali_net)
        return

    # Validation net for topology
    vali_net_topo = validation_net.validation_net(inp_dim, np.max(topo_indices) + 1, one_hot=True)
    vali_net_topo.learn(predicted_state[:train_border], topo_indices[:train_border], weights=weights,
                        validation_data=predicted_state[train_border:],
                        validaton_pose=topo_indices[train_border:], validation_weight=vali_weights)

    print("Train validation net takes ", time.time() - print_time, " seconds")
    # Now load different datasets and test their performance  with the validation nets
    if (maze == 1):

        # validate(vali_net,vali_net_topo, srl, model_path, "maze1id312cyclus1" , 400, meanfile, cut_sequence, mean_pose, std_pose, maze, plot)
        validate(vali_net, vali_net_topo, srl, model_path, "nav01", 500, meanfile, cut_sequence, mean_pose, std_pose,
                 maze, plot)
        validate(vali_net, vali_net_topo, srl, model_path, "nav01short", 500, meanfile, cut_sequence, mean_pose,
                 std_pose, maze, plot)
        validate(vali_net, vali_net_topo, srl, model_path, "nav01mix", 500, meanfile, cut_sequence, mean_pose,
                 std_pose, maze, plot)
        # validate(vali_net,vali_net_topo, srl, model_path, "nav01rlnew" , 500, meanfile, cut_sequence, mean_pose, std_pose, maze, plot)
        # validate(vali_net,vali_net_topo, srl, model_path, "nav01rlpose" , 500, meanfile, cut_sequence, mean_pose, std_pose, maze, plot)
    else:
        validate(vali_net, vali_net_topo, srl, model_path, "nav0" + str(maze), 500, meanfile, cut_sequence, mean_pose,
                 std_pose,
                 maze, plot)
        validate(vali_net, vali_net_topo, srl, model_path, "nav0" + str(maze) + "short", 500, meanfile, cut_sequence,
                 mean_pose,
                 std_pose,
                 maze, plot)
        validate(vali_net, vali_net_topo, srl, model_path, "nav0" + str(maze) + "mix", 500, meanfile, cut_sequence,
                 mean_pose,
                 std_pose,
                 maze, plot)


def validate(vali_net, vali_net_topo, srl, model_path, data_set, num_episodes, meanfile, cut_sequence, mean_pose,
             std_pose, maze=1, plot=True):
    if (data_set.find("mix") >= 0):
        print(data_set, data_set.find("mix"))
        _, data_test = load_data(data_set.replace("mix", ""), True, 500, dataset_type="_test")
    else:
        _, data_test = load_data(data_set, False, 500, dataset_type="_test")
    predicted_state = flatten_seq(
        srl.phi(data_test['observations'][:, :cut_sequence], data_test['actions'][:, :cut_sequence], model_path,
                exclude=EXCLUDE,
                meanfile=meanfile))

    file_prefix = model_path.replace("saved_models/", "") + "/" + data_set
    if not os.path.exists("plots/" + file_prefix):
        os.makedirs("plots/" + file_prefix)
    filename = "plots/" + file_prefix + "/reconstructed_"

    # Reshape and clean the data
    without_first_pose = data_test['pose'][:,
                         EXCLUDE:cut_sequence]  # temporal_net.remove_first_10(data['pose'][:,:cut_sequence])
    seq_len = np.shape(without_first_pose)[1]
    flatten_pose = flatten_seq(without_first_pose)

    # flatten_pose[:, :2] = (flatten_pose[:, :2] - np.mean(flatten_pose, axis=(0, 1), keepdims=True)[:, :2]) / np.std(
    #    flatten_pose, axis=(0, 1), keepdims=True)[:, :2]
    flatten_pose[:, :2] = (flatten_pose[:, :2] - mean_pose[:, :2]) / std_pose[:, :2]
    pose = np.concatenate((flatten_pose[:, :2], [[np.cos(x), np.sin(x)] for x in flatten_pose[:, 2]]), axis=1)

    not_normed_pose = flatten_seq(without_first_pose)

    # Coloring information for plotting
    color_codex, color_codex_hex = temporal_net.get_positional_coloring(without_first_pose, flatten_inp=False,
                                                                        maze=maze)
    color_codex_orient, color_codex_orient_hex = temporal_net.get_orientational_coloring(without_first_pose,
                                                                                         flatten_inp=False, maze=maze)

    topo_coloring = coloring.Coloring(maze=maze)
    topo_indices = topo_coloring.get_topo_pose_index(not_normed_pose)

    # First Calculate the KNN-MSE Score:
    knn_score = vali_net_topo.knn_mse(predicted_state, pose)

    if vali_net:
        score_pose, std_error_pose = vali_net.calculate_score(predicted_state, pose)
        trained_states = vali_net.predict(predicted_state)
    else:
        diff = np.linalg.norm(predicted_state - pose, axis=1) ** 2
        score_pose = np.average(diff)
        std_error_pose = np.std(diff) / len(diff) ** 0.5
        trained_states = predicted_state
    print("The averaged distance to the ground truth pose after testing with the training_set is: \n ", score_pose)
    maze_plotter = lambda fig_name: plotting.plot_maze(maze="nav0" + str(maze), figure_name=fig_name,
                                                       means=mean_pose[0, :], stds=std_pose[0, :])
    if (plot):
        temporal_net.plot_representation(trained_states[:, :2], color_codex,
                                         name=None,  # "Reconstructed Position",
                                         add_colorbar=False,
                                         filename=filename + "Position.png",
                                         axis_labels=["X-Coordinate", "Y-Coordinate"], plotting_walls=maze_plotter)
        temporal_net.plot_representation(trained_states[:, 2:], color_codex_orient,
                                         name=None,  # "Reconstructed Orientation",
                                         add_colorbar=False,
                                         filename=filename + "Orientation.png",
                                         axis_labels=['Cos(theta)', 'Sin(theta)'])

    score_topo = vali_net_topo.calculate_score(predicted_state, topo_indices)
    print("The percentage of correct classified topological position is: \n ", score_topo)
    predicted_topo_index = vali_net_topo.predict(predicted_state)
    topo_distances = topo_coloring.get_topo_distance(predicted_topo_index, topo_indices)

    # Plot a topological Heatmap
    if (plot):
        plot_topo_heatmap(topo_indices, predicted_topo_index, topo_coloring,
                          filename=filename + "TopologicalHeatmap.png", mean=mean_pose, std=std_pose)

    print(
        "The average topological distance of the predicted region to the real region is: \n",
        np.average(topo_distances))
    if (plot):
        temporal_net.plot_representation(
            topo_coloring.get_pose_by_topo(predicted_topo_index, mean_pose[0, :], std_pose[0, :])[:, :2],
            color_codex, name="Reconstructed topological position",
            filename=filename + "Topology.png", axis_labels=["X-Index", "Y-Index"], plotting_walls=maze_plotter)

        # Plot predicted states
        # plot_representation_sequence_coloring(states,np.shape(without_first_pose)[1],file_prefix)
        plot_representation(predicted_state, not_normed_pose, file_prefix, maze=maze)
        plot_representation_sequence_coloring(predicted_state[:50 * seq_len], seq_len, file_prefix)
    print("Score on " + file_prefix + " data Position: ", score_pose, " and Topology: ", score_topo,
          "with average distance of :", np.average(topo_distances))

    with open("plots/results.csv", 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow([file_prefix, score_pose, score_topo, np.mean(topo_distances), knn_score, std_error_pose,
                         np.std(topo_distances) / len(topo_distances) ** 0.5])


def load_and_plot_validation(srl, cut_sequence, model_path, meanfile, mean_pose, std_pose, vali_net=None):
    _, data_plot = data_utils.load_data_tensorflow(filename="nav01_plotting_test", num_episodes=2)

    predicted_state_plot = flatten_seq(
        srl.phi(data_plot['observations'][:, :cut_sequence], data_plot['actions'][:, :cut_sequence], model_path,
                exclude=EXCLUDE,
                meanfile=meanfile))
    if vali_net is not None:
        recon_pose = vali_net.predict(predicted_state_plot)
    else:
        recon_pose = None
    plotting.plot_maze(means=mean_pose[0, :], stds=std_pose[0, :])
    plot_validation_traj(predicted_state_plot, data_plot['pose'], cut_sequence, vali_state=recon_pose, num_plots=2,
                         mean=mean_pose, std=std_pose)
    print("plotted validation trajectories?")
    return


def plot_validation_traj(predicted_state, pose, cut_sequence, vali_state=None, num_plots=10, plot_maze=None,
                         mean=[477.20581, 265.0655, 0], std=[292.33704, 144.69109, 1], name=""):
    if (len(np.shape(mean)) > 1):
        compare_pose = flatten_seq(pose[:, EXCLUDE:cut_sequence])

        compare_pose[:, :2] = (compare_pose[:, :2] - mean[:, :2]) / std[:, :2]
    else:
        compare_pose = (flatten_seq(pose[:, EXCLUDE:cut_sequence]) - mean) / std
    compare_pose = np.asarray([[p[0], p[1], np.cos(p[2]), np.sin(p[2])] for p in compare_pose])
    if vali_state is None:
        print("mean squared error ", np.mean((predicted_state - compare_pose) ** 2))
    else:
        print("mean squared error ", np.mean((vali_state - compare_pose) ** 2))
    # print(data_plot['pose'][0])
    # print(predicted_state_plot[:100])

    for i in range(num_plots):
        di = cut_sequence - EXCLUDE
        plt.plot(compare_pose[i * (di):(i + 1) * di, 0], compare_pose[i * (di):(i + 1) * di, 1],
                 label="Ground truth position", color="green")
        # plt.plot(compare_pose[i * (di), 0], compare_pose[i * (di), 1], 'o', color="green")
        # plt.plot(compare_pose[(i + 1) * (di) - 1, 0], compare_pose[(i + 1) * (di) - 1, 1], 'o', color="red")

        if vali_state is not None:
            plt.plot(vali_state[i * (di):(i + 1) * di, 0], vali_state[i * (di):(i + 1) * di, 1],
                     label="Reconstructed Position", color="red")
            # plt.plot(vali_state[i * (di), 0], vali_state[i * (di), 1], 'o', color="green")
            # plt.plot(vali_state[(i + 1) * (di) - 1, 0], vali_state[(i + 1) * (di) - 1, 1], 'o', color="red")
        else:
            plt.plot(predicted_state[i * (di):(i + 1) * di, 0], predicted_state[i * (di):(i + 1) * di, 1],
                     label="Supervised learned state", color="red")
            # plt.plot(predicted_state[i * (di), 0], predicted_state[i * (di), 1], 'o', color="green")
            # plt.plot(predicted_state[(i + 1) * (di) - 1, 0], predicted_state[(i + 1) * (di) - 1, 1], 'o', color="red")

        if plot_maze is not None:
            plot_maze()
        if (num_plots > 2):
            plt.legend(loc='best')
            # plt.show()
            # plt.pause(2)
            plt.waitforbuttonpress()
            plt.close('all')
    if (num_plots <= 2):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(),loc='upper left',bbox_to_anchor=(1.04,1), borderaxespad=0)
        plt.legend(by_label.values(), by_label.keys(), loc='lower left', bbox_to_anchor=(.04, 1.04), borderaxespad=0)
        # plt.show()
        # plt.pause(2)
        plt.savefig("plots/sonstiges/Trajectories" + name + ".png", dpi=1200, bbox_inches="tight")


def plot_topo_heatmap(real_topo, calculated_topo, topo_coloring, filename, mean, std):
    points_total = np.zeros((topo_coloring.map_x, topo_coloring.map_y))
    points_correct = np.zeros((topo_coloring.map_x, topo_coloring.map_y))
    for i in range(len(real_topo)):
        x = int(real_topo[i] / topo_coloring.map_y)
        y = int(real_topo[i] % topo_coloring.map_y)
        # x = int(calculated_topo[i] / topo_coloring.map_y)
        # y = int(calculated_topo[i] % topo_coloring.map_y)
        points_total[x, y] += 1
        points_correct[x, y] += int(real_topo[i] == calculated_topo[i])

    grid = np.zeros((topo_coloring.map_x * 100, topo_coloring.map_y * 100))
    for x in range(topo_coloring.map_x):
        for y in range(topo_coloring.map_y):
            if (points_total[x, y] != 0):
                grid[x * 100:x * 100 + 100, y * 100:y * 100 + 100] = int(
                    100 * points_correct[x, y] / float(points_total[x, y]))

    plt.imshow(grid.T, cmap='autumn', vmin=0, vmax=100)  # , interpolation='nearest')
    plt.gca().invert_yaxis()
    ax = plt.gca()
    plt.colorbar(orientation='horizontal')
    if (topo_coloring.map_x == 10):
        plotting.plot_maze()
    elif (topo_coloring.map_x == 15):
        plotting.plot_maze(maze="nav02")
    elif (topo_coloring.map_x == 20):
        plotting.plot_maze(maze="nav03")

    try:
        folder = filename[:filename.rfind("/")] + "/../training/landmarks.txt"
        landmarks = np.loadtxt(folder, delimiter=",", dtype=str)
        landmarks = np.vectorize(float)(landmarks)
        landmarks = landmarks * std + mean
        plt.scatter(landmarks[:, 0], landmarks[:, 1], color=(0, 0, 0), marker="*")
    except:
        print("Could not finde a landmark file")
    for x in range(topo_coloring.map_x):
        for y in range(topo_coloring.map_y):
            plt.text(x * 100 + 20, y * 100 + 50, "{:d}%".format(
                int(grid[x * 100 + 50, y * 100 + 50])), fontsize=6)

    plt.savefig(filename, dpi=1200, bbox_inches="tight")
    plt.close('all')


def load_data(task, mixture, num_episodes, dataset_type="_train"):
    # Load and eventually mix Data from different data sets
    if (task == "nav01short" or (task.find("rl") >= 0 and not mixture)):
        steps_per_e = 400
    else:
        steps_per_e = 100
    time_before_loading = time.time()
    state_step_sizes, data = data_utils.load_data_tensorflow(filename=task + dataset_type, with_depth=True,
                                                             num_episodes=num_episodes, steps_per_episode=steps_per_e)
    if (mixture and task.find("rlnew") >= 0):

        state_step_sizes_short, data_short = data_utils.load_data_tensorflow(filename="nav01" + 'short' + dataset_type,
                                                                             with_depth=True, num_episodes=2000,
                                                                             steps_per_episode=100)
        state_step_sizes_long, data_long = data_utils.load_data_tensorflow(filename="nav01" + dataset_type,
                                                                           with_depth=True,
                                                                           num_episodes=1000, steps_per_episode=100)
        # state_step_sizes_late, data_late = data_utils.load_data_tensorflow(filename="nav01rllate" + '_train', with_depth=True, num_episodes=2000, steps_per_episode=100)
        for key in data_short.keys():
            print(np.shape(data[key]), "before")
            # data[key] = np.concatenate((data[key],data_short[key],data_long[key],data_late[key]),axis=0)
            data[key] = np.concatenate((data[key], data_short[key], data_long[key]), axis=0)
            print(np.shape(data[key]), "after")
        # state_step_sizes = (state_step_sizes + state_step_sizes_short + state_step_sizes_long + state_step_sizes_late) / 4.0
        state_step_sizes = (state_step_sizes + state_step_sizes_short + state_step_sizes_long) / 4.0
    elif (mixture):
        state_step_sizes_short, data_short = data_utils.load_data_tensorflow(filename=task + 'short' + dataset_type,
                                                                             with_depth=True, num_episodes=num_episodes,
                                                                             steps_per_episode=100)
        for key in data_short.keys():
            print(np.shape(data[key]), "before")
            data[key] = np.concatenate((data[key], data_short[key]), axis=0)
            print(np.shape(data[key]), "after")
        state_step_sizes = (state_step_sizes + state_step_sizes_short) / 2.0
    print("time for loading dataset: {:.4f}".format(time.time() - time_before_loading))

    return state_step_sizes, data


def plot_representation_sequence_coloring(train_states, seq_len, file_prefix):
    episodes = int(np.shape(train_states)[0] / seq_len)
    dim = np.shape(train_states)[1]
    color_codex, color_codex_hex = temporal_net.get_sequential_coloring(
        np.reshape(train_states, (episodes, seq_len, dim)))
    temporal_net.plot_representation(train_states[:, :2], color_codex, name="Statespace Sequential Coloring",
                                     add_colorbar=False,
                                     filename="plots/" + file_prefix + "/SequentialColoringFirstTwoDimensionsStateSpace.png",
                                     axis_labels=['State dimension 1', 'State dimension 2'])
    temporal_net.plot_representation(train_states[:, 2:], color_codex, name="Statespace Sequential Coloring",
                                     add_colorbar=False,
                                     filename="plots/" + file_prefix + "/SequentialColoringLastTwoDimensionsStateSpace.png",
                                     axis_labels=['State dimension 3', 'State dimension 4 '])
    temporal_net.plot_representation_matrix(train_states, color_codex_hex,
                                            filename="plots/" + file_prefix + "/SequentialScatterMatrix")


def plot_representation(train_states, flatten_pose, file_prefix, maze=1):
    color_codex, color_codex_hex = temporal_net.get_positional_coloring(flatten_pose, flatten_inp=True, maze=maze)
    ori_color_codex, ori_color_codex_hex = temporal_net.get_orientational_coloring(flatten_pose, flatten_inp=True)
    temporal_net.plot_representation(train_states[:, :2], color_codex,
                                     name="First Two Dimension State Space" + file_prefix, add_colorbar=False,
                                     filename="plots/" + file_prefix + "/FirstTwoDimensionsStateSpace.png",
                                     axis_labels=['State dimension 1', 'State dimension 2'])
    temporal_net.plot_representation(train_states[:, 2:], color_codex,
                                     name="Last Two Dimension State Space" + file_prefix, add_colorbar=False,
                                     filename="plots/" + file_prefix + "/LastTwoDimensionsStateSpace.png",
                                     axis_labels=['State dimension 3', 'State dimension 4'])
    temporal_net.plot_representation(train_states[:, :2], ori_color_codex,
                                     name="First Two Dimension State Space" + file_prefix, add_colorbar=False,
                                     filename="plots/" + file_prefix + "/OrientFirstTwoDimensionsStateSpace.png",
                                     axis_labels=['State dimension 1', 'State dimension 2'])
    temporal_net.plot_representation(train_states[:, 2:], ori_color_codex,
                                     name="Last Two Dimension State Space" + file_prefix, add_colorbar=False,
                                     filename="plots/" + file_prefix + "/OrientLastTwoDimensionsStateSpace.png",
                                     axis_labels=['State dimension 3', 'State dimension 4'])

    temporal_net.plot_representation_pca(train_states, color_codex, name="PCA State Space" + file_prefix,
                                         filename="plots/" + file_prefix + "/PCAStateSpace.png")
    print("Now plotting representation Matrix")
    temporal_net.plot_representation_matrix(train_states, color_codex_hex,
                                            filename="plots/" + file_prefix + "/PositionalScatterMatrix")
    temporal_net.plot_representation_matrix(train_states, ori_color_codex_hex,
                                            filename="plots/" + file_prefix + "/OrientationalScatterMatrix")
    print("Done with plotting representation Matrix")


def flatten_seq(data):
    return np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))


def flatten_and_normed(data):
    data = (data - np.mean(data, axis=(0, 1), keepdims=True)) / np.std(data, axis=(0, 1), keepdims=True)
    return np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))


if __name__ == '__main__':
    main()
    
