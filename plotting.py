import matplotlib as mpl
"""pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                    # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
}"""
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
mpl.rcParams.update(params)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import numpy as np
import coloring
import data_utils
from matplotlib.animation import FuncAnimation
import pandas as pd
import temporal_net
import main
import validation_net
import tensorflow as tf
from collections import Counter
import os
import seaborn as sns
import q_net
import pickle
from tensorflow.python.tools import inspect_checkpoint as chkp

dir_path = os.path.dirname(os.path.realpath(__file__))
FILE_PATH = dir_path + "/../"

def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def test_consitency():
    state_step_sizes, data = data_utils.load_data_tensorflow(filename='nav01_train', with_depth=True, num_episodes=1000)
    episode = 0
    for e in data['pose']:
        episode += 1
        for i in range(98):
            if dist(e[i], e[i + 1]) >= 350:
                print(episode, "Distance was ", dist(e[i], e[i + 1]), "between ", e[i], " and ", e[i + 1])


def test_start_poses():
    state_step_sizes, data = data_utils.load_data_tensorflow(filename='nav01short_train', with_depth=True,
                                                             num_episodes=400, steps_per_episode=400)
    starts = {}
    for e in data['pose']:

        x = round(e[0, 0], 4)
        # print(x)
        y = round(e[0, 1], 4) * 0.00001
        if ((x + y) not in starts.keys()):

            starts[(x + y)] = 1
        else:
            print("this start existed", (x + y))
            starts[(x + y)] += 1
    print("differentStarts : ", len(starts.keys()))


def plot_q_values(iids=1):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # sess.run(init)
        net = q_net.q_net(4, 6, hidden_nodes=256, middle_layer=True, relu_layer=False)
        for iid in iids:
            # net = q_net.q_net(4,6,hidden_nodes=512,middle_layer=True)
            net.load_model(sess, FILE_PATH + "agents/Q_net7000id" + str(iid))
            x = np.linspace(-1, 1, 250)
            y = np.linspace(-1, 1, 125)
            o = np.linspace(-1 * np.pi, np.pi, 10)
            X, Y, O = np.meshgrid(x, y, o)
            X = X.flatten()
            Y = Y.flatten()
            O = O.flatten()
            coordinates = np.asarray(list(zip(X, Y, O)))
            print(np.shape(coordinates))
            coor_ext = np.concatenate((coordinates[:, :2], [[np.cos(i), np.sin(i)] for i in coordinates[:, 2]]), axis=1)
            q_value = np.max(net.predict(coor_ext, sess), axis=1)
            # q_value = np.sum(coordinates,axis=1)
            print("All rotations Min q {} Max q {} ".format(np.min(q_value), np.max(q_value)))
            grid = np.max(np.reshape(q_value, (125, 250, 10)), axis=2)
            print("Min q {} Max q {} ".format(np.min(grid), np.max(grid)))
            # plt.imshow(grid, cmap='hot')#, interpolation='nearest')
            # plt.gca().invert_yaxis()
            # plt.colorbar()
            # plt.show()
            plt.imshow(grid, cmap='hot')  # , interpolation='nearest')
            plt.gca().invert_yaxis()
            plt.colorbar()
            plot_maze(means=np.asarray([0., 0.]), stds=np.asarray([4., 4.]))
            plt.savefig(FILE_PATH + "agents/heatmap_q_values{}.png".format(iid), bbox_inches="tight")
            plt.close('all')


def plot_performance(id=13):
    per = np.loadtxt(FILE_PATH + "agents/validatintimes5000id" + str(id))
    plt.plot(range(len(per)), per)
    print("average validation time: ", np.mean(per[-500:]))
    plt.show()
    # plt.waitforbuttonpress()


def plot_performance_errorbar(ids, num_episodes=5000, cycles=0, close=True, file_path_extension="", use_std=True,
                              average_window_start=1, label="", color=None):
    path = FILE_PATH + "agents/" + file_path_extension
    plot_path = "plots/" + file_path_extension
    val = []
    rand = []
    train = []
    cum_val = []
    cum_rand = []
    cum_train = []
    cum = True
    if (cycles):
        for i in ids:
            val.append([100 - x if x != 3000 else -x for j in range(cycles) for x in
                        np.loadtxt(path + "validatintimes" + str(num_episodes) + "id" + str(i) + "0" + str(j))])

            # val.append([100-x if x!=3000 else -x for x in np.loadtxt(path+"validatintimes7000id"+str(i)) ])
            try:
                rand.append([100 - x if x != 3000 else -x for j in range(cycles) for x in np.loadtxt(
                    path + "validatinrandomtimes" + str(num_episodes) + "id" + str(i) + "0" + str(j))])
                # rand.append([100-x if x!=3000 else -x for x in np.loadtxt(path+"validatinrandomtimes7000id"+str(i))])
            except:
                print("random file with id {} not there".format(i))
            train.append([100 - x if x != 3000 else -x for j in range(cycles) for x in
                          np.loadtxt(path + "trainingperformance" + str(num_episodes) + "id" + str(i) + "0" + str(j))])
            # train.append([100-x if x!=3000 else -x for x in np.loadtxt(path+"trainingperformance7000id"+str(i))])

        val = np.reshape(val, (len(ids), cycles, -1))
        rand = np.reshape(rand, (len(ids), cycles, -1))
        train = np.reshape(train, (len(ids), cycles, -1))
        cum = False
    else:
        for i in ids:
            val.append([100 - x if x != 3000 else -x for x in
                        np.loadtxt(path + "validatintimes" + str(num_episodes) + "id" + str(i))])
            try:
                rand.append([100 - x if x != 3000 else -x for x in
                             np.loadtxt(path + "validatinrandomtimes" + str(num_episodes) + "id" + str(i))])
            except:
                print("random file with id {} not there".format(i))
            try:
                cum_rand.append(np.loadtxt(path + "CummultativeRewardRandVal" + str(num_episodes) + "id" + str(i)))
                cum_val.append(np.loadtxt(path + "CummultativeRewardVal" + str(num_episodes) + "id" + str(i)))
                cum_train.append(np.loadtxt(path + "CummultativeRewardTrain" + str(num_episodes) + "id" + str(i)))

            except:
                cum = False
                print("No Cummultative File with id {}".format((i)))
            train.append([100 - x if x != 3000 else -x for x in
                          np.loadtxt(path + "trainingperformance" + str(num_episodes) + "id" + str(i))])
        val = np.asarray(val)
        rand = np.asarray(rand)
        train = np.asarray(train)
        cum_rand = np.asarray(cum_rand)
        cum_val = np.asarray(cum_val)
        cum_train = np.asarray(cum_train)
    print(np.shape(val), np.shape(rand), np.shape(train))

    names = ["Fixed Validation", "Training", "Random Validation", "CummultRandom", "CummultFixed", "CumultTrain"]
    data = [val, train, rand, cum_rand, cum_val, cum_train]
    for i, values in enumerate(data):
        if (average_window_start > 1 and (i == 1 or i == 5)):
            average_window = average_window_start * 4
        else:
            average_window = average_window_start
        if (np.shape(values)[0] >= 1):
            name = names[i]

            plt.figure(name)

            if (cycles):
                x = np.shape(values)[1]  # - average_window +1
                x1, x2, y1, y2 = plt.axis()

                plt.axis((0, max(x - 1, x2), -3000, 100))
                # plt.axis((0, 5000, -3000, 100))
                after_training = int(0.8 * np.shape(values)[2])
                if (use_std):
                    std = np.std(np.mean(values[:, :, -after_training:], axis=2), axis=0) / (
                                (np.shape(values)[0]) ** 0.5)
                else:
                    std = np.zeros(np.shape(values)[1])
                if (average_window > 1):

                    mean = np.convolve(np.mean(values[:, :, -after_training:], axis=(0, 2)),
                                       np.ones((average_window,)) / average_window, mode='valid')
                    std = std[average_window / 2 - 1:-average_window / 2]
                else:
                    mean = np.mean(values[:, :, -after_training:], axis=(0, 2))
                # plt.fill_between(np.linspace(0, np.shape(values)[1], x), mean - std, mean + std, alpha=0.4)
                plt.fill_between(list(range(x)), mean - std, mean + std, alpha=0.4, color=color)
                # plt.errorbar(np.linspace(0,np.shape(values)[1],x), mean, yerr=std, label= label)
                plt.plot(list(range(x)), mean, label=label, color=color)
            else:
                x = np.shape(values)[1] + 1  # - average_window +1
                x1, x2, y1, y2 = plt.axis()
                if (x2 < 2):
                    x2 = 10000
                # plt.axis((0, max(int(num_episodes), x2), -3000, 100))
                plt.axis((0, min(int(num_episodes), x2), -3000, 100))
                # plt.axis((0, 5000, -3000, 100))
                std = np.std(values, axis=0) / (len(ids)) ** 0.5 if use_std else np.zeros(np.shape(values)[1:])
                if (average_window > 1):

                    mean = np.mean(values, axis=0)
                    # mean = np.median(values, axis=0)

                    # mean = np.concatenate((np.ones((average_window))* mean[0],mean))
                    mean = np.concatenate((np.ones((average_window)) * -3000., mean))
                    mean = np.convolve(mean, (np.ones((average_window,)) / average_window), mode='valid')
                    # mean = np.convolve(np.mean(values, axis=0), (np.ones((average_window,)) / average_window), mode='valid')
                    # std =  std[average_window/2-1:-average_window/2 ]
                    std = np.concatenate((np.ones((average_window)) * std[0], std))
                    std = np.convolve(std, (np.ones((average_window,)) / average_window), mode='valid')
                else:
                    mean = np.mean(values, axis=0)
                    # mean = np.median(values, axis=0)
                # plt.errorbar(np.linspace(0,int(num_episodes),x), mean, yerr=std, label=label)
                plt.plot(np.linspace(0, int(num_episodes), x), mean, label=label, color=color)
                print(np.shape(mean))

                plt.fill_between(np.linspace(0, int(num_episodes), x), mean - std, mean + std, alpha=0.4, color=color)
                # plt.fill_between(range(num_episodes), mean - std, mean + std, alpha=0.4)
            if (close):
                plt.title("Cumultative Reward for " + name + " \n IDs: " + str(ids))
                plt.xlabel('Episodes')
                plt.ylabel('Cumultative Reward')
                plt.legend(loc='best')
                plt.savefig(plot_path + name + "ids" + "".join([str(i) for i in ids]) + ".png", bbox_inches="tight")

    if close:
        plt.close('all')
    return cum


def plot_grouped_errorbars(ids, num_episodes, cycles=0, names=None, file_extension="", fig_name="", file_name="",
                           colors=None):
    cum = True
    if names is not None:
        names = np.asarray(names)
        assert np.shape(names)[0] == np.shape(ids)[0]
    for i, id in enumerate(ids):
        if colors is not None:
            color = colors[i]
        else:
            color = None
        if (cycles):
            cum = plot_performance_errorbar(id, num_episodes=num_episodes[i], cycles=cycles, use_std=True, close=False,
                                            average_window_start=1, label=names[i], file_path_extension=file_extension,
                                            color=color) and cum
        else:
            cum = plot_performance_errorbar(id, num_episodes=num_episodes[i], cycles=cycles, use_std=True, close=False,
                                            average_window_start=20, label=names[i], file_path_extension=file_extension,
                                            color=color) and cum

    if cum:
        figs = ["Fixed Validation", "Training", "Random Validation", "CummultRandom", "CummultFixed", "CumultTrain"]
    else:
        figs = ["Fixed Validation", "Training", "Random Validation"]
    plot_path = "plots/" + file_extension
    for fig in figs:
        plt.figure(fig)
        plt.title("Cumultative Reward for " + fig + " \n with " + fig_name)
        if (cycles):
            plt.xlabel('Cycle')
        else:

            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xlabel('Episodes')
        plt.ylabel('Cumultative Reward')
        plt.legend(loc='lower left')
        plt.savefig(plot_path + file_name + fig + ".png", bbox_inches="tight")
    plt.close("all")


def see_saved_variables(filename):
    chkp.print_tensors_in_checkpoint_file(filename, tensor_name='', all_tensors=False)


def save_normalization_data():
    state_step_sizes_short, data_short = data_utils.load_data_tensorflow(filename='nav01short_train', with_depth=True,
                                                                         num_episodes=500, steps_per_episode=400)
    state_step_sizes_long, data_long = data_utils.load_data_tensorflow(filename='nav01_train', with_depth=True,
                                                                       num_episodes=1000, steps_per_episode=100)

    state_step_sizes_short2, data_short2 = data_utils.load_data_tensorflow(filename="nav01short_train", with_depth=True,
                                                                           num_episodes=1000, steps_per_episode=100)
    for key in data_short2.keys():
        data_short2[key] = np.concatenate((data_long[key], data_short2[key]), axis=0)
    state_step_sizes_short2 = (state_step_sizes_long + state_step_sizes_short2) / 2.0

    mean_obs_short = np.mean(data_short['observations'], axis=(0, 1), keepdims=True)
    std_obs_short = np.std(data_short['observations'], ddof=1)

    mean_obs_long = np.mean(data_long['observations'], axis=(0, 1), keepdims=True)
    std_obs_long = np.std(data_long['observations'], ddof=1)

    mean_obs_short2 = np.mean(data_short2['observations'], axis=(0, 1), keepdims=True)
    std_obs_short2 = np.std(data_short2['observations'], ddof=1)

    print("short stepsize", state_step_sizes_short, "mean obs ", mean_obs_short, "std ", std_obs_short)
    print("long stepsize", state_step_sizes_long, "mean obs ", mean_obs_long, "std ", std_obs_long)
    print("mix stepsize", state_step_sizes_short2, "mean obs ", mean_obs_short2, "std ", std_obs_short2)

    print("action means short ", np.mean(data_short['actions'], axis=(0, 1), keepdims=True))
    print("action means long ", np.mean(data_long['actions'], axis=(0, 1), keepdims=True))
    print("action means mix ", np.mean(data_short2['actions'], axis=(0, 1), keepdims=True))

    np.savez("saved_models/shortinfo", stepsize=state_step_sizes_short, mean_obs=mean_obs_short, std_obs=std_obs_short)
    np.savez("saved_models/longinfo", stepsize=state_step_sizes_long, mean_obs=mean_obs_long, std_obs=std_obs_long)
    np.savez("saved_models/mixinfo", stepsize=state_step_sizes_short2, mean_obs=mean_obs_short2, std_obs=std_obs_short2)


def show_npz_data():
    # with np.load('../simple_navigation_task_test.npz') as data:
    with np.load('../maze_data/nav01_test.npz') as data:
        # with np.load('../slot_car_task_test.npz') as data:
        print(data.keys())
        for key in data.keys():
            if (key != 'rgbd'):
                print(np.shape(data[key]))
                print(data[key][:5])
        print((data['pose'][0, :] + data['vel'][1, :]) - data['pose'][1, :])
        print((data['pose'][1, :] + data['vel'][2, :]) - data['pose'][2, :])
        print((data['pose'][2, :] + data['vel'][3, :]) - data['pose'][3, :])


def plot_observations(observations, name='Observation Samples'):
    plt.ion()
    plt.figure(name)
    m, n = 8, 10
    for i in range(m * n):
        plt.subplot(m, n, i + 1)
        plt.imshow(observations[i].reshape(32, 32, 3), interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    plt.pause(0.0001)


def show_img_data():
    task = 'nav01'
    data = data_utils.load_data_tensorflow(filename=task + '_test', with_depth=False, episode_shape=False)
    print(np.shape(data['observations']))
    plot_observations(data['observations'], name="Observation Samples (Subset of Training Data) -- Nav01 Task")


def show_replay_buffer():
    with open(FILE_PATH + "/agents/replaybuffer1000", 'rb') as fp:
        replay_buffer = pickle.load(fp)
    x = list(replay_buffer)  #
    print(np.shape(x))
    # x = list(itertools.islice(replay_buffer,0,10000))
    # print(x[:10])
    unfold = zip(*x)
    # print(np.shape(unfold[0]))
    # print(np.shape(unfold[1]))
    # print(np.shape(unfold[2]))
    # print(np.shape(unfold[3]))
    database = np.concatenate((np.asarray(unfold[0]), np.asarray(unfold[1])[:, np.newaxis],
                               np.asarray(unfold[2])[:, np.newaxis], np.asarray(unfold[3])), axis=1)
    # print("shape", np.shape(database))
    # print(database[:100])
    np.savetxt(FILE_PATH + "agent/replay_buffer.txt", database, fmt='%4.1f')
    print(np.where(database[:, 4] == -1))


def plot_comparion_loss(models, names):
    max_epoch = 0
    max_loss = 0
    for i, model in enumerate(models):
        losses = np.asarray(np.loadtxt("plots/" + model + "/training/losses.txt", delimiter=","))
        losses[:, 0:2] = losses[:, 0:2] / 35.
        """
        if(model.find("supervised") >=0):
            plt.plot(list(range(len(losses))), losses[:, 0], label=names[i])
        else:
            plt.plot(list(range(len(losses))), losses[:, 1], label=names[i])
        """
        plt.plot(list(range(len(losses))), losses[:, 3], label=names[i])

        max_epoch = max(max_epoch, len(losses))
        max_loss = max(max_loss, np.max(losses[:, 0]))
    plt.legend(names, loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel(("Loss"))
    plt.xlim([0, max_epoch])
    plt.ylim([0, max_loss + 1])
    plt.savefig("plots/comparedLossesSupUsePrior.png", bbox_inches="tight")
    plt.close('all')


def plot_loss(model_file, supervised=False, gradient=False, cycle=1):
    print("Modelpath provided: ", model_file)
    for i in range(cycle):
        if (cycle > 1):
            try:
                losses = np.asarray(np.loadtxt("plots/" + model_file + "/cycle{}losses.txt".format(i), delimiter=","))
                losses[:, 0:2] = losses[:, 0:2] / 35.
                if (gradient):
                    gradients = np.asarray(
                        np.loadtxt("plots/" + model_file + "/cycle{}gradients.txt".format(i), delimiter=","))
                save_name = "plots/" + model_file + "/cycle{}".format(i)
            except IOError:
                print("Cycle {} not found".format(i))
                continue
        else:
            losses = np.asarray(np.loadtxt("plots/" + model_file + "/training/losses.txt", delimiter=","))
            losses[:, 0:2] = losses[:, 0:2] / 35.

            print("Loaded the loss")
            if (gradient):
                gradients = np.asarray(np.loadtxt("plots/" + model_file + "/training/gradients.txt", delimiter=","))
                print("Loaded the gradient")
            save_name = "plots/" + model_file + "/"
        if (np.shape(losses)[1] == 4):
            plt.plot(list(range(len(losses))), losses[:, [0, 2, 3]])
        else:
            plt.plot(list(range(len(losses))), losses[:, [0, 1, 2]])
        # plt.legend(["supervised", "unsupervised", "supervised validation"], loc="upper right")
        # plt.legend(["supervised",  "supervised validation"], loc="upper right")
        # plt.plot(list(range(len(losses))), losses[:, [0,1, 2, 3]])
        plt.legend(["Training", "RL-loss", "Orig Validation"], loc="upper right")
        plt.savefig(save_name + "losses.png", bbox_inches="tight")
        plt.close('all')
        if (gradient):
            plt.plot(list(range(len(gradients))), gradients[:, :-1])
            plt.legend(["Reference Point", "Temporal Coherence", "Variation", "Proportionality"],
                       loc="upper right")  # ,"repeatability"])
            plt.xlabel("Epoch")
            plt.ylabel(("Weighted Gradient"))
            plt.xlim([0, len(gradients)])
            plt.ylim([0, 1200])

            plt.savefig(save_name + "gradients.png", bbox_inches="tight")
            plt.close('all')


def print_latex_table(file):
    data = pd.read_csv("plots/" + file, error_bad_lines=False)
    data = pd.concat([data.iloc[:, 0].str.split("/", 1, expand=True), data.iloc[:, 1:]], axis=1)

    scoremeasure = ["pose_distance", "topological_distance", "topological_score", "knn_score"]
    data = pd.concat([data.iloc[:, 0].str.split("_", 2, expand=True), data.iloc[:, 1:]], axis=1)
    data = data.iloc[:, 1:]
    col_names = ["trainmaze", "Model", "scoremaze"]
    col_names.extend(data.columns[3:])
    data.columns = col_names
    data.replace(["nl5", "nl16", "nl32", "nl64", "sp", "supervised"],
                 ["5 Landmarks", "16 Landmarks", "32 Landmarks", "64 Landmarks", "Similar Points", "Supervised"],
                 inplace=True)
    datagroup = data.groupby("Model")
    for x in datagroup:
        maze1 = x[1].loc[x[1]['trainmaze'] == "mix"]
        maze2 = x[1].loc[x[1]['trainmaze'] == "mix2"]
        maze3 = x[1].loc[x[1]['trainmaze'] == "mix3"]
        results = [x[0], maze1.iloc[0]["pose_distance"], maze1.iloc[0]["topological_score"],
                   maze1.iloc[0]["topological_distance"], maze1.iloc[0]["knn_score"],
                   maze2.iloc[0]["pose_distance"], maze2.iloc[0]["topological_score"],
                   maze2.iloc[0]["topological_distance"], maze2.iloc[0]["knn_score"],
                   maze3.iloc[0]["pose_distance"], maze3.iloc[0]["topological_score"],
                   maze3.iloc[0]["topological_distance"], maze3.iloc[0]["knn_score"]]
        to_string = lambda x: x if type(x) is str else "{0:.2f}".format(x)
        print("&".join(map(to_string, results)) + "\\ \n \hline ")

def plot_scores(Scores, prefixes):
    get_column = lambda y: [x[y] for x in Scores]
    #x_ax = list(range(0,len(Scores)*2,2))
    x_ax = list(range(len(Scores)))
    plt.figure( figsize=(14,4))
    plt.bar(x_ax,get_column(1), align="center")
    plt.title("Distance to the Ground Truth Pose")
    plt.xticks(x_ax,get_column(0) )
    plt.savefig("plots/"+prefixes + "Distances.png", bbox_inches="tight")
    plt.close('all')
    plt.figure("Percentage Topology Classification", figsize=(14,4))
    plt.bar(x_ax,get_column(2), align="center")
    plt.title("Percentage Topology Classification")
    plt.xticks(x_ax,get_column(0) )
    plt.savefig("plots/"+prefixes + "TopologyClassification.png", bbox_inches="tight")
    plt.close('all')
    plt.figure("Topology distance to the right field", figsize=(14,4))
    plt.bar(x_ax,get_column(3),align="center")
    plt.title("Topology distance to the right field")
    plt.xticks(x_ax,get_column(0) )
    plt.savefig("plots/"+prefixes + "TopologyDistance.png", bbox_inches="tight")
    plt.figure("Knn Score", figsize=(14,4))
    plt.bar(x_ax,get_column(4),align="center")
    plt.title("Knn Score")
    plt.xticks(x_ax,get_column(0) )
    plt.savefig("plots/"+prefixes + "KnnScore.png", bbox_inches="tight")


def data_and_plot():
    Scores = [["supervised", 0.15, 0.75, 0.44],
              ["prop_loss5", 0.51, 0.57, 0.63],                        
              ["repeat_model",0.77, 0.4, 1.07],
              ["newLandmark32", 0.91, 0.33, 1.7],
              ["newLandmark128", 1.61, 0.19, 5.3],
              ["selected_landmark", 1.58, 0.118, 4.8],
              ["no_landmark",2.9, 0.02, 11.9]]
    plot_scores(Scores, "overview")
    Scores_new_landmark = [["selected \n landmark", 1.58, 0.118, 4.8],
                           ["new \n repeat8", 1.48, 0.14, 3.59],
                           ["closest \n repeat8" ,1.03, 0.30, 2.12],
                           ["new \n Landmark32", 0.91, 0.33, 1.7],
                           ["new \n repeat32",0.72, 0.43, 1.19],
                           ["closest \n repeat32",0.75, 0.42, 1.24],
                           ["new \n repeat64",0.73, 0.43, 1.19],
                           ["new \n Landmark128", 1.61, 0.19, 5.3],
                           ["closest \n repeat128",0.74,0.44, 1.29]]
            
    plot_scores(Scores_new_landmark, "use_landmarks")
    
    Scores_vortrag = [["Supervised",0.07,0.74,0.41],
                      ["Closest \n Points", 0.48, 0.57, 0.60],
                      ["Landmarks", 0.80, 0.38, 1.5  ],
                      ["No Landmarks", 2.73, 0.02, 11]]
    plot_scores(Scores_vortrag, "Vortrag")

    Scores_nl = [#["nl8",2.19805,0.055662921348314603,8.09276404494382,3.0954924],
                 #["nl32",1.8670988,0.12444943820224719,6.030112359550562,2.9991896],
                 #["nl64",1.3281258,0.2792808988764045,3.0535280898876405,2.2578673],
                 #["nl128",1.0933311,0.33269662921348314,2.1696629213483147,1.9565344],
                 #["nl256", 1.1525284, 0.29152808988764045, 2.659730337078652, 2.0768564],
                 #["nl512", 1.0858271, 0.3161573033707865, 2.123370786516854, 1.9588162],
                 ["nl8_remaster", 0.7671874, 0.39730337078651684, 1.1812359550561797, 1.3057733],
                 ["nl32_remaster",1.30612,0.24471910112359552,3.3516404494382024,2.1951725],
                 ["nl64_remaster",0.8836622,0.3657303370786517,1.4734606741573033,1.5317937],
                 ["nl128_remasterd",0.7141833,0.45591011235955053,0.9669662921348314,1.2741692],
                 ["nl16_even",0.78177166,0.364876404494382,1.2262022471910112,1.3239335],
                 ["nl32_even",0.56787884,0.5266516853932585,0.6609438202247191,0.9695892],
                 ["nl50_even", 0.43335107, 0.6315280898876404, 0.45107865168539324, 0.7038748],
                 ["nl64_even", 0.4287467, 0.6510561797752809, 0.42979775280898874, 0.6929353],
                 ["similar points", 0.4721322, 0.5756853932584269, 0.535191011235955, 0.7771594]]

    plot_scores(Scores_nl, "NewLossScores")


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        print(np.shape(ax))
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in enumerate(axs):
            for ax2 in ax:

                _show_on_single_plot(ax2)
    else:
        _show_on_single_plot(axs)

def plot_factors(arg1,arg2,arg3,data, name = "", y_label = None, x_label = None, title = None, y_lim=None):
    #sns.factorplot(arg1,arg2,arg3, data=data, palette="muted", legend=True, kind="bar")
    ax = sns.catplot(arg1, arg2, arg3, data=data, palette="muted", legend=False, kind="bar")#, ci='sd')
    if y_label is not None:
        ax.set_ylabels(y_label)
    if(y_lim is not None):

        plt.ylim(y_lim)
    if x_label is not None:
        ax.set_xlabels(x_label)
    if title is None:
        plt.title(name)
    else:
        pass# plt.title(title)
    print("plotting" + name)
    #show_values_on_bars(ax.axes)
    plt.savefig("plots/Comparisons/"+name+".png",bbox_inches="tight", dpi= 1200)

def plot_knn_with_pos(data,name, measure_compare1="topological_distance", measure_compare2="knn_score", x_label =None, y_label=None):

    plt.close('all')
    #ax = sns.regplot(x="pose_distance", y="knn_score", data=data, hue='type', legend =True)
    #ax = sns.lmplot(x="pose_distance", y="knn_score", data=data, fit_reg=False, hue='scoremaze', legend=False)
    #sns.regplot(x="pose_distance", y="knn_score", data=data, scatter=False, ax=ax.axes[0, 0])
    ax = sns.lmplot(x=measure_compare1, y=measure_compare2, data=data, fit_reg=True, hue='scoremaze', legend=False)
    #sns.regplot(x="pose_distance", y="knn_score", data=data, scatter=False, ax=ax.axes[0, 0])
    if y_label is not None:
        ax.set_ylabels(y_label)
    if x_label is not None:
        ax.set_xlabels(x_label)
    plt.legend(loc='lower right')
    #plt.title("Correlation of the distance to ground truth  pose and knn score")
    plt.savefig("plots/Comparisons/"+name+"KnnScoreReg.png",bbox_inches="tight")


def load_scores(start_line, end_line = None, name = "", file = "results.csv", choose = None, errorbar=False):
    data = pd.read_csv("plots/" + file, error_bad_lines=False)
    if end_line is None:
        data = data.tail(start_line)
    else:
        data = data[start_line:end_line]
    #print(data.tail(5))
    #print(data.columns)
    #print(data.iloc[:,0])#.str.split("/",1,expand=True))
    #print(data.iloc[:,1:])
    data = pd.concat([data.iloc[:,0].str.split("/",1,expand=True),data.iloc[:,1:]], axis=1)


    scoremeasure = ["pose_distance", "topological_distance", "topological_score" ,"knn_score"]
    #scoremeasure_ylim = [[0,4],None,[0,1],[0,5]]
    scoremeasure_ylim = [None, None, None, None]
    #scoremeasure_nice = ["Positional Error", "Topological Distance", "Topological Score", "KNN Score"]
    scoremeasure_nice = ["Positional Error", "Tile Distance", "Tile Score", "KNN Score"]
    #transform dataframe for factor plots
    if(errorbar):
        data = pd.concat([data.iloc[:, 0].str.split("_", 3, expand=True), data.iloc[:, 1:]], axis=1)
        data = data.iloc[:, 1:]
        col_names = ["trainmaze", "Model", "number", "scoremaze"]
        col_names.extend(data.columns[4:])
        data.replace(["error{:02d}".format(i) for i in [0,5,10,15,20,30,40,50,60]],[i for i in [0,5,10,15,20,30,40,50,60]], inplace=True)
    else:
        data = pd.concat([data.iloc[:, 0].str.split("_", 2, expand=True), data.iloc[:, 1:]], axis=1)
        data = data.iloc[:,1:]
        col_names = ["trainmaze", "Model", "scoremaze"]
        col_names.extend(data.columns[3:])
    data.columns = col_names

    data = data.loc[data["scoremaze"].isin(["nav01mix",  "nav02mix", "nav03mix"])]
    #data = data.loc[data["scoremaze"].isin(["nav01","nav01short"])]#,"nav02", "nav02short","nav03", "nav03short"])]
    #data = data.loc[data["scoremaze"].isin(["nav01", "nav01short","nav02", "nav02short","nav03", "nav03short"])]
    data.replace(["nav01mix","nav02mix","nav03mix"],["Mixed Dataset","Mixed Dataset","Mixed Dataset"],inplace=True)
    data.replace(["nav01","nav01short"],["Long Dataset","Short Dataset"],inplace=True)
    data.replace(["nav02", "nav02short"], ["Long Dataset \n Medium Maze", "Short Dataset \n Medium Maze"], inplace=True)
    data.replace(["nav03", "nav03short"], ["Long Dataset \n Large Maze", "Short Dataset \n Large Maze"], inplace=True)

    """
    for i,sm in enumerate(scoremeasure):
        plot_knn_with_pos(data, "CompleteKnn" + sm, sm, x_label=scoremeasure_nice[i], y_label="KNN Score")
    return
    """
    if(choose is not None):
        data = data.loc[data['Model'].isin(choose)]
    data.replace(["nl5","nl16","nl32","nl64","sp","supervised", "lm64"], ["5 Landmarks", "16 Landmarks", "32 Landmarks", "64 Landmarks", "Similar Points", "Supervised","64 Landmarks"], inplace=True)
    data.replace(["sp_novar", "sp_notemp", "sp_nolandmark", "sp_noprop"],
                 ["Without Variation", "Without Temporal Coherence", "Without Reference Point", "Without Proportionality"],
                 inplace=True)
    """
    datagroup = data.groupby("scoremaze")
    for x in datagroup:
        for sm in scoremeasure:
            plot_factors("Model", sm, "trainmaze", data=x[1],name= name +"_"+ str(x[0])+"_measure_" + sm)

    datagroup = data.groupby("Model")
    for x in datagroup:
        for sm  in scoremeasure:
            plot_factors("scoremaze", sm, "trainmaze", data=x[1], name=name+"_" + str(x[0])+"_" + sm)
    """
    #plot_knn_with_pos(data, name )
    datagroup = data.groupby("trainmaze")
    for x in datagroup:
        title = "Evaluation of models trained on the " + str(x[0]) + " dataset"
        print(title)
        for i, sm in enumerate(scoremeasure):
            if(errorbar):
                plot_stability(x[1], sm, name +"_Stability_" + sm, scoremeasure_nice[i], "Reference Point Error")
            #plot_knn_with_pos(x[1], str(x[0])+ "Knn" + sm, sm,x_label=scoremeasure_nice[i], y_label="KNN Score")
            #plot_factors("scoremaze", sm, "Model", data=x[1], name=name +"_"+str(x[0]) + "_" + sm,y_label=scoremeasure_nice[i], x_label= "Evaluation Dataset", title=title,y_lim=scoremeasure_ylim[i])
            plot_factors("scoremaze", sm, "Model", data=x[1], name=name + "_" + str(x[0]) + "_" + sm,
                         y_label=scoremeasure_nice[i], x_label="Evaluation Dataset" , title=title,
                         y_lim=scoremeasure_ylim[i], )

            if(errorbar):
                group2 = x[1].groupby("scoremaze")
                for y in group2:
                    plot_factors("Model", sm, "number", data=y[1], name=name + "_" + str(x[0]) + "_Typecompare_"+str(y[0]) + sm,y_label=scoremeasure_nice[i],x_label= "Evaluation Dataset", title=title,y_lim=scoremeasure_ylim[i])

    #plot_factors("scoremaze", "pose_distance", "trainmaze", data=data)
    #plot_factors("type", "pose_distance", "trainmaze", data=data)
    #plot_factors("type","topological_score", "trainmaze",data)
    #plot_factors("type","knn_score", "scoremaze", data=data)chmod

def plot_stability(data, sm, name, y_label,x_label):
    plt.close("all")
    ax = sns.lineplot("number", sm, data=data, hue="Model", palette="muted")  # , ci='sd')
    if y_label is not None:
        plt.ylabel(y_label)
        #ax.set_ylabels(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
        #ax.set_xlabels(x_label)
    else:
        pass  # plt.title(title)
    print("plotting" + name)
    # show_values_on_bars(ax.axes)
    plt.savefig("plots/Comparisons/" + name + ".png", bbox_inches="tight", dpi=1200)

def plot_actiation_functions():
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    relu = lambda x: np.max(np.asarray([np.zeros(len(x)),x]),axis=0)
    x = np.linspace(-2,2,1000)
    plt.axis([-2,2,-2,2])
    plt.plot(x,sigmoid(x), label="sigmoid")
    plt.plot(x,tanh(x), label = "tanh")
    plt.plot(x, relu(x), label="relu")
    plt.title("Activation Functions")
    plt.legend()
    plt.savefig("Activationfunction.png")

def show_pause(show=False, pause=0.0):
    '''Shows a plot by either blocking permanently using show or temporarily using pause.'''
    if show:
        plt.ioff()
        plt.show()
    elif pause:
        plt.ion()
        plt.pause(pause)


def plot_maze(maze='nav01', margin=50.0, means=None, stds=None, figure_name=None, show=False, pause=False, Coloring=None):
    if figure_name is not None:
        plt.figure(figure_name)

    if 'nav01' in maze:
        x_max = 1000
        y_max = 500
        walls = np.array([
            # horizontal
            [[0, 500], [1000, 500]],
            [[400, 400], [500, 400]],
            [[600, 400], [700, 400]],
            [[800, 400], [1000, 400]],
            [[200, 300], [400, 300]],
            [[100, 200], [200, 200]],
            [[400, 200], [700, 200]],
            [[200, 100], [300, 100]],
            [[600, 100], [900, 100]],
            [[0, 0], [1000, 0]],
            # vertical
            [[0, 0], [0, 500]],
            [[100, 100], [100, 200]],
            [[100, 300], [100, 500]],
            [[200, 200], [200, 400]],
            [[200, 0], [200, 100]],
            [[300, 100], [300, 200]],
            [[300, 400], [300, 500]],
            [[400, 100], [400, 400]],
            [[500, 0], [500, 200]],
            [[600, 100], [600, 200]],
            [[700, 200], [700, 300]],
            [[800, 200], [800, 400]],
            [[900, 100], [900, 300]],
            [[1000, 0], [1000, 500]],
        ],dtype=np.float16)
        rooms = [
            # [[400, 200], 300, 200]
            ]
        if(means is not None and stds is not None ):
            pass
            plt.xlim(([-margin, 1000. + margin] - means[0])/stds[0])
            plt.ylim(([-margin, 500. + margin] - means[1])/stds[1])
        else:
            plt.xlim([-margin, 1000+margin])
            # plt.xlim([-100, 1101])
            plt.ylim([-margin, 500+margin])
            # plt.ylim([-100, 601])

    if 'nav02' in maze:
        x_max = 1500
        y_max = 900
        walls = np.array([
            # horizontal
            [[0, 900], [1500, 900]],
            [[100, 800], [400, 800]],
            [[500, 800], [600, 800]],
            [[800, 800], [1000, 800]],
            [[1100, 800], [1200, 800]],
            [[1300, 800], [1400, 800]],
            [[100, 700], [600, 700]],
            [[700, 700], [800, 700]],
            [[1000, 700], [1100, 700]],
            [[1200, 700], [1400, 700]],
            [[900, 600], [1200, 600]],
            [[1300, 600], [1500, 600]],
            [[0, 500], [100, 500]],
            [[1300, 500], [1400, 500]],
            [[100, 400], [200, 400]],
            [[1200, 400], [1400, 400]],
            [[300, 300], [800, 300]],
            [[900, 300], [1200, 300]],
            [[400, 200], [600, 200]],
            [[700, 200], [800, 200]],
            [[1200, 200], [1500, 200]],
            [[200, 100], [300, 100]],
            [[500, 100], [700, 100]],
            [[800, 100], [900, 100]],
            [[1100, 100], [1400, 100]],
            [[0, 0], [1500, 0]],
            # vertical
            [[0, 0], [0, 900]],
            [[100, 0], [100, 300]],
            [[100, 500], [100, 600]],
            [[100, 700], [100, 800]],
            [[200, 100], [200, 200]],
            [[200, 300], [200, 400]],
            [[200, 500], [200, 700]],
            [[300, 100], [300, 300]],
            [[400, 0], [400, 200]],
            [[500, 800], [500, 900]],
            [[700, 100], [700, 200]],
            [[700, 700], [700, 800]],
            [[800, 200], [800, 800]],
            [[900, 100], [900, 700]],
            [[1000, 0], [1000, 200]],
            [[1000, 700], [1000, 800]],
            [[1100, 700], [1100, 800]],
            [[1100, 100], [1100, 300]],
            [[1200, 800], [1200, 900]],
            [[1200, 400], [1200, 700]],
            [[1300, 200], [1300, 300]],
            [[1300, 500], [1300, 600]],
            [[1400, 300], [1400, 500]],
            [[1400, 700], [1400, 800]],
            [[1500, 0], [1500, 900]],
        ])
        rooms = [
            # [[900, 300], 300, 300]
            ]
        if (means is not None and stds is not None):
            pass
            plt.xlim(([-margin, 1500. + margin] - means[0]) / stds[0])
            plt.ylim(([-margin, 900. + margin] - means[1]) / stds[1])
        else:
            plt.xlim([-margin, 1500 + margin])
            plt.ylim([-margin, 900 + margin])

    if 'nav03' in maze:
        x_max = 2000
        y_max = 1300
        walls = np.array([
            # horizontal
            [[0, 1300], [2000, 1300]],
            [[100, 1200], [500, 1200]],
            [[600, 1200], [1400, 1200]],
            [[1600, 1200], [1700, 1200]],
            [[0, 1100], [600, 1100]],
            [[1500, 1100], [1600, 1100]],
            [[1600, 1000], [1800, 1000]],
            [[800, 1000], [900, 1000]],
            [[100, 1000], [200, 1000]],
            [[700, 900], [800, 900]],
            [[1600, 900], [1800, 900]],
            [[200, 800], [300, 800]],
            [[800, 800], [1200, 800]],
            [[1300, 800], [1500, 800]],
            [[1600, 800], [1900, 800]],
            [[900, 700], [1400, 700]],
            [[1500, 700], [1600, 700]],
            [[1700, 700], [1900, 700]],
            [[700, 600], [800, 600]],
            [[1400, 600], [1500, 600]],
            [[1600, 600], [1700, 600]],
            [[100, 500], [200, 500]],
            [[300, 500], [500, 500]],
            [[600, 500], [700, 500]],
            [[1400, 500], [1900, 500]],
            [[100, 400], [200, 400]],
            [[400, 400], [600, 400]],
            [[1500, 400], [1600, 400]],
            [[1700, 400], [1800, 400]],
            [[200, 300], [300, 300]],
            [[400, 300], [500, 300]],
            [[600, 300], [800, 300]],
            [[900, 300], [1100, 300]],
            [[1300, 300], [1500, 300]],
            [[1600, 300], [1700, 300]],
            [[100, 200], [200, 200]],
            [[500, 200], [600, 200]],
            [[800, 200], [1100, 200]],
            [[1200, 200], [1400, 200]],
            [[1500, 200], [1600, 200]],
            [[200, 100], [300, 100]],
            [[500, 100], [800, 100]],
            [[1000, 100], [1200, 100]],
            [[1400, 100], [1600, 100]],
            [[1800, 100], [1900, 100]],
            [[0, 0], [2000, 0]],
            # vertical
            [[0, 0], [0, 1300]],
            [[100, 0], [100, 300]],
            [[100, 400], [100, 1000]],
            [[200, 300], [200, 400]],
            [[200, 600], [200, 800]],
            [[200, 900], [200, 1000]],
            [[300, 100], [300, 600]],
            [[300, 800], [300, 1100]],
            [[400, 0], [400, 300]],
            [[400, 1200], [400, 1300]],
            [[500, 100], [500, 200]],
            [[600, 200], [600, 400]],
            [[600, 1100], [600, 1200]],
            [[700, 200], [700, 300]],
            [[700, 400], [700, 1100]],
            [[800, 100], [800, 200]],
            [[800, 300], [800, 500]],
            [[800, 600], [800, 700]],
            [[800, 1000], [800, 1100]],
            [[900, 0], [900, 100]],
            [[900, 300], [900, 600]],
            [[900, 900], [900, 1200]],
            [[1000, 100], [1000, 200]],
            [[1200, 100], [1200, 200]],
            [[1300, 0], [1300, 100]],
            [[1400, 100], [1400, 700]],
            [[1500, 700], [1500, 1000]],
            [[1500, 1100], [1500, 1200]],
            [[1600, 200], [1600, 400]],
            [[1600, 600], [1600, 700]],
            [[1600, 1000], [1600, 1100]],
            [[1600, 1200], [1600, 1300]],
            [[1700, 1100], [1700, 1200]],
            [[1700, 700], [1700, 800]],
            [[1700, 500], [1700, 600]],
            [[1700, 0], [1700, 300]],
            [[1800, 100], [1800, 400]],
            [[1800, 600], [1800, 700]],
            [[1800, 900], [1800, 1200]],
            [[1900, 800], [1900, 1300]],
            [[1900, 100], [1900, 600]],
            [[2000, 0], [2000, 1300]],
        ])
        rooms = [
                # [[300, 500], 400, 600],
                #  [[900, 800], 600, 400],
                #  [[900, 300], 500, 400],
                 ]
        if (means is not None and stds is not None):
            pass
            plt.xlim(([-margin, 2000. + margin] - means[0]) / stds[0])
            plt.ylim(([-margin, 1300. + margin] - means[1]) / stds[1])
        else:
            plt.xlim([-margin, 2000 + margin])
            plt.ylim([-margin, 1300 + margin])


    if means is not None:

        #walls -= means['pose'][:, :, :2]
        walls = walls -  means[:2]
    if stds is not None:
        #walls /= stds['pose'][:, :, :2]
        walls = walls / stds[:2]
    # color = (0.8, 0.8, 0.8)
    color = (0, 0, 0)

    plt.plot(walls[:, :, 0].T, walls[:, :, 1].T, color=color)
    
    for room in rooms:
        plt.gca().add_patch(Rectangle(*room, facecolor=(0.85, 0.85, 0.85), linewidth=0))

    if Coloring is not None:
        for x in range(0,x_max,100):
            for y in range(0,y_max,100):
                colors = Coloring.get_color(np.asarray([[x,y]]))
                plt.gca().add_patch(Rectangle([x,y],100,100, facecolor=(colors[0][0],colors[0][1],colors[0][2]), linewidth=0, alpha=0.7))
    plt.gca().set_aspect('equal')
    #plt.show()

def plot_trajectories(data, figure_name=None, show=False, pause=False, emphasizes=None, mincolor=0.0, similar = None, extras=None):
    if figure_name is not None:
        plt.figure(figure_name)
    #for i, trajectories in enumerate(data['s']):
    #    color = np.random.uniform(low=mincolor, high=1.0, size=3)
    #    plt.plot(trajectories[:, 0], trajectories[:, 1], color=color, linewidth=0.3, zorder=0)
    if emphasizes is not None:
        colors = [[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]
        for i, emphasize in enumerate(emphasizes):
            plt.plot(data['s'][emphasize, :, 0], data['s'][emphasize, :, 1], color=colors[i%3], linewidth=3, zorder=0)
            plt.plot(data['s'][emphasize, :, 0], data['s'][emphasize, :, 1], 'o', markerfacecolor='None',
             markeredgecolor=[0.0, 0.0, 0.0], markersize=16)
            plt.quiver(data['s'][emphasize, :, 0], data['s'][emphasize, :, 1], np.cos(data['s'][emphasize, :, 2]),
                   np.sin(data['s'][emphasize, :, 2]), color=[0.0, 0.0, 0.0], zorder=1, headlength=0, headaxislength=0, scale=40, width=0.004)
    if similar is not None:
        plt.plot(similar[:,0,0],similar[:,0,1], 'o', markerfacecolor='None',
             markeredgecolor=[0.0, 0.0, 0.0],
             markersize=5)
        plt.plot(similar[:,1,0],similar[:,1,1], 'o', markerfacecolor='None',
             markeredgecolor=[0.0, 0.0, 0.0],
             markersize=5)
        plt.quiver(similar[:,0,0],similar[:,0,1], np.cos(similar[:,0,2]),
                   np.sin(similar[:,0,2]), color=[0.0, 0.0, 0.0], zorder=1, headlength=0, headaxislength=0.1, scale=50, width=0.05)        
        plt.quiver(similar[:,1,0],similar[:,1,1], np.cos(similar[:,1,2]),
                   np.sin(similar[:,1,2]), color=[0.0, 0.0, 0.0], zorder=1, headlength=0, headaxislength=0.1, scale=50, width=0.05)        
        for x,y in similar:
            plt.plot([x[0],y[0]],[x[1],y[1]],color=[0.0,1.0,0.0],linewidth=0.5)
    if extras is not None:
        #ml=np.asarray([(-40,0),(20,0),(0,20),(20,0),(-40,0),(-40,0),(0,20),(0,20)])
        ml = np.asarray([(-40, 0), (20, 0), (0, 20), (-40, 0), (0, 20)])
        for i,  e in enumerate(extras):

            plt.annotate(str(chr(i+97))+")",(data['s'][e[0],e[1],0],data['s'][e[0],e[1],1]), xytext=(data['s'][e[0],e[1],0] + ml[i,0],data['s'][e[0],e[1],1] + ml[i,1]) ,
                         arrowprops=dict( arrowstyle = '->', alpha=0.8, color=[0.2,0.4,0.2]),size=21.) #width=1,headlength=0.8,headwidth=0.2))
            plt.plot()
    plt.gca().set_aspect('equal')
    show_pause(show, pause)

def plot_observations(data, i=0, n=16, figure_name=None, show=False, pause=False):

    plt.figure(figsize=(8,2))
    for i in range(n):
        # plt.figure('Normalized image')
        # plt.gca().clear()
        # plt.imshow(0.5 + rgbds[i, :, :, :3]/10, interpolation='nearest')
        # plt.pause(0.001)
        #
        # plt.figure('Depth image')
        # plt.gca().clear()
        # plt.imshow(0.5 + rgbds[i, :, :, 3] / 10, interpolation='nearest', cmap='coolwarm', vmin=0.0, vmax=1.0)
        # plt.pause(0.001)


        # plt.gca().clear()
        plt.subplot(2, 8, i+1)
        plt.imshow(np.clip(data['o'][0, i, :, :, :]/255.0, 0.0, 1.0), interpolation='nearest')
        plt.axis('off')
        # plt.tight_layout(pad=0.1)
        # plt.pause(0.1)
    plt.savefig("Observations.png")
    show_pause(show, pause)

def load_and_plot_obs(filename = "mazeid100", use_cycles = False, just_show = False,plot_seq_video=False):
    #data = data_utils.load_data(filename="nav01short_train",steps_per_episode=400)
    if(use_cycles):
        for cycle in range(10):
            data = data_utils.load_data(data_path="../agents/maze_data/" ,filename=filename + "cyclus{}".format(cycle), steps_per_episode=200 , num_episodes=150)
            #data = data_utils.load_data(filename="nav01rlnew_train",steps_per_episode=400)
            print(np.shape(data['s']))
            plot_maze()
            plot_trajectories(data,show=True,emphasizes=list(range(1)))
            plot_maze()
            plot_trajectories(data, show=True, emphasizes=list(range(147,148)))
            #plot_observations(data)
    elif(just_show):
        data = data_utils.load_data(data_path="../agents/maze_data/" ,filename=filename, steps_per_episode=200 , num_episodes=100)


        for t in range(100):
            plt.figure(figsize=(10, 10))
            for i in range(100):
                # plt.figure('Normalized image')
                # plt.gca().clear()
                # plt.imshow(0.5 + rgbds[i, :, :, :3]/10, interpolation='nearest')
                # plt.pause(0.001)
                #
                # plt.figure('Depth image')
                # plt.gca().clear()
                # plt.imshow(0.5 + rgbds[i, :, :, 3] / 10, interpolation='nearest', cmap='coolwarm', vmin=0.0, vmax=1.0)
                # plt.pause(0.001)

                # plt.gca().clear()
                plt.subplot(10, 10, i + 1)
                plt.imshow(np.clip(data['o'][t, i, :, :, :] / 255.0, 0.0, 1.0), interpolation='nearest')
                plt.axis('off')
            plt.show()
        return
        #data = data_utils.load_data(filename="nav01rlnew_train",steps_per_episode=400)
        print(np.shape(data['s']))
        #plot_maze()
        #plot_trajectories(data,show=True,emphasizes=list(range(10)))
        #plot_trajectories(data, show=True, emphasizes=list(range(490,500)))
        for i in range(100,110):
            print(i)
            plot_maze()
            plot_trajectories(data, show=True, emphasizes=list(range(i,i+1)))
            
        #plot_observations(data)

    elif(plot_seq_video):
        data = data_utils.load_data(data_path="maze_data/", filename="nav01_plotting_train", steps_per_episode=100,
                                    num_episodes=2)
        epoch_data = {}
        for i in range(99):
            plt.close("all")

            plt.figure(figsize=(15, 15))
            #gs = gridspec.GridSpec(1,20)

            plt.subplot(2,1 ,2)
            #plt.subplot(gs[0, 0:13])
            plot_maze(margin=0)
            epoch_data['s'] = data['s'][:, i:i+1]
            plot_trajectories(epoch_data, show=False, emphasizes=[0])
            plt.axis("off")
            #plt.savefig("movietraj{:02d}".format(i), bbox_inches="tight")
            #plt.close("all")
            plt.subplot(2, 1,1)
            #plt.subplot(gs[0, 13:])
            plt.imshow(np.clip(data['o'][0, i, :, :, :3] / 255.0, 0.0, 1.0), interpolation='nearest')
            plt.axis("off")
            plt.savefig("movieobs{:02d}".format(i), bbox_inches="tight")
    else:
        data = data_utils.load_data(data_path="maze_data/", filename="nav01_plotting_train", steps_per_episode=100,
                                    num_episodes=2)
        #obs_list = [(0, 2), (0, 43), (0, 66), (0, 81), (1, 0), (1, 41), (1, 92), (1, 98)]
        obs_list = [(0, 2), (0, 43), (0, 66), (1, 0), (1, 98)]

        topo_col = coloring.Coloring()
        print(np.shape(data['s']))
        plot_maze(Coloring=topo_col)
        plot_trajectories(data, show=True,emphasizes=[0,1], extras=obs_list)

        tr = 1


        plt.figure(figsize=(4, 2))
        j=0
        for i in obs_list:
            j +=1
            plt.subplot(2, 4, j )
            plt.imshow(np.clip(data['o'][i[0], i[1], :, :, :3] / 255.0, 0.0, 1.0), interpolation='nearest')
            plt.axis('off')
            # plt.tight_layout(pad=0.1)
            # plt.pause(0.1)
        plt.savefig("SelectedObservations2.png")

        """

        for i in range(100):

            print(i)
            fig, ax = plt.subplots(num=str(i))
            plot_maze(figure_name=str(i))
            ax.scatter(data['s'][tr,i,0], data['s'][tr,i,1])
            #plt.imshow(data['o'][0, i,:,:,:3])#,aspect='auto', extent=(40, 40, 80, 80), zorder=-1)

            plt.autoscale(enable=True, axis=u'both')

            xrng = plt.xlim()
            yrng = plt.ylim()
            scale = .4  # the image takes this fraction of the graph
            ax.imshow(np.clip(data['o'][tr, i, :, :, :3]/255.0, 0.0, 1.0), interpolation='nearest', aspect='auto', extent=(
            xrng[0], xrng[0] + scale * (xrng[1] - xrng[0]), yrng[0], yrng[0] + scale * (yrng[1] - yrng[0])), zorder=-1)
            plt.xlim(xrng)
            plt.ylim(yrng)
            plt.show()
        """

def view_data(data):
    # overview plot
    for poses in data['s']:
        plt.figure('Overview')
        plt.plot(poses[:, 0], poses[:, 1])

        # # sample plot
        # for poses, velocities, rgbds in zip(data['pose'], data['vel'], data['rgbd']):
        #     # for poses in data['pose']:
        #     plt.ioff()
        #     plt.figure('Sample')
        #     # plt.plot(poses[:, 0], 'r-')
        #     # plt.plot(poses[:, 1], 'g-')
        #     plt.plot(poses[:, 2], 'b-')
        #     # plt.plot(velocities[:, 0], 'r--')
        #     # plt.plot(velocities[:, 1], 'g--')
        #     plt.plot(velocities[:, 2], 'b--')
        #     plt.show()
        #
        #     # for i in range(100):
        #     #     plt.figure('Normalized image')
        #     #     plt.gca().clear()
        #     #     plt.imshow(0.5 + rgbds[i, :, :, :3]/10, interpolation='nearest')
        #     #     plt.pause(0.001)
        #     #
        #     #     plt.figure('Depth image')
        #     #     plt.gca().clear()
        #     #     plt.imshow(0.5 + rgbds[i, :, :, 3] / 10, interpolation='nearest', cmap='coolwarm', vmin=0.0, vmax=1.0)
        #     #     plt.pause(0.001)
        #     #
        #     #     plt.figure('Real image')
        #     #     plt.gca().clear()
        #     #     plt.imshow((rgbds*stds['rgbd'][0] + means['rgbd'][0])[i, :, :, :3]/255.0, interpolation='nearest')
        #     #     plt.pause(0.1)


def plot_sup():
    #TODO: Load dataset, phi(dataset) for sup and unsupervised, validationnet for unsupervised, plot trajectory

    os.environ["CUDA_VISIBLE_DEVICES"]=""
    mixture = True
    task = "nav01"
    num_episodes = 50
    use_lstm = True
    sup_model = "saved_models/supervised_mix_do0.5hn512hs"
    unsup_model = "saved_models/mix_nl32_even"
    cs = 40
    EXCLUDE = 10
    state_step_sizes_train, data = main.load_data(task, mixture, num_episodes)

    dim_img = np.shape(data['observations'])[2]

    print(tf.contrib.framework.get_name_scope())

    with tf.variable_scope(tf.contrib.framework.get_name_scope(), reuse=tf.AUTO_REUSE):
        """
        srl_sup = temporal_net.temporal_net(dim_img, 4, learning_rate=0.0001, l1_reg=0.01, num_epochs=1, maze=1,
                                        use_lstm=use_lstm, hidden_nodes=512, skip_connection=True)

        sup_state = main.flatten_seq(
                srl_sup.phi(data['observations'][:, :cs], data['actions'][:, :cs], sup_model, exclude=EXCLUDE,
                    meanfile="saved_models/mixinfo.npz"))
        print("Supervised worked")
        np.savetxt("plots/supervised_states",sup_state)
        """
        sup_state = np.loadtxt("plots/supervised_states")
        srl_unsup = temporal_net.temporal_net(dim_img, 5, learning_rate=0.0001, l1_reg=0.01, num_epochs=1, maze=1,
                                        use_lstm=use_lstm, hidden_nodes=512, skip_connection=True)

        unsup_state = main.flatten_seq(
            srl_unsup.phi(data['observations'][:, :cs], data['actions'][:, :cs], unsup_model, exclude=EXCLUDE,
                meanfile="saved_models/mixinfo.npz"))



    without_first_pose = data['pose'][:,
                         EXCLUDE:cs]
    flatten_pose = main.flatten_seq(without_first_pose)

    mean_pose = np.mean(flatten_pose, axis=0, keepdims=True)
    std_pose = np.std(flatten_pose, axis=0, keepdims=True)

    flatten_pose[:, :2] = (flatten_pose[:, :2] - mean_pose[:, :2]) / std_pose[:, :2]
    pose = np.concatenate((flatten_pose[:, :2], [[np.cos(x), np.sin(x)] for x in flatten_pose[:, 2]]), axis=1)
    not_normed_pose = main.flatten_seq(without_first_pose)



    inp_size = np.shape(pose)[0]
    inp_dim = 5
    out_dim = np.shape(pose)[1]

    train_border = int(inp_size * 7 / 10)
    topo_coloring = coloring.Coloring(maze=1)
    topo_indices = topo_coloring.get_topo_pose_index(not_normed_pose)

    pose_counter = Counter(topo_indices)
    weights = np.asarray([len(topo_indices) / pose_counter[x] for x in topo_indices[:train_border]])
    vali_weights = np.asarray([len(topo_indices) / pose_counter[x] for x in topo_indices[train_border:]])

    # Validation net for pose
    vali_net = validation_net.validation_net(inp_dim, out_dim)
    vali_net.learn(unsup_state[:train_border], pose[:train_border], weights=weights,
                   validation_data=unsup_state[train_border:],
                   validaton_pose=pose[train_border:], validation_weight=vali_weights)

    unsup_val_state = vali_net.predict(unsup_state)

    pm = lambda : plot_maze(means=mean_pose[0, :], stds=std_pose[0, :])

    main.plot_validation_traj(sup_state, data['pose'], cs, unsup_val_state, 50, pm)

def plot_colored_maze():
    starts=[[(50,450,"R"),(950,450,"G"),(650,150,"B")],[(150,750,"R"),(1350,550,"G"),(1350,750,"B")],[(250,350,"R"),(1450,550,"G"),(1750,750,"B")]]
    for i in range(1,4):
        col = coloring.Coloring(maze=i)

        plot_maze(maze='nav0{}'.format(i), Coloring=col)
        for s in starts[i-1]:
            print(s)
            plt.plot(s[0],s[1],  color=[0,0,0], marker='*', markersize = 5)

            #plt.annotate(s[2],s[0],s[1],size=15.)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig("plots/ColoredMaze{}.png".format(i), dpi=600, bbox_inches="tight")
        plt.close('all')
def plot_annotated_rl_maze():
    plt.close('all')
    plot_maze()
    plt.gca().add_patch(Rectangle([0, 0], 1000, 500, facecolor=(.80, 0.1, 0.1), linewidth=0))
    plt.gca().add_patch(Rectangle([400, 200], 100, 100, facecolor=(0,1.0,0), linewidth=0))
    #for x in range(10):
    #    for y in range(5):
    #        plt.plot(x*100+50,y*100+50,marker="v",color=(0,0,1.0), markersize = 3)

    plt.plot(650,150,marker="v", color=(0,0,0), markersize = 3)
    plt.annotate("Fixed Start", [650,150], [605,170], fontsize = 11) #,arrowprops = {'arrowstyle' : "->"}
    #plt.annotate("Goal Area", [410,280],[410,280], fontsize  = 11)
    plt.annotate("+100", [400,230],[400,230], fontsize  = 11)
    plt.annotate("-1", [525, 430], [525,430], fontsize = 11)
    plt.annotate("-1", [025, 230], [025, 230], fontsize=11)
    plt.annotate("-1", [925, 130], [925, 130], fontsize=11)
    plt.axis("off")
    plt.savefig("plots/AnnotaedRLMaze2.png",dpi = 1200,bbox_inches="tight" )
if __name__ == '__main__':
    pass
    """ #Plot Comparison of Different RL Models
    colors = ["#377eb8","#ff7f00","black","#4daf4a","#e41a1c"]
    num_episodes = [7000, 7000, 7000, 7000, 7000]
    names = ["Similar Point", "Ground Truth Pose", "Random policy", "64 Landmarks", "Supervised"]
    ids = [[x, x + 1, x + 2, x + 3] for x in [4034, 3054, 5004, 4074, 4044]]
    plot_grouped_errorbars(ids, num_episodes, names=names, fig_name="Single Reward", file_name="ModelComparisonSingle",colors=colors)
    ids = [[x, x + 1, x + 2, x + 3] for x in [4030, 3050, 5000, 4070, 4040]]
    plot_grouped_errorbars(ids, num_episodes, names=names, fig_name="Topological Reward", file_name="ModelComparisonTopo",colors=colors)
    colors = ["#377eb8","#ff7f00",  "#984ea3", "#4daf4a","black"]
    num_episodes = [5000,7000, 5000, 7000, 7000]
    names = ["5 Landmarks","16 Landmarks","32 Landmarks","64 Landmarks", "Random policy"]
    #ids = [[x, x + 1, x + 2, x + 3] for x in [4084, 4014, 4074, 5004]]
    #plot_grouped_errorbars(ids, num_episodes, names=names, fig_name="LMSingle Reward")
    ids = [[x, x + 1, x + 2, x + 3] for x in [4120,4080,4010, 4070, 5000]]
    plot_grouped_errorbars(ids, num_episodes, names=names, fig_name="Topological Reward", file_name="5LandmarksComparisonTopo",colors=colors)
    """

    #Plots barplots of different models
    #load_scores(5000, name="Results", file="results.csv", choose=["nl5","nl16","nl32","nl64","sp","supervised"])

    #Other used plots
    #plot_colored_maze()
    #data_and_plot()
    #plot_actiation_functions()
    #plot_maze(Coloring=coloring.Coloring())
    #plot_annotated_rl_maze()
    #load_and_plot_obs(plot_seq_video=True)