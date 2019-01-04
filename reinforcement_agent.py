# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:51:30 2018

@author: marco
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import pprint
import sys
import matplotlib.pyplot as plt
import q_net
import coloring
import temporal_net
import tensorflow as tf
import deepmind_lab
import os
import time
import pickle
plt.ion()

dir_path = os.path.dirname(os.path.realpath(__file__))
save_dir = dir_path + "/saved_data/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plotting = False
SUPERVISED = False

def _action(*entries):
    return np.array(entries, dtype=np.intc)


class Reinforcement_Agent:
    """Reinforcement Agent for DeepMind Lab."""

    ACTIONS = {
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'look_left': _action(-60, 0, 0, 0, 0, 0, 0),
        'look_right': _action(60, 0, 0, 0, 0, 0, 0),
        'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
        'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
        'backward': _action(0, 0, 0, -1, 0, 0, 0),
    }
    ACTIONS_LIST = [_action(0, 0, 0, 1, 0, 0, 0),
                    _action(-60, 0, 0, 0, 0, 0, 0),
                    _action(60, 0, 0, 0, 0, 0, 0),
                    _action(0, 0, -1, 0, 0, 0, 0),
                    _action(0, 0, 1, 0, 0, 0, 0),
                    _action(0, 0, 0, -1, 0, 0, 0)]

    plot_colorbar = True

    def __init__(self, dim_img=4096, state_dim=4, maze=1, q_lr=0.99, modelpath_srl=None, split_gpu=False,
                 hidden_nodes=512, run_id=0, learning_rate=0.005, use_srl=False):

        self.turning_count = 0
        self.model_path_srl = modelpath_srl
        self.buffer = experience_buffer()
        self.last_state = None
        self.last_action = None

        if split_gpu:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        self.q = q_net.q_net(state_dim, len(self.ACTIONS), hidden_nodes=hidden_nodes, middle_layer=True, regularization=1, reg_par=0.001, learning_rate=learning_rate)
        self.q_net_initializer = tf.global_variables_initializer()
        self.sess.run(self.q_net_initializer)


        if use_srl:
            self.state_rep = True
            self.srl_hn = 512
            srl_model_name = "SRL_RL_id{}".format(run_id)
            landmarks=20
            self.srl = temporal_net.temporal_net(dim_img, state_dim, learning_rate=0.0001, l1_reg=0.01,
                                                 num_epochs=3000,
                                                 plot=False, model_file=srl_model_name, similar_points=landmarks,
                                                 maze=maze, use_landmark=False, supervised=SUPERVISED,
                                                 hidden_nodes=self.srl_hn,
                                                 dropout_prob=0.5)

            self.lstm_state = np.zeros((2, 2, 1, self.srl_hn))
            
            self.sess.run(tf.global_variables_initializer())
            if ( modelpath_srl[-1] != "/"):
                self.srl.restore_model(self.sess, modelpath_srl,
                                   modelpath_srl+"_mean.npz")

            variables_names = [v.name for v in tf.trainable_variables()]

            print("Trainable variable Names : " , variables_names)
        else:
            self.state_rep = False

        self.tensorboard_writer = tf.summary.FileWriter(save_dir + "agents/logs/train" + str(run_id), self.sess.graph)
        self.step_count = 0
        self.target_update_interval = 10000
        self.q_lr = q_lr
        # self.batch_size = 500
        self.batch_size = 256
        self.id = run_id

        # For the wanderer purpose:
        self.turning_count = 0
        self.next_turn_direction = random.choice([self.ACTIONS['look_left'], self.ACTIONS['look_right']])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()
        self.buffer.save(save_dir + "agents/emergencysave")

    def step(self, reward, image, encode, learn=True):
        """Gets an image state, and action and a reward, returns an action."""
        if(learn):
            self.step_count += 1

        if self.state_rep:
            # Predict current state
            raw_state, self.lstm_state = self.srl.phi_step_with_session(image, encode, self.sess, self.lstm_state)
            state = raw_state[-1]

        else:
            state = image

        # Adds last experience to replay buffer
        if np.sum(self.last_state is not None) > 0:
            self.buffer.add([self.last_state, self.last_action, reward, state])
            if learn and self.step_count % 4 == 0:
                self.q_learn()
        # Select next action
        next_action = self.select_action(state, learn)

        if self.step_count % self.target_update_interval == 0 and learn:
            self.q.copy_to_t_net(self.sess)

        self.last_state = state
        self.last_action = next_action

        return self.ACTIONS_LIST[next_action], state


    def update_when_done(self, reward, image, encode):
        if self.state_rep:
            # Predict current state
            raw_state, self.lstm_state = self.srl.phi_step_with_session(image, encode, self.sess, self.lstm_state)
            state = raw_state[-1]
        else:
            state = image
        if np.sum(self.last_state is not None) > 0:
            self.buffer.add([self.last_state, self.last_action, reward, state])
            self.q_learn()

    def q_learn(self):
        if self.step_count < self.batch_size:
            return

        s, a, r, s1 = self.buffer.sample(self.batch_size)
        batch_size = len(s)
        Q1 = np.argmax(self.q.predict(s1, self.sess), axis=1)

        Q2 = self.q.predict(s1, self.sess, target_net=True)[range(batch_size), Q1]

        target_q = r + self.q_lr * Q2

        if self.step_count % 2000 == 0:

            loss, summary = self.q.learn(s, target_q, a, self.sess, summary=True)
            self.tensorboard_writer.add_summary(summary, self.step_count)
        else:
            loss = self.q.learn(s, target_q, a, self.sess)

    def select_action(self, state, learn):
        q_values = self.q.predict(state[np.newaxis, :], self.sess)
        # epsilon greedy strategy
        eps = max(1 - (self.step_count * 0.000001), 0.1)
        if np.random.rand() <= eps and learn:
            return random.randint(0, len(self.ACTIONS_LIST) - 1)
        else:
            return np.argmax(q_values)

    def save(self, model_name="q_net"):
        self.q.save_model(self.sess, model_name)

    def load(self, q_net, replay_buffer=None, step=None):
        self.q.load_model(self.sess, save_dir + "agents/" + q_net)
        if replay_buffer:
            self.buffer.load(save_dir + "agents/" + replay_buffer)
        if step:
            self.step_count = step

    def reset(self):
        """ clearing the history, keep the model """
        self.last_state = None
        self.last_action = None
        if self.state_rep:
            self.lstm_state = np.zeros((2, 2, 1, self.srl_hn))

    def reset_rl_agent(self):
        #Reset Q_values
        self.sess.run(self.q_net_initializer)
        self.reset()
        self.buffer.reset()
        self.step_count = 0

    # Use the wanderer policy to generate a Dataset
    def wanderer_step(self, reward, image, encode):
        """Gets an image state and a reward, returns an action."""
        image = np.reshape(image, (32, 32, 4))
        height, width = image.shape[:2]
        scan = image[height // 2, :, -1]
        scan_max_i = np.argmax(scan)
        scan_max = scan[scan_max_i]
        scan_min_i = np.argmin(scan)
        scan_min = scan[scan_min_i]
        min_direction = scan_min_i / (width - 1) * 2.0 - 1.0
        max_direction = scan_max_i / (width - 1) * 2.0 - 1.0
        if random.random() < 0.1:
            next_action = random.choice(self.ACTIONS.values())
        else:
            if self.turning_count > 0:
                self.turning_count -= 1
                next_action = self.next_turn_direction
            else:
                if scan_max > 220:

                    # turn = max_direction * 15 - min_direction * 30
                    turn = max_direction * 20 - min_direction * 20
                    if max_direction > 0.5 and np.random.random() < 0.1:
                        strafe = 1
                    elif max_direction < -0.5 and np.random.random() < 0.1:
                        strafe = -1
                    else:
                        strafe = 0
                    # strafe = 0
                    if random.random() < 0.3:
                        forward = 1
                    else:
                        forward = 0
                    # forward = np.random.randint(0, 2)
                    self.next_turn_direction = random.choice(
                        [self.ACTIONS['look_left'], self.ACTIONS['look_right']])
                    next_action = _action(turn, 0, strafe, forward, 0, 0, 0)
                else:
                    self.turning_count = np.random.randint(1, 10)
                    next_action = self.next_turn_direction

        return next_action



class experience_buffer:
    def __init__(self, buffersize=1000000):
        self.buffersize = buffersize
        self.buffer = []
        self.kind_of_reward = {}
        self.reward_position = {}
        self.position = 0

    def add(self, experience):
        # Check if buffer is already Full, set at position if so, otherwise append
        if len(self.buffer) >= self.buffersize:
            self.buffer[self.position] = experience
        else:
            self.buffer.append(experience)

        #If reward not observed before, add to dict
        if(experience[2] not in self.kind_of_reward.keys()):
            self.kind_of_reward[experience[2]] = []
            self.reward_position[experience[2]] = 0

        #Add the observation to the seperate distinct reward buffer
        if(len(self.kind_of_reward[experience[2]])>= self.buffersize/10):
            self.kind_of_reward[experience[2]][int(self.reward_position[experience[2]])] = experience
        else:
            self.kind_of_reward[experience[2]].append(experience)

        self.reward_position[experience[2]] =   int((self.reward_position[experience[2]]+ 1) % int(self.buffersize/10))    

        self.position = int((self.position + 1) % self.buffersize)

    def sample(self, size=200):
        if size > len(self.buffer):
            sample = random.sample(self.buffer, len(self.buffer))
        else:
            sample = random.sample(self.buffer, size)
        #Add 10 samples of each observed reward
        for key in self.kind_of_reward.keys():
            sample_reward  =  random.sample(self.kind_of_reward[key],min(len(self.kind_of_reward[key]),10))
            sample = np.concatenate((sample, sample_reward))

        unfold = zip(*sample)
        return np.asarray(unfold[0]), np.asarray(unfold[1]), np.asarray(unfold[2]), np.asarray(unfold[3])

    def size(self):
        return len(self.buffer)

    def save(self, name="outfile"):
        with open(name, 'wb') as fp:
            pickle.dump(self.buffer, fp)

    def load(self, name="outfile"):
        with open(name, 'rb') as fp:
            self.buffer = pickle.load(fp)

    def reset(self, buffersize=1000000):
        self.__init__(buffersize)



class normal_reward:
    def __init__(self, goal=None):
        if goal is None:
            goal = [450, 250]
        self.goal = goal

    def check_reward(self, pose):
        if (self.goal[0] - 50 <= pose[0] <= self.goal[0] + 50 and  self.goal[1] - 50 <= pose[
            1] <= self.goal[1] + 50):
            return 100
        else:
            return -1

    def reset(self):
        pass


class Topological_reward:
    def __init__(self, goal=[450, 250], maze=1):
        self.topo_color = coloring.Coloring(maze=maze)
        self.topo_color.vectorize_topo_map()
        self.last_dist = -1
        self.goal = goal

    def check_reward(self, pose):
        dist = self.topo_color.get_topo_distance_by_pose(pose, self.goal)

        if dist == 0:
            reward = 100
        elif dist < self.last_dist:
            reward = 10
        elif dist > self.last_dist >= 0:
            reward = -10
        else:
            reward = -1
        self.last_dist = dist
        return reward

    def reset(self):
        self.last_dist = -1



def validation_run(env, agent, reward_class=None, use_srl=False, use_wanderer_step=False):
    env.reset()
    agent.reset()
    reward_class.reset()
    episode_length = 3000
    last_pose = None
    pos=[]
    cumultative_reward = 0
    for i in xrange(episode_length):
        if not env.is_running():
            print('Environment stopped early')
            # print(np.shape(np.asarray(episode_poses)))
            break
        obs = env.observations()
        reward = reward_class.check_reward(obs['POSE'])
        cumultative_reward += reward
        if np.sum(last_pose is not None) > 0:
            encode = relative_encode(obs['POSE'], last_pose) * np.random.normal(1.0, 0.1, 3)
        else:
            encode = np.asarray([0, 0, 0])
        last_pose = np.copy(obs['POSE'])

        if use_srl  or use_wanderer_step:
            image = np.reshape(obs['RGBD_INTERLACED'], 4096)
        else:
            image = np.copy(obs['POSE'])
            image = normalize_pose(image)

        # Reached goal area
        if reward >= 20:
            print("Validation agent reached the goal at step ", i)
            return i , cumultative_reward
        else:
            if use_wanderer_step:
                action = agent.wanderer_step(reward, image, encode)
            else:

                action, _ = agent.step(reward, image, encode, learn=False)
            reward_unused = env.step(action, num_steps=1)

    print("Validation agent did not reached the goal, ended up here: ", obs['POSE'])

    return episode_length, cumultative_reward


def learn_rl(agent, env, test_env, reward_class, episodes, episode_length, agent_id, collect_data = False, use_wanderer_step = False):
    log = {'rgbd': [], 'vel': [], 'pose': []}
    start_poses = []
    time_to_goal = []
    validation_time = []
    cumultative_reward_val = []
    cumultative_reward_randval = []
    cumultative_reward_train = []
    validation_random_time = []
    goal_starts = []
    training_performance = []
    mean_diffs = []
    for episode in xrange(episodes):
        episode_log = {'rgbd': [], 'vel': [], 'pose': []}
        last_pose = None
        env.reset()
        agent.reset()
        reward_class.reset()
        steps = episode_length
        cul_reward = 0
        for i in xrange(episode_length):
            if not env.is_running():
                print('Environment stopped early')
                break

            obs = env.observations()

            reward = reward_class.check_reward(obs['POSE'])
            cul_reward += reward

            if np.sum(last_pose is not None) > 0:
                encode = relative_encode(last_pose, obs['POSE']) * np.random.normal(1.0, 0.1, 3)
            else:
                encode = np.asarray([0, 0, 0])

            # Reached goal area
            if agent.state_rep or use_wanderer_step:
                image = np.reshape(obs['RGBD_INTERLACED'], 4096)
            else:
                image = np.copy(obs['POSE'])
                image = normalize_pose(image)
                
            if i == 0:
                start_poses.append(obs['POSE'])

            last_pose = obs['POSE']

            if reward >= 20:
                agent.update_when_done(reward, image, encode)
                #print("Reached the goal after {} steps".format(i), "startPose: ", start_poses[-1])
                goal_starts.append(start_poses[-1])
                time_to_goal.append(i)
                steps = i
                break
            else:
                if use_wanderer_step:
                    action = agent.wanderer_step(reward, image, encode)
                else:
                    action, state = agent.step(reward, image, encode)

                _ = env.step(action, num_steps=1)

                if collect_data:
                    episode_log['rgbd'].append(obs['RGBD_INTERLACED'])
                    episode_log['vel'].append(np.array([obs['VEL.TRANS'][0], obs['VEL.TRANS'][1], obs['VEL.ROT'][1]]))
                    episode_log['pose'].append(obs['POSE'])



        training_performance.append(steps)
        cumultative_reward_train.append(cul_reward)

        if collect_data == "short":
            len_episode = len(episode_log['vel'])
            if len_episode >= 200:
                start = np.random.randint(0, len_episode-199)
                for key in log.iterkeys():
                    log[key].extend(episode_log[key][start:start+200])

        elif collect_data =="long":
            len_episode = len(episode_log['vel'])
            if( len_episode > 1200):
                start = np.random.randint(0, len_episode - 1199)
                for key in log.iterkeys():
                    log[key].extend(episode_log[key][start:start+1200:6])


        if episode % 1000 == 999:
            agent.save(save_dir + "agents/Q_net{}id{}".format(episode + 1, agent_id))

        #Validation run from fixed start
        if episode % 20 == 0:
            val_t, cum_r = validation_run(test_env, agent, reward_class, use_srl=agent.state_rep, use_wanderer_step=use_wanderer_step)
            validation_time.append(val_t)
            cumultative_reward_val.append(cum_r)
        #Validation run from random start
        if episode % 20 == 0:
            val_t, cum_r = validation_run(env, agent, reward_class, use_srl=agent.state_rep, use_wanderer_step=use_wanderer_step)
            validation_random_time.append(val_t)
            cumultative_reward_randval.append(cum_r)


    #Save Evaluation Parameter:
    if(not collect_data):
        agent.buffer.save(save_dir + "agents/replaybuffer{}id{}".format(episodes, agent_id))
    goal_starts = np.asarray(goal_starts)
    np.savetxt(save_dir + "agents/goal_starts{}id{}".format(episodes, agent_id), goal_starts)
    np.savetxt(save_dir + "agents/validatintimes{}id{}".format(episodes, agent_id), np.asarray(validation_time))
    np.savetxt(save_dir + "agents/validatinrandomtimes{}id{}".format(episodes, agent_id),
               np.asarray(validation_random_time))
    np.savetxt(save_dir + "agents/trainingperformance{}id{}".format(episodes, agent_id),
               np.asarray(training_performance))
    np.savetxt(save_dir + "agents/CummultativeRewardVal{}id{}".format(episodes, agent_id),
               np.asarray(cumultative_reward_val))
    np.savetxt(save_dir + "agents/CummultativeRewardRandVal{}id{}".format(episodes, agent_id),
               np.asarray(cumultative_reward_randval))
    np.savetxt(save_dir + "agents/CummultativeRewardTrain{}id{}".format(episodes, agent_id),
               np.asarray(cumultative_reward_train))

    agent.save(save_dir + "agents/Q_netid{}".format(agent_id))
    print("Goal is reached in {} % of the time, after an average of {} steps".format(
        (100.0 * len(time_to_goal)) / episodes, np.average(time_to_goal)))

    return log

def run(episodes, episode_length, width, height, fps, level, split_gpu, agent_id, gpus, hidden_nodes,
        topological_reward=False, goal=[450, 250], q_net_file=None,
        learning_rate=0.005, srl_model="", replay_buffer_file = None, collect_data=False, use_wanderer=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    """Spins up an environment and runs the reinforcement agent."""
    env = deepmind_lab.Lab(
        level, ['RGBD_INTERLACED', 'VEL.ROT', 'VEL.TRANS', 'POSE'],
        config={
            'fps': str(fps),
            'width': str(width),
            'height': str(height)
        })
    test_env = deepmind_lab.Lab(
        'nav_maze_static_01_fixed_start', ['RGBD_INTERLACED', 'VEL.ROT', 'VEL.TRANS', 'POSE'],
        config={
            'fps': str(fps),
            'width': str(width),
            'height': str(height)
        })
    env.reset()
    test_env.reset()
    observation_spec = env.observation_spec()
    pprint.pprint(observation_spec)
    action_spec = env.action_spec()
    pprint.pprint(action_spec)
    sys.stdout.flush()
    print(os.getcwd())

    """ Start the reinforcement agent """
    if topological_reward:
        reward_class = Topological_reward(goal=goal)
    else:
        reward_class = normal_reward(goal=goal)
    if srl_model and not SUPERVISED:
        state_dim = 5
        
    else:
        state_dim = 4

    srl_path = save_dir + "saved_models/" + srl_model

    with Reinforcement_Agent(split_gpu=split_gpu, state_dim=state_dim, modelpath_srl=srl_path,
                             hidden_nodes=hidden_nodes, run_id=agent_id, learning_rate=learning_rate,
                             use_srl=("" != srl_model)) as agent:
        if q_net_file is not None:
            try:
                agent.load(q_net_file, replay_buffer=replay_buffer_file, step=1000000)
            except IOError:
                print("Replay buffer not found")

        log = learn_rl(agent, env, test_env, reward_class, episodes, episode_length, agent_id,collect_data=collect_data, use_wanderer_step=use_wanderer)

        # Save buffer
        if collect_data:
            np.savez(save_dir + "agents/maze_data/mazeid{}".format(agent_id)+str(collect_data), rgbd=log['rgbd'], vel=log['vel'],
                     pose=log['pose'])

    return 1

def extend_position(pose):
    return np.asarray([pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2])])


def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi


def relative_encode(start, end):
    abs_d_x = end[0] - start[0]
    abs_d_y = end[1] - start[1]
    d_theta = wrap_angle(wrap_angle(end[2] * np.pi / 180.0) - wrap_angle(start[2] * np.pi / 180.0))

    s = np.sin(wrap_angle(start[2] * np.pi / 180.0))
    c = np.cos(wrap_angle(start[2] * np.pi / 180.0))
    rel_d_x = c * abs_d_x + s * abs_d_y
    rel_d_y = s * abs_d_x - c * abs_d_y
    return np.asarray([rel_d_x, rel_d_y, d_theta])

def normalize_pose(pose):
    pose[0] = (pose[0] - 500.) / 500.
    pose[1] = (pose[1] - 250.) / 250.
    pose[2] = (pose[2] / 360. * 2 * np.pi)

    # add noise to pose:
    #pose = pose* np.random.normal(1.0, 0.1, 3)
    pose = extend_position(pose)
    return pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps_length', type=int, default=3000,
                        help='Number of steps to run the agent in each episode')
    parser.add_argument('--num_eps', type=int, default=1000,
                        help='Number of episodes to run the agent')
    parser.add_argument('--width', type=int, default=32,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=32,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=10,
                        help='Number of frames per second')
    parser.add_argument('--level_script', type=str, default='nav_maze_static_01',
                        help='The environment level script to load')
    parser.add_argument('--split_gpu', action="store_true", default=False,
                        help="If selected, the Q-net uses only a fourth of the gpu memory")
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which Gpu is used')
    parser.add_argument('--id', type=int, default=random.randint(1000, 2000),
                        help='Id of current agent')
    parser.add_argument('--hidden_q_nodes', type=int, default=256,
                        help='Number of hidden nodes in each hidden layer')
    parser.add_argument('--q_net_file', type=str, default=None,
                        help='Q-net to load ')
    parser.add_argument("--srl", type=str, default="",
                        help="Specify SRL file")
    parser.add_argument('--topological_reward', action="store_true", default=False,
                        help="If selected, topological reward is used")
    parser.add_argument('--collect_data', type=int, default=0,
                        help='1: short dataset collected, 2: long dataset collected')
    parser.add_argument('--wanderer', action="store_true", default=False,
                        help="If selected, the wanderer policy is used instead of Reinforcement Learning")

    args = parser.parse_args()

    if(args.collect_data ==1):
        collect_data = "short"
    elif(args.collect_data == 2):
        collect_data = "long"
    else:
        collect_data = False

    run(args.num_eps, args.eps_length, args.width, args.height, args.fps, args.level_script, args.split_gpu, args.id,
        args.gpu, args.hidden_q_nodes, args.topological_reward , q_net_file=args.q_net_file, srl_model=args.srl,
        collect_data=collect_data, use_wanderer=args.wanderer)
