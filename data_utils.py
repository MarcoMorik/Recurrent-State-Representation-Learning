import numpy as np
import matplotlib.pyplot as plt
import os
from plotting import *

def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

def load_data(data_path='maze_data/', filename='nav01_train', steps_per_episode=100,with_depth=False,num_episodes = 200):

    data = dict(np.load(os.path.join(data_path,  filename + '.npz')))
    # reshape data
    for key in data.keys():
        # 'vel': (100, 1000, 3), 'rgbd': (100, 1000, 32, 32, 4), 'pose': (100, 1000, 3)
        data[key] = np.reshape(data[key], [-1, steps_per_episode] + list(data[key].shape[1:])).astype('float32')

    # convert degrees into radients and
    for key in ['pose', 'vel']:
        data[key][:, :, 2] *= np.pi / 180
    # angles should be between -pi and pi
    data['pose'][:, :, 2] = wrap_angle(data['pose'][:, :, 2])

    abs_d_x = (data['pose'][:, 1:, 0:1] - data['pose'][:, :-1, 0:1])
    abs_d_y = (data['pose'][:, 1:, 1:2] - data['pose'][:, :-1, 1:2])
    d_theta = wrap_angle(data['pose'][:, 1:, 2:3] - data['pose'][:, :-1, 2:3])
    s = np.sin(data['pose'][:, :-1, 2:3])
    c = np.cos(data['pose'][:, :-1, 2:3])
    rel_d_x = c * abs_d_x + s * abs_d_y
    rel_d_y = s * abs_d_x - c * abs_d_y

    if(with_depth):
        image_dim=4
    else: 
        image_dim=3

    print("Data Dimensions : ", np.shape(data['pose']))
    
    np.random.seed(1)    
    #epoch_permutation = np.random.permutation(num_episodes)#[:num_episodes]
    epoch_permutation = range(num_episodes)
    # define observations, states, and actions for the filter, use current and previous velocity measurement as action
    # and ignore the last timestep because we don't have the velocity of that step
    return {'o': data['rgbd'][epoch_permutation, 1:, :, :, :image_dim],
            's': data['pose'][epoch_permutation, 1:, :],
            'a': np.concatenate([rel_d_x, rel_d_y, d_theta], axis=-1)[epoch_permutation]}
            # 'a': np.concatenate([data['vel'][:, :-1, None, :], data['vel'][:, 1:, None, :]], axis=-2)}

"""Retun data as dictionary with keys:
observations (num_episodes,episode_size,32*32*4)
actions (num_episodes,episode_size)
rewards (num_episodes, episode_size)"""
def load_data_tensorflow(data_path='maze_data/', filename='nav01_train', steps_per_episode=100,with_depth=True,episode_shape=True,num_episodes = 200):
    train_data = load_data(data_path, filename, steps_per_episode, with_depth, num_episodes)    
    means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(train_data)
    #plot_trajectories(train_data, show=True, pause=0.0)
    #actions = discretize_action(train_data['a'])
    noise_factor = 1
    #actions = actions / state_step_sizes
    actions = train_data['a'] * np.random.normal(1.0, 0.1 * noise_factor, train_data['a'].shape)
    reward = calculate_reward(train_data['s'])
    
    num_obs = np.shape(actions)[0] * np.shape(actions)[1]
    dim_img = np.prod(np.shape(train_data['o'])[2:])
    episode_starts = np.zeros(num_obs)
    episode_starts[0::(steps_per_episode-1)] = True
    
    print("Train data: " ,np.shape(train_data['o']))
    if(episode_shape):
        data = {'observations': np.reshape(train_data['o'],(np.shape(actions)[0],np.shape(actions)[1],dim_img)),
                'pose': train_data['s'],
                'actions': actions,
                 'rewards': reward}
    else:
    
        data = {'observations': np.reshape(train_data['o'],(num_obs,dim_img)),
                'actions': np.reshape(actions,(num_obs,np.shape(actions)[2])),
                 'rewards': np.reshape(reward,(num_obs,)),
                     'pose': np.reshape(train_data['s'](num_obs,3)),
                 'episode_starts': episode_starts }
    return state_step_sizes , data
    

def compute_staticstics(data):

    means = dict()
    stds = dict()
    state_step_sizes = []
    state_mins = []
    state_maxs = []

    for key in 'osa':
        # compute means
        means[key] = np.mean(data[key], axis=(0, 1), keepdims=True)
        if key == 's':
            means[key][:, :, 2] = 0  # don't touch orientation because we'll feed this into cos/sin functions
        if key == 'a':
            # TODO
            # means[key][:, :, :, :] = 0  # don't change means of velocities, 0.0, positive and negative values have semantics
            means[key][:, :, :] = 0  # don't change means of velocities, 0.0, positive and negative values have semantics

        # compute stds
        axis = tuple(range(len(data[key].shape) - 1))  # compute std by averaging over all but the last dimension
        stds[key] = np.std(data[key] - means[key], axis=axis, keepdims=True)
        if key == 's':
            stds[key][:, :, :2] = np.mean(stds[key][:, :, :2])  # scale x and by by the same amount
        if key == 'a':
            # TODO
            # stds[key][:, :, :, :2] = np.mean(stds[key][:, :, :, :2])  # scale x and by by the same amount
            stds[key][:, :, :2] = np.mean(stds[key][:, :, :2])  # scale x and by by the same amount

    # compute average step size in x, y, and theta for the distance metric
    for i in range(3):
        steps = np.reshape(data['s'][:, 1:, i] - data['s'][:, :-1, i], [-1])
        if i == 2:
            steps = wrap_angle(steps)
        state_step_sizes.append(np.mean(abs(steps)))
    state_step_sizes[0] = state_step_sizes[1] = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes = np.array(state_step_sizes)

    # compute min and max in x, y and theta
    for i in range(3):
        state_mins.append(np.min(data['s'][:, :, i]))
        state_maxs.append(np.max(data['s'][:, :, i]))
    state_mins = np.array(state_mins)
    state_maxs = np.array(state_maxs)

    return means, stds, state_step_sizes, state_mins, state_maxs


def split_data(data, ratio=0.8, categories=['train', 'val']):
    split_data = {categories[0]: dict(), categories[1]: dict()}
    for key in data.keys():
        split_point = int(data[key].shape[0] * ratio)
        split_data[categories[0]][key] = data[key][:split_point]
        split_data[categories[1]][key] = data[key][split_point:]
    for key in split_data.keys():
        print('SPLIT --> {}: {}'.format(key, len(split_data[key]['s'])))
    return split_data


def reduce_data(data, num_episodes):
    new_data = dict()
    for key in 'osa':
        new_data[key] = data[key][:num_episodes]
    return new_data

def noisyfy_data(data, noise_factor = 1.0):
    print("noisyfying data ... ")
    new_data = dict()
    new_data['s'] = data['s']
    new_data['a'] = data['a'] * np.random.normal(1.0, 0.1 * noise_factor, data['a'].shape)
    new_o = np.zeros([data['o'].shape[0], data['o'].shape[1], 24, 24, 3])
    for i in range(data['o'].shape[0]):
        for j in range(data['o'].shape[1]):
            offsets = np.random.random_integers(0, 8, 2)
            new_o[i, j] = data['o'][i, j, offsets[0]:offsets[0]+24, offsets[1]:offsets[1]+24, :]
    new_o += np.random.normal(0.0, 20 * noise_factor, new_o.shape)
    # for i in range(data['o'].shape[0]):
    #     for j in range(data['o'].shape[1]):
    #         plt.figure()
    #         plt.imshow(new_o[i,j]/255, interpolation='nearest')
    #         plt.figure()
    #         plt.imshow(data['o'][i,j]/255, interpolation='nearest')
    #         plt.show()
    new_data['o'] = new_o
    return new_data

def noisify_only_odom(data, noise_factor = 1.0):
    print("noisyfying only odometry ... ")
    new_data = dict()
    new_data['s'] = data['s']
    new_data['a'] = data['a'] * np.random.normal(1.0, 0.1 * noise_factor, data['a'].shape)
    new_o = np.zeros([data['o'].shape[0], data['o'].shape[1], 24, 24, 3])
    for i in range(data['o'].shape[0]):
        for j in range(data['o'].shape[1]):
            offsets = (4, 4)
            new_o[i, j] = data['o'][i, j, offsets[0]:offsets[0]+24, offsets[1]:offsets[1]+24, :]
    new_data['o'] = new_o
    return new_data

def dont_noisyfy_data(data):
    print("NOT noisyfying data ... ")
    new_data = dict()
    new_data['s'] = data['s']
    new_data['a'] = data['a']
    new_o = np.zeros([data['o'].shape[0], data['o'].shape[1], 24, 24, 3])
    for i in range(data['o'].shape[0]):
        for j in range(data['o'].shape[1]):
            offsets = (4, 4)
            new_o[i, j] = data['o'][i, j, offsets[0]:offsets[0]+24, offsets[1]:offsets[1]+24, :]
    new_data['o'] = new_o
    return new_data


def make_batch_iterator(data, batch_size=32, seq_len=10):
    # go through data and select a subsequence from each sequence
    # seed list allows to go through a list of repeating batches, a seed list of length 100 will return the first batch again at run 101
    while True:
        episodes = np.random.random_integers(0, len(data['s']) - 1, size=batch_size)
        start_steps = np.random.random_integers(0, len(data['s'][0]) - seq_len - 1, size=batch_size)
        # print('episodes', episodes)
        # print('start_steps', start_steps)
        batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for i, j in zip(episodes, start_steps)]) for k in data.keys()}
        yield batches

def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10):
    # go through data and select a subsequence from each sequence
    # seed list allows to go through a list of repeating batches, a seed list of length 100 will return the first batch again at run 101
    repeating_episodes = np.random.random_integers(0, len(data['s']) - 1, size=[epoch_len, batch_size])
    repeating_start_steps = np.random.random_integers(0, len(data['s'][0]) - seq_len - 1, size=[epoch_len, batch_size])
    while True:
        for episodes, start_steps in zip(repeating_episodes, repeating_start_steps):
            batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for i, j in zip(episodes, start_steps)]) for k in data.keys()}
            yield batches

def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    num_episodes = len(data['s'])
    num_start_steps = len(data['s'][0]) - seq_len
    batch_indices = [(i, j) for i in range(num_episodes) for j in range(num_start_steps)]
    while batch_indices != []:
        # for bis in batch_indices[::batch_size]
        #range(len(batch_indices) // batch_size + 1):
        batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for (i, j) in batch_indices[:batch_size]]) for k in data.keys}
        batch_indices = batch_indices[batch_size:]
        yield batches


def calculate_reward(pos_data):
    num_batches = np.shape(pos_data)[0]
    batch_size = np.shape(pos_data)[1]
    pos_data = np.reshape(pos_data, (num_batches*batch_size,3))
    reward = np.zeros(np.shape(pos_data)[0])
    dead_end = [x for x in range(np.shape(pos_data)[0]-1) if (pos_data[x,0]<=100 and pos_data[x,1] >=300) \
                                                    or (pos_data[x,0]>=400 and pos_data[x,0] <=600 and pos_data[x,1] <= 200) ]
    box = [x for x in range(np.shape(pos_data)[0]-1) if (pos_data[x,0]>=400 and pos_data[x,0] <= 700 and pos_data[x,1] <=400 and pos_data[x,1] >= 200)]       
    
    reward[dead_end] = -1
    reward[box] = 2
    
    return np.reshape(reward,(num_batches,batch_size))    
        
def discretize_action(action):
    #Input a list of actions of shape [relative_x, relative_y, d_theta]
    new_shape = np.shape(action)[:2]+(9,)
    new_action= np.zeros(new_shape)    
    for i in range(np.shape(action)[0]):
        for j in range(np.shape(action)[1]):
            if(action[i,j,0]>=0):
                new_action[i,j,0]=1 # Forwards
            else:
                new_action[i,j,1]=1 # Backwards
            if(action[i,j,1]>=0):
                new_action[i,j,2]=1 # To the left
            else:
                new_action[i,j,3]=1 # To the right
            if(action[i,j,2]>=0):
                new_action[i,j,4]=1 # Turn to the left
            else:
                new_action[i,j,5]=1 # Turn to the right
            if(np.abs(action[i,j,0]) >=80):
                new_action[i,j,6]=1 # Fast forward
            if(np.abs(action[i,j,1]) >=40):
                new_action[i,j,7]=1 # Fast left/right
            if(np.abs(action[i,j,2]) >=0.78):
                new_action[i,j,8]=1 # Fast turn
            
    return new_action
if __name__ == '__main__':

    task = 'nav01'

    data = load_data(filename=task + '_train')
    #print(np.mean(np.abs(data['a'][0,:,:]),axis=0))
    
    actions = discretize_action(data['a'])[0,:,:]
    #print(np.where(np.prod(actions[[range(90)]]== actions[0],axis=1)))[0]
    find_same_actions = lambda index, minibatch: np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1))[0]
    same = find_same_actions(4,range(98))
    print(same)
    print(data['a'][0,same,:])
    
    reward = calculate_reward(data['s'])
    
    #print(np.shape(np.where(reward[0,]==-1)))
    #print(data['s'][0,np.where(reward[0,]==-1)[0],:])
#    data = noisyfy_data(data)

    # data = split_data(data)
    # means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data)
"""
    # batch_iterator = make_batch_iterator(data['train'])
    plt.figure()
    plot_trajectories(data, emphasize=0, mincolor=0.3)
    plot_maze(task)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.savefig("plots/"+task +".png",
               bbox_inches='tight',
               transparent=False,
               pad_inches=0,
               dpi=200)
    plt.savefig("plots/"+task +".pdf",
               bbox_inches='tight',
               transparent=False,
               pad_inches=0)

    plt.figure()
    plot_observations(data)
    plt.savefig("plots/"+ task +"_obs.png",
               bbox_inches='tight',
               transparent=False,
               pad_inches=0,
               dpi=200)

    plt.show()
"""
