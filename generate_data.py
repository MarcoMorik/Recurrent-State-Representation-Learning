import reinforcement_agent
from multiprocessing import Pool
import argparse
import os
import deepmind_lab
import numpy as np

def slice_log(log, length, steps):
    len_episode = len(episode_log['vel'])
    if len_episode >= length:
        start = np.random.randint(0, len_episode - length - 1)
        for key in log.iterkeys():
            log[key].extend(episode_log[key][start:start + length:steps])

if __name__ == '__main__':
    if not os.path.exists("../maze_data/"):
        os.makedirs("../maze_data")
    parser = argparse.ArgumentParser(usage="Generate Datasets with the Deepmind Lab")
    parser.add_argument("--maze", type=int, default=1,
                        help="Specify the environment. The small maze is 1, medium 2 and large 3")
    parser.add_argument("--type", type=str, default="mix",
                        help="Specify the dataset type, short, long or mix")

    args = parser.parse_args()
    assert args.maze in [1, 2, 3], "Please specify a valid maze (1,2,3)"
    assert args.type in ["mix","short","long"], "Please specify a dataset tyoe maze (short,long,mix)"
    for t in ["test","train"]:
        print("Start to collect the " + t + " dataset")
        #Starting Environment
        env = deepmind_lab.Lab(
            "nav_maze_static_0{}".format(args.maze), ['RGBD_INTERLACED', 'VEL.ROT', 'VEL.TRANS', 'POSE'],
            config={
                'fps': str(10),
                'width': str(32),
                'height': str(32)
            })
        #Initializing Reinforcement Agent that only generates Data with the Wanderer
        with reinforcement_agent.Reinforcement_Agent(use_srl=False) as agent:
            log = {'rgbd': [], 'vel': [], 'pose': []}
            for episode in xrange(500):
                episode_log = {'rgbd': [], 'vel': [], 'pose': []}
                env.reset()
                agent.reset()
                steps = 1000
                short = False
                for i in xrange(episode_length):
                    #Getting Observation
                    obs = env.observations()
                    image = np.reshape(obs['RGBD_INTERLACED'], 4096)
                    #Performing Wanderer Action
                    action = agent.wanderer_step(None, image, None)
                    _ = env.step(action, num_steps=1)

                    episode_log['rgbd'].append(obs['RGBD_INTERLACED'])
                    episode_log['vel'].append(
                        np.array([obs['VEL.TRANS'][0], obs['VEL.TRANS'][1], obs['VEL.ROT'][1]]))
                    episode_log['pose'].append(obs['POSE'])

                if args.type == "short":
                    slice_log(log, 200, 1)
                elif args.type == "long":
                    slice_log(log, 1200, 6)
                elif args.type == "mix":
                    if short:
                        slice_log(log, 200, 1)
                    else:
                        slice_log(log, 1200, 6)
                    short = not short

        np.savez("../maze_data/nav0{}".format(args.maze) + str(args.type)+"_"+t, rgbd=log['rgbd'],
                     vel=log['vel'],
                     pose=log['pose'])

