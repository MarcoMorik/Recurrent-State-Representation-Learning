# Recurrent State Representation Learning with Robotic Priors #

Code of my Master thesis. [Manuscript in preperation]
## Description


In robotics, the sensory inputs are high dimensional, however, only a small subspace is important to guide the actions of a robot. Additionally, the observation space is often non Markovian. 
We extend idea of robotic priors to work on non Markovian observation spaces. For this, we train a recurrent neural network on trajectories, such that the network learns to encode past information in its hidden state. With this we are able to learn a Markovian state space. To train this network, we combine and modify existing robotic priors to work in non Markovian environments. We test our method in a 3D maze environment. To evaluate the quality of the learned state representation, we introduce a validation network that maps from the learned states to the ground truth. 
Using the new validation network, we show that the learned state space contains both positional and structural information of the environment. Furthermore, we use reinforcement learning and show that the learned state space is sufficient to solve a maze navigation task. 

Bellow, we show the colored maze and observations from the DeepmindLab and the 5 dimensional state representatin that we learned. The clustering of similar colored states indicates, that the state encodes the positional and topological information of the maze::

![Observations](plots/2trajObsCol.png?raw=true "Input")

![Output](plots/PositionalScatterMatrix.png?raw=true "Output")


The code is organized as follows:


**main.py** organizes the loading of data to train or validate a network.

**temporal_net.py** contains the recurrent neural network, with it's learn function and several help functions for plotting etc.

**validation_net.py** contains the neural network that validates the representation learned by the RNN both with the true position, as also with the tile position.

**reinforcement_agent.py** trains a RL agent with Double-DQN that uses the learned state representation as input.

**q_net.py** implements the Q-network for the RL agent.

**data_utils.py**  loads the dataset.

**generate_data.py** generates the dataset in the DeepmindLab.

**coloring.py** is used for the different colorings of the plots (orientational, positional and topological) . It also provides the topological classification for the poses.

**plotting.py** contains several functions for plotting, e.g. the maze, comparison of different models ...







## Installation

To generate the dataset and to run the reinforcement learning agent, the DeepmindLab is needed.

First, download and extract the initial release of the Deepmindlab https://github.com/deepmind/lab/releases/tag/release-2016-12-06.

Second, copy content of deepmindlab_modifications in the deepmindlab folder (we removed the posters in the navigation maps to obtain a non Markovian environment) and all python scripts in the deepmindlab/python folder.

Third, run the data generation script from a shell in the deepmind folder:   
 ``` bazel run: generate_data ```



## Usage 

#### To train a new model: ###


python main.py -t (-m model name) (-d number of representive states) ( -n  number of training sequences) (-l uses Landmark-Prior instead of Similar point) ... (-h for a list of all parameters)   

For example:

```python main.py -t -m first_model -d 5 -n 1000 -bs 45 -bn 30 -sl 25```   

This trains a new model with 5 States, using all 1000 Sequences of the data, having a batchsize of 45 with a seq_len of 25, 30 Batches each epoch  and at most 2000 epochs in total.

#### To validate an already trained model: ###

python main.py -e (-m a pretrained model) (-d number of representive states of the trained model)  (-plot to enable plotting)

For example:

```python main.py -e -m similar_point_model -d 5 -plot```

#### To run the RL agent using the learned state:

```bazel run: reinforcement_agent -- --srl PATH/TO/RSRL_FILE```


