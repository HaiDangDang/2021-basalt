import numpy as np
import torch as th
from torch import nn
import gym
import minerl
from tqdm.notebook import tqdm
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display
import logging
import cv2
import pickle
import torch
import random

from torch import nn
import gym
import minerl
from tqdm.notebook import tqdm
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display
import logging
import cv2
import pickle
import torch
import time
import math
import torchvision
from collections import Counter

from hummingbird.ml import convert
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans


DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 64
DATA_SAMPLES = 1000000

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *input_shape)).shape[1]

        self.linear_stack = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # self.flat_cnn = nn.Sequential(
        #     nn.Linear(n_flatten, 512),
        #     nn.ReLU(),
        # )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor, data: th.Tensor) -> th.Tensor:
        x1 = self.cnn(observations)
        x2 = self.linear_stack(data)
        x = torch.cat((x1, x2), dim=1)

        return self.linear(x)





def process_inventory(obs, angle_1, angle_2, angle_2_f, last_action):

    data = np.zeros(7)

    if obs['equipped_items']['mainhand']['type'] in ['snowball']:
        data[0] = 1
    if obs['equipped_items']['mainhand']['type'] in ['other', 'none', 'air']:
        data[1] = 1

    jumping = 0
    if last_action is not None:
        jumping = last_action['jump']
    data[2] = jumping

    data[3] = np.clip(obs['inventory']['snowball'] / 1, 0, 1)
    data[4] = np.clip((angle_1[-1] + 90) / 180, 0, 1)
    data[5] = 0
    angle_2 = angle_2[-1]
    if angle_2 > 180:
        angle_2 = 360 - angle_2
        data[5] = 1
    data[6] = np.clip(np.abs(angle_2) / 180, 0, 1)

    data_f = copy.deepcopy(data)
    data_f[5] = 0
    angle_2_f = angle_2_f[-1]
    if angle_2_f > 180:
        angle_2_f = 360 - angle_2_f
        data_f[5] = 1

    data_f[6] = np.clip((angle_2_f) / 180, 0, 1)

    # angle_1 = np.clip((np.array(angle_1) + 90) / 180, 0, 1)
    # data = np.concatenate((data, angle_1), axis=0)
    #
    #
    # angle_2 = np.clip(np.array(angle_2) / 360, 0, 1)
    # angle_2_f =np.clip(np.array(angle_2_f) / 360, 0, 1)
    #
    # data_f = copy.deepcopy(data)
    #
    # data = np.concatenate((data, angle_2), axis=0)
    # data_f = np.concatenate((data_f, angle_2_f), axis=0)

    return data, data_f


# range_angle = np.array([0, -15, -10, -7, -5, -3, -1, -0.5, 0.5, 1, 3, 5, 7, 10, 15])
#
# range_angle_2 = np.array([0, -25, -15, -8, -5, -3,-1, 1, 3, 5, 8, 15, 25])

range_angle = np.array([ -20.4, -10.6, -5.9, 0, 5.2, 8.8, 15.3, 32.5])

range_angle_2 = np.array([-55.0, -31.4, -19.5, -11.5, -6.0, 0, 7.5, 14.4, 24.9, 42.6])

range_angle = np.array([-20.4, -10.6, -5.9, -2.8, -1.0,0, 1.0, 2.7, 5.2, 8.8, 15.3, 32.5])

range_angle_2 = np.array([-55.0, -31.4, -19.5, -11.5, -6.0, -2.2,0, 2.8, 7.5, 14.4, 24.9, 42.6])

list_key = [range_angle, range_angle_2,
            ['0', 'j_f', 'f', 'b', 'l', 'r' ], ['0', 'equip_snowball', 'a', 'use']]
list_key = np.array(list_key)
to_action_0 = [range_angle, range_angle_2,
                [{}, {'forward': 1, 'jump': 1}, {'forward': 1},  {'back': 1},  {'left': 1},  {'right': 1}],
               [{}, {'equip': 'snowball'}, {'attack': 1}, {'use': 1}]]
delta_a = 5

def dataset_action_batch_to_actions(actions):
    action_with_key = actions
    dict_action = {}
    a1 = action_with_key['camera'][0]
    a2 = action_with_key['camera'][1]

    ks = [0, 0, f'0', f'0']

    a1 = np.abs(range_angle - a1)
    ks[0] = np.argmin(a1)

    a2 = np.abs(range_angle_2 - a2)
    ks[1] =np.argmin(a2)

    if action_with_key['jump'] == 1:
        dict_action['jump'] = 1
        dict_action['forward'] = 1
        ks[2] = f'j_f'
    elif action_with_key['forward'] == 1:
        dict_action['forward'] = 1
        ks[2] = f'f'
    elif action_with_key['back'] == 1:
        dict_action['back'] = 1
        ks[2] = f'b'
    elif action_with_key['left'] == 1:
        dict_action['left'] = 1
        ks[2] = f'l'
    elif action_with_key['right'] == 1:
        dict_action['right'] = 1
        ks[2] = f'r'

    ks[2] = np.where(np.array(list_key[2]) == ks[2])[0][0]


    if action_with_key['equip'] not in ['none', 'other' ]:
        equip = action_with_key['equip']
        k = f'equip_{equip}'
        dict_action['equip'] = equip
        ks[3] = k

    elif action_with_key['use'] == 1:
        dict_action['use'] = 1
        k = f'use'
        ks[3] = k
    elif action_with_key['attack'] == 1:
        dict_action['attack'] = 1
        k = f'a'
        ks[3] = k
    ks[3] = np.where(np.array(list_key[3]) == ks[3])[0][0]

    ks_f = copy.deepcopy(ks)
    a2 = action_with_key['camera'][1] * -1
    a2 = np.abs(range_angle_2 - a2)
    ks_f[1] = np.argmin(a2)

    return ks, ks_f

max_shape = 3

list_angle_grey = [0, 45, 90, 135, 180, 225, 270, 315, 360]
list_obs = [-90,-45,45,90]

# x = np.array(a1_a).reshape(-1, 1)
# kmeans = KMeans(n_clusters=12)
# kmeans.fit(x)
# action_centroids = kmeans.cluster_centers_
# x = action_centroids.astype(np.float32)
# x = np.around(x, decimals=1)
# x = np.ndarray.flatten(x)
# x = sorted(x)

def train():
    print("Prepare_Data")

    data = minerl.data.make("MineRLBasaltFindCave-v0",  data_dir='data', num_workers=4)

    all_actions = []
    all_pov_obs = []
    all_data_obs = []
    all_visibale_obs = []
    stack_angle_1 = []
    print("Loading data")
    trajectory_names = data.get_trajectory_names()
    counter = 0
    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    a1_a = []
    a2_a = []
    # np.sum((np.abs(a1_a) > 0))
    # np.sum((np.array(a1_a) > 0))
    #
    # np.sum((np.array(a1_a) == 0) & (np.array(a2_a) == 0))
    # np.sum((np.abs(a1_a) > 5) & (np.abs(a1_a) < 10))
    #
    #
    # np.sum((np.abs(a2_a) > 0))
    # np.sum((np.array(a2_a) < 0))
    #
    # np.sum((np.array(a2_a) == 0) & (np.array(a2_a) == 0))
    # np.sum((np.abs(a2_a) > 30) & (np.abs(a2_a) < 45))
    # np.histogram(np.abs(a1))

    counter_all = 0
    while True:
        if counter == len(trajectory_names):
            break

        trajectory_name = trajectory_names[counter]
        # trajectory_name = 'v3_accomplished_pattypan_squash_ghost-6_44486-45290'
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        time = 0
        angle_1 = 0
        angle_2 = 0
        angle_2_f = 0
        last_action = None
        stack_a_2 = [0 for _ in range(20)]
        stack_a_2_f = [0 for _ in range(20)]
        stack_a_1 = [0 for _ in range(20)]

        stack_grey = {x: np.zeros((64,64)) for x in list_angle_grey}
        stack_grey_f = {x: np.zeros((64,64)) for x in list_angle_grey}

        for obs, action, r, _, _ in trajectory:
            # a1_a.append(action['camera'][0])
            # a2_a.append(action['camera'][1])
            final = obs['pov']
            final_f = np.fliplr(obs['pov'])

            stack_a_1.append(angle_1)
            stack_a_2.append(angle_2)
            stack_a_2_f.append(angle_2_f)
            if len(stack_a_2) > 20:
                del stack_a_1[0]
                del stack_a_2[0]
                del stack_a_2_f[0]

            key, key_f = dataset_action_batch_to_actions(action)
            after_proces, after_proces_f = process_inventory(obs, stack_a_1, stack_a_2, stack_a_2_f, last_action)
            last_action = action

            # grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])
            #
            # angle_2_fix = (np.round(int(angle_2) / 45) * 45)
            # stack_grey[angle_2_fix] = grey
            #
            # grey_f = np.dot(final_f[..., :3], [0.2989, 0.5870, 0.1140])
            # angle_2_fix = (np.round(int(angle_2_f) / 45) * 45)
            # stack_grey_f[angle_2_fix] = grey_f
            #
            # for location in list_obs:
            #     tmp = angle_2 + location
            #     if tmp > 360:
            #         tmp = tmp - 360
            #     elif tmp < 0:
            #         tmp = 360 + tmp
            #     tmp = np.round(int(tmp) / 45) * 45
            #     final = np.concatenate((final, stack_grey[tmp][:, :, None]), axis=-1)
            #
            #     tmp = angle_2_f + location
            #     if tmp > 360:
            #         tmp = tmp - 360
            #     elif tmp < 0:
            #         tmp = 360 + tmp
            #     tmp = np.round(int(tmp) / 45) * 45
            #     final_f = np.concatenate((final_f, stack_grey_f[tmp][:, :, None]), axis=-1)
            # final = final.astype(np.uint8)
            # final_f = final_f.astype(np.uint8)

            angle_2 += action['camera'][1]
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2

            angle_2_f += action['camera'][1] *-1
            if angle_2_f > 360:
                angle_2_f = angle_2_f - 360
            elif angle_2_f < 0:
                angle_2_f = 360 + angle_2_f

            angle_1 -= action['camera'][0]
            angle_1 = np.clip(angle_1, -90, 90)

            time += 1
            all_pov_obs.append(final)
            all_actions.append(key)
            all_data_obs.append(after_proces)

            # all_pov_obs.append(final_f)
            # all_actions.append(key_f)
            # all_data_obs.append(after_proces_f)

            a1_a.append(angle_1)
            a1_a.append(angle_1)

        delta_down = len(all_data_obs)
        for j in range(200):
            new_index = delta_down - j -1
            if new_index >= 0 and new_index >= counter_all:
                if j % 2 == 0:
                    all_visibale_obs.append(all_pov_obs[new_index])
                    stack_angle_1.append(a1_a[new_index])
                all_pov_obs.append(all_pov_obs[new_index])
                all_actions.append(all_actions[new_index])
                all_data_obs.append(all_data_obs[new_index])
                a1_a.append(a1_a[new_index])

        counter_all = len(all_pov_obs)
        counter += 1

    # img = plt.imshow(final[:,:,4])
    # plt.show()
    final_f.shape




    all_actions = np.array(all_actions)
    # all_pov_obs = np.array(all_pov_obs)
    all_data_obs = np.array(all_data_obs)
    all_data_obs.shape
    all_actions.shape
    network = NatureCNN((max_shape, 64, 64), len(list_key[0])).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1])).cuda()
    optimizer_1 = th.optim.Adam(network_1.parameters(), lr=LEARNING_RATE)
    loss_function_1 = nn.CrossEntropyLoss()

    network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2])).cuda()
    optimizer_2 = th.optim.Adam(network_2.parameters(), lr=LEARNING_RATE)
    loss_function_2 = nn.CrossEntropyLoss()

    network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3])).cuda()
    optimizer_3 = th.optim.Adam(network_3.parameters(), lr=LEARNING_RATE)
    loss_function_3 = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []

    print("Training")
    for _ in range(14):
        # Randomize the order in which we go over the samples
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):
            # break
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            obs = np.zeros((len(batch_indices), 64, 64, max_shape), dtype=np.float32)
            for j in range(len(batch_indices)):
                obs[j] = all_pov_obs[batch_indices[j]]
            # Load the inputs and preprocess
            # obs = all_pov_obs[batch_indices].astype(np.float32)
            # Transpose observations to be channel-first (BCHW instead of BHWC)
            obs = obs.transpose(0, 3, 1, 2)

            obs = th.from_numpy(obs).float().cuda()

            # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            obs /= 255.0

            # Map actions to their closest centroids
            # actions = all_actions[batch_indices]

            # distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
            # actions = np.argmin(distances, axis=0)
            # Obtain logits of each action
            inven = th.from_numpy(all_data_obs[batch_indices]).float().cuda()
            logits = network(obs, inven)
            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            actions = all_actions[batch_indices, 0]
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            logits = network_1(obs, inven)
            actions = all_actions[batch_indices, 1]
            loss_1 = loss_function_1(logits, th.from_numpy(actions).long().cuda())
            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            logits = network_2(obs, inven)
            actions = all_actions[batch_indices, 2]
            loss_2 = loss_function_2(logits, th.from_numpy(actions).long().cuda())
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            logits = network_3(obs, inven)
            actions = all_actions[batch_indices, 3]
            loss_3 = loss_function_3(logits, th.from_numpy(actions).long().cuda())
            optimizer_3.zero_grad()
            loss_3.backward()
            optimizer_3.step()

            update_count += 1
            losses.append([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
            if (update_count % 1000) == 0:
                mean_loss = np.mean(losses, axis=0)
                tqdm.write("Iteration {}. Loss {:<10.3f} {:<10.3f} {:<10.3f}  {:<10.3f}".format(
                    update_count, mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]))
                losses.clear()


    th.save(network.state_dict(), 'cave/a_13.pth')
    th.save(network_1.state_dict(), 'cave/a_23.pth')
    th.save(network_2.state_dict(), 'cave/a_33.pth')
    th.save(network_3.state_dict(), 'cave/a_43.pth')
    del data




network = NatureCNN((max_shape, 64, 64), len(list_key[0])).cuda()
network.load_state_dict(th.load('cave/a_13.pth'))

network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1])).cuda()
network_1.load_state_dict(th.load('cave/a_23.pth'))

network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2])).cuda()
network_2.load_state_dict(th.load('cave/a_33.pth'))

network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3])).cuda()
network_3.load_state_dict(th.load('cave/a_43.pth'))


env = gym.make('MineRLBasaltFindCaveHighRes-v0')


rewards = []
# all_visibale_obs = []
for episode in range(20):
    obs = env.reset()
    total_reward = 0

    done = False
    steps = 0
    # BC part to get some logs:

    angle_1 = 0
    angle_2 = 0

    time_add = 0
    all_obs = []


    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    cv2.moveWindow('image', -30, 30)

    last_action = None
    stack_a_2 = [0 for _ in range(20)]
    stack_a_1 = [0 for _ in range(20)]
    stack_grey = {x: np.zeros((64, 64)) for x in list_angle_grey}

    for i in range(1700):
        final = cv2.resize(obs['pov'],
                          (64,64),
                           interpolation=cv2.INTER_AREA)
        grey = np.dot(final[..., :3], [0.2989, 0.5870, 0.1140])

        angle_2_fix = (np.round(int(angle_2) / 45) * 45)
        stack_grey[angle_2_fix] = grey


        for location in list_obs:
            tmp = angle_2 + location
            if tmp > 360:
                tmp = tmp - 360
            elif tmp < 0:
                tmp = 360 + tmp
            tmp = np.round(int(tmp) / 45) * 45
            # final = np.concatenate((final, stack_grey[tmp][:, :, None]), axis=-1)
        final = final.astype(np.uint8)

        final = final.transpose(2, 0, 1).astype(np.float32)
        final = th.from_numpy(final[None]).float().cuda()
        final /= 255.0

        stack_a_1.append(angle_1)
        stack_a_2.append(angle_2)
        if len(stack_a_2) > 20:
            del stack_a_1[0]
            del stack_a_2[0]

        inven,_ = process_inventory(obs, stack_a_1, stack_a_2, [1], last_action)
        inven = th.from_numpy(inven[None]).float().cuda()

        new_a = env.action_space.noop()
        # if time_now == 0:
        #     probabilities = th.softmax(network(final, inven), dim=1)[0]
        #     probabilities = probabilities.detach().cpu().numpy()
        #     action = np.random.choice(np.arange(len(to_action_0[0])), p=probabilities)
        #
        #     new_a['camera'][0] =to_action_0[0][action]
        #     probabilities = th.softmax(network_1(final, inven), dim=1)[0]
        #     probabilities = probabilities.detach().cpu().numpy()
        #     action = np.random.choice(np.arange(len(to_action_0[1])), p=probabilities)
        #     new_a['camera'][1] = to_action_0[1][action]
        #
        #
        #     if jump:
        #         probabilities = th.softmax(network_2(final, inven), dim=1)[0]
        #         probabilities = probabilities.detach().cpu().numpy()
        #         action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        #         for k, v in to_action_0[2][action].items():
        #             new_a[k] = v
        #     if attack:
        #         probabilities = th.softmax(network_3(final, inven), dim=1)[0]
        #         probabilities = probabilities.detach().cpu().numpy()
        #         action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        #         for k, v in to_action_0[3][action].items():
        #             new_a[k] = v
        #     obs, reward, done, info = env.step(new_a)
        #
        #     time_now = 1
        # elif time_now == 1:
        #     probabilities = th.softmax(network_2(final, inven), dim=1)[0]
        #     probabilities = probabilities.detach().cpu().numpy()
        #     action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        #     for k,v in to_action_0[2][action].items():
        #         new_a[k] = v
        #     if attack:
        #         probabilities = th.softmax(network_3(final, inven), dim=1)[0]
        #         probabilities = probabilities.detach().cpu().numpy()
        #         action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        #         for k, v in to_action_0[3][action].items():
        #             new_a[k] = v
        #     obs, reward, done, info = env.step(new_a)
        #
        #     time_now = 2
        # elif time_now == 2:
        #     if jump:
        #         probabilities = th.softmax(network_2(final, inven), dim=1)[0]
        #         probabilities = probabilities.detach().cpu().numpy()
        #         action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        #         for k, v in to_action_0[2][action].items():
        #             new_a[k] = v
        #
        #     probabilities = th.softmax(network_3(final, inven), dim=1)[0]
        #     probabilities = probabilities.detach().cpu().numpy()
        #     action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        #     for k,v in to_action_0[3][action].items():
        #         new_a[k] = v
        #     obs, reward, done, info = env.step(new_a)
        #
        #     time_now = 0
        # jump = bool(new_a['jump'])
        # attack = bool(new_a['attack'])
        # time_add += 1
        probabilities = th.softmax(network(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[0])), p=probabilities)

        new_a['camera'][0] = to_action_0[0][action]
        probabilities = th.softmax(network_1(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[1])), p=probabilities)
        new_a['camera'][1] = to_action_0[1][action]

        probabilities = th.softmax(network_2(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        for k, v in to_action_0[2][action].items():
            new_a[k] = v

        probabilities = th.softmax(network_3(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        for k, v in to_action_0[3][action].items():
            new_a[k] = v

        obs, reward, done, info = env.step(new_a)

        angle_1 -= new_a['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += new_a['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        print(new_a['camera'])
        obs, reward, done, info = env.step(new_a)
        last_action = new_a
        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
        total_reward += reward
        steps += 1
        if done:
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        # time.sleep(0.05)
    cv2.destroyAllWindows()
    print(obs['equipped_items'])



import time
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
for i in range(4000,len(all_visibale_obs)):
    image = all_visibale_obs[len(all_visibale_obs) - i - 1]
    print(stack_angle_1[len(all_visibale_obs) - i - 1])

    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 1000, 1000)
    time.sleep(0.05)
    if cv2.waitKey(10) & 0xFF == ord('o'):
        break

counter = 0
all_pov = []
all_action = []

while True:
    if counter == len(trajectory_names):
        break

    trajectory_name = trajectory_names[counter]
    # trajectory_name = 'v3_accomplished_pattypan_squash_ghost-6_44486-45290'
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    angle_1 = 0
    angle_2 = 0

    for obs, action, r, _, _ in trajectory:
        all_pov.append(obs['pov'])
        all_action.append(action)

    counter += 1

cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
for i in range(len(all_pov)):
    action = all_action[i]
    cv2.imshow('image', cv2.cvtColor(all_pov[i], cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 1000, 1000)
    time.sleep(0.15)
    if cv2.waitKey(10) & 0xFF == ord('o'):
        break
    print(action['camera'][1])


    angle_1 += action['camera'][0]
    angle_1 = np.clip(angle_1, -90, 90)

    angle_2 += action['camera'][1]
    if angle_2 > 360:
        angle_2 = angle_2 - 360
    elif angle_2 < 0:
        angle_2 = 360 + angle_2
