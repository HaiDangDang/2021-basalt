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
from torch.autograd import Variable

from sklearn.cluster import KMeans

DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
DATA_SAMPLES = 1000000
max_last_action = 15
max_his = 20

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim, data_size=16, hidden_size=64):
        super().__init__()
        n_input_channels = input_shape[0]
        self.hidden_size = hidden_size
        self.layer_dim = 1
        self.hidden_dim = 100
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
            nn.Linear(data_size + 9 * max_last_action, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self.rnn_s = nn.LSTM(n_flatten + 512, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn = nn.LSTM(n_flatten + 512, self.hidden_dim, self.layer_dim, batch_first=True)

        # self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, observations: th.Tensor, data: th.Tensor) -> th.Tensor:
        batch_size, timesteps, C, H, W = observations.size()
        c_in = observations.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        data_out = self.linear_stack(data)
        r_in_cnn = c_out.view(batch_size, timesteps, -1)
        r_in_data = data_out.view(batch_size, timesteps, -1)
        r_in = torch.cat((r_in_cnn, r_in_data), dim=-1)

        h0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        out, (hn, cn) = self.rnn(r_in[:, :int(max_his / 2)], (h0.detach(), c0.detach()))
        # out = self.fc(out[:,-1,:])

        h0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        out_s, (hn, cn) = self.rnn_s(r_in[:, int(max_his / 2):], (h0.detach(), c0.detach()))

        out_in = torch.cat((out[:, -1, :], out_s[:, -1, :]), dim=-1)

        out = self.fc(out_in)

        return out

    def initHidden(self, BATCH):
        return (Variable(torch.zeros(self.layer_dim, BATCH, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.layer_dim, BATCH, self.hidden_dim)).cuda())


def process_inventory(obs, angle_1, angle_2):

    data = np.zeros(16)

    if obs['equipped_items']['mainhand']['type'] in ['carrot']:
        data[0] = 1
    if obs['equipped_items']['mainhand']['type'] in ['fence']:
        data[1] = 1
    if obs['equipped_items']['mainhand']['type'] in ['fence_gate']:
        data[2] = 1
    if obs['equipped_items']['mainhand']['type'] in ['snowball']:
        data[3] = 1
    if obs['equipped_items']['mainhand']['type'] in ['wheat']:
        data[4] = 1
    if obs['equipped_items']['mainhand']['type'] in ['wheat_seeds']:
        data[5] = 1
    if obs['equipped_items']['mainhand']['type'] in ['other', 'none', 'air']:
        data[6] = 1

    data[7] = np.clip(obs['inventory']['carrot'] / 1, 0, 1)
    data[8] = np.clip(obs['inventory']['fence'] / 64, 0, 1)
    data[9] = np.clip(obs['inventory']['fence_gate'] / 64, 0, 1)

    data[10] = np.clip(obs['inventory']['snowball'] / 1, 0, 1)
    data[11] = np.clip(obs['inventory']['wheat'] / 1, 0, 1)
    data[12] = np.clip(obs['inventory']['wheat_seeds'] / 1, 0, 1)

    # data[18] = t/18000
    data[13] = np.clip((angle_1 + 90) / 180, 0, 1)
    if angle_2 > 180:
        angle_2 = 360 - angle_2
        data[15] = 1
    data[14] = np.clip(angle_2/ 180, 0, 1)

    return data


#
# range_angle = np.array([-65.8, -20.2, -10.5, -5.1, -2.4, -0.8, 0.0, 1.0, 2.5, 5.1, 9.6, 19.0])
# range_angle_2 = np.array([-82.3, -35.5, -20.1, -10.2, -3.7, -0.0, 3.5, 9.6, 18.4, 30.2, 47.6, 103.0])
# range_angle = np.array([-10, -7, -3, -1, 0, 1, 3, 7, 10])
# range_angle_2 = np.array([-30, -15, -8, -5, 0, 5, 8, 15, 30])

range_angle = np.array([-30, -15, -10, -5, 0, 5, 10, 15, 30])
range_angle_2 = np.array([-35, -25, -15, -10, -5, 0, 5, 10, 15, 25, 35])

list_key = [ ['0','equip_carrot', 'equip_fence',
               'equip_fence_gate', 'equip_snowball', 'equip_wheat',
               'equip_wheat_seeds', 'a', 'use'], ['0', 'j_f', 'f', 'b', 'l', 'r'],range_angle_2, range_angle ]

to_action_0 = [[{}, {'equip': 'carrot'}, {'equip': 'fence'},
               {'equip': 'fence_gate'}, {'equip': 'snowball'}, {'equip': 'wheat'},
               {'equip': 'wheat_seeds'}, {'attack': 1}, {'use': 1}],
                [{}, {'forward': 1, 'jump': 1}, {'forward': 1},  {'back': 1},  {'left': 1},  {'right': 1}],
               range_angle_2,  range_angle]
list_key = np.array(list_key)

a1_0_Index = np.argmin(np.abs(range_angle - 0))
a2_0_Index = np.argmin(np.abs(range_angle_2 - 0))
range_angle_delta = np.max(range_angle) - np.min(range_angle)
range_angle_2_delta = np.max(range_angle_2) - np.min(range_angle_2)


def dataset_action_batch_to_actions(actions):
    action_with_key = actions
    dict_action = {}
    a1 = action_with_key['camera'][0]
    a2 = action_with_key['camera'][1]

    ks = [f'0', f'0',0, 0]

    a1 = np.abs(range_angle - a1)
    ks[3] = np.argmin(a1)

    a2 = np.abs(range_angle_2 - a2)
    ks[2] = np.argmin(a2)

    if action_with_key['jump'] == 1:
        dict_action['jump'] = 1
        dict_action['forward'] = 1
        ks[1] = f'j_f'
    elif action_with_key['forward'] == 1:
        dict_action['forward'] = 1
        ks[1] = f'f'
    elif action_with_key['back'] == 1:
        dict_action['back'] = 1
        ks[1] = f'b'
    elif action_with_key['left'] == 1:
        dict_action['left'] = 1
        ks[1] = f'l'
    elif action_with_key['right'] == 1:
        dict_action['right'] = 1
        ks[1] = f'r'

    ks[1] = np.where(np.array(list_key[1]) == ks[1])[0][0]

    if action_with_key['equip'] not in ['none', 'other' ]:
        equip = action_with_key['equip']
        k = f'equip_{equip}'
        dict_action['equip'] = equip
        ks[0] = k
    elif action_with_key['use'] == 1:
        dict_action['use'] = 1
        k = f'use'
        ks[0] = k
    elif action_with_key['attack'] == 1:
        dict_action['attack'] = 1
        k = f'a'
        ks[0] = k
    ks[0] = np.where(np.array(list_key[0]) == ks[0])[0][0]
    return ks

max_shape = 3
ENV_RUN = 'MineRLBasaltCreateVillageAnimalPen-v0'

index_1 = np.flip(np.arange(0, int(max_his / 2), 1))
index_2 = index_1 * 15
main_all_i = np.concatenate((index_1, index_2), axis=0)



def train():
    print("Prepare_Data")

    data = minerl.data.make(ENV_RUN,  data_dir='data', num_workers=4)
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    all_actions = []
    all_pov_obs = []
    all_last_action = []
    c_history = []
    all_index_x = []

    print("Loading data")


    counter = 0

    all_data_obs = [[], [], [], []]

    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

        angle_1 = 0
        angle_2 = 0

        stack_all_a_ = [np.zeros((1, 9)) for _ in range(max_last_action)]


        for obs, action, r, _, _ in trajectory:


            main_data = []
            final = obs['pov']
            key = dataset_action_batch_to_actions(action)
            after_proces = process_inventory(obs, angle_1, angle_2)

            main_data.append(after_proces)

            x = np.array([action['use'], action['attack']])
            after_proces = np.concatenate((after_proces, x), axis=0)
            main_data.append(after_proces)

            x = np.array([action['jump'], action['forward'], action['back']
                             , action['right'], action['left']])
            after_proces = np.concatenate((after_proces, x), axis=0)
            main_data.append(after_proces)

            after_proces = np.concatenate(
                (after_proces, [(range_angle_2[key[2]] + np.max(range_angle_2)) / range_angle_2_delta]), axis=0)
            main_data.append(after_proces)



            angle_2 += action['camera'][1]
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2

            angle_1 -= action['camera'][0]
            angle_1 = np.clip(angle_1, -90, 90)



            if not (key[3] == a1_0_Index and key[2] == a2_0_Index and key[0] == 0 and key[1] == 0):
                all_index_x.append(len(all_pov_obs))
            all_pov_obs.append(final)
            all_actions.append(key)
            all_last_action.append(np.ndarray.flatten(np.array(stack_all_a_)))
            c_history.append(counter)
            all_data_obs[0].append(main_data[0])
            all_data_obs[1].append(main_data[1])
            all_data_obs[2].append(main_data[2])
            all_data_obs[3].append(main_data[3])

            action['c2'] = (range_angle_2[key[2]] + np.min(range_angle_2) * -1) / range_angle_2_delta
            action['c1'] = (range_angle[key[3]] + np.min(range_angle) * -1) / range_angle_delta
            del action['camera']
            del action['equip']
            del action['sprint']
            del action['sneak']

            stack_all_a_.append(np.expand_dims(np.array(list(action.values())), axis=0))
            if len(stack_all_a_) > max_last_action:
                del stack_all_a_[0]

        counter += 1


    all_actions = np.array(all_actions)
    for i in range(len(all_data_obs)):
        all_data_obs[i] = np.array(all_data_obs[i])
    all_last_action = np.array(all_last_action)
    c_history = np.array(c_history)
    x = np.array(all_data_obs[0][:, 2], dtype=np.uint8)
    np.bincount(x)



    network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1]), all_data_obs[1].shape[1]).cuda()
    optimizer_1 = th.optim.Adam(network_1.parameters(), lr=LEARNING_RATE)
    loss_function_1 = nn.CrossEntropyLoss()

    network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2]), all_data_obs[2].shape[1]).cuda()
    optimizer_2 = th.optim.Adam(network_2.parameters(), lr=LEARNING_RATE)
    loss_function_2 = nn.CrossEntropyLoss()


    network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3]), all_data_obs[3].shape[1]).cuda()
    optimizer_3 = th.optim.Adam(network_3.parameters(), lr=LEARNING_RATE)
    loss_function_3 = nn.CrossEntropyLoss()

    all_index_x = np.array(all_index_x)
    num_samples = all_index_x.shape[0]

    update_count = 0
    losses = []

    print("Training")
    for index__ in range(10):
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):

            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]
            batch_indices = all_index_x[batch_indices]
            obs = np.zeros((len(batch_indices), max_his, 64, 64, max_shape), dtype=np.float32)
            l_a = np.zeros((len(batch_indices), max_his, all_last_action.shape[1]), dtype=np.float32)

            inven_1 = np.zeros((len(batch_indices), max_his, all_data_obs[1].shape[1]), dtype=np.float32)
            inven_2 = np.zeros((len(batch_indices), max_his, all_data_obs[2].shape[1]), dtype=np.float32)
            inven_3 = np.zeros((len(batch_indices), max_his, all_data_obs[3].shape[1]), dtype=np.float32)
            for j in range(len(batch_indices)):
                index = batch_indices[j]
                c_h = c_history[index]
                counter_i = 0
                for b in main_all_i:
                    n_index = index - b
                    if n_index >= 0:
                        o_h = c_history[n_index]
                        if o_h == c_h:
                            obs[j, counter_i] = all_pov_obs[n_index]
                            l_a[j, counter_i] = all_last_action[n_index]
                            inven_1[j, counter_i] = all_data_obs[1][n_index]
                            inven_2[j, counter_i] = all_data_obs[2][n_index]
                            inven_3[j, counter_i] = all_data_obs[3][n_index]

                    counter_i += 1
            obs = obs.transpose(0, 1, 4, 2, 3)
            obs = th.from_numpy(obs).float().cuda()

            obs /= 255.0


            inven_1 = np.concatenate((inven_1, l_a), axis=-1)
            inven_1 = th.from_numpy(inven_1).float().cuda()

            inven_2 = np.concatenate((inven_2, l_a), axis=-1)
            inven_2 = th.from_numpy(inven_2).float().cuda()

            inven_3 = np.concatenate((inven_3, l_a), axis=-1)
            inven_3 = th.from_numpy(inven_3).float().cuda()


            logits = network_1(obs, inven_1)
            actions = all_actions[batch_indices, 1]
            loss_1 = loss_function_1(logits, th.from_numpy(actions).long().cuda())
            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            logits = network_2(obs, inven_2)
            actions = all_actions[batch_indices, 2]
            loss_2 = loss_function_2(logits, th.from_numpy(actions).long().cuda())
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            logits = network_3(obs, inven_3)
            actions = all_actions[batch_indices, 3]
            loss_3 = loss_function_3(logits, th.from_numpy(actions).long().cuda())
            optimizer_3.zero_grad()
            loss_3.backward()
            optimizer_3.step()

            update_count += 1
            losses.append([ loss_1.item(), loss_2.item(), loss_3.item()])
            if (update_count % 1000) == 0:
                mean_loss = np.mean(losses, axis=0)
                tqdm.write("Iteration {}. Loss  {:<10.3f} {:<10.3f}  {:<10.3f}".format(
                    update_count, mean_loss[0], mean_loss[1], mean_loss[2]))
                losses.clear()


    network = NatureCNN((max_shape, 64, 64), len(list_key[0]), all_data_obs[0].shape[1]).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    tmp = all_actions[all_index_x, 0]
    index_0 = np.where(tmp == 0)[0]
    index_2 = np.where(tmp != 0)[0]

    index_0 = np.random.choice(index_0, 30000)
    x = np.concatenate((index_0, index_2))
    all_index_new = all_index_x
    num_samples = all_index_new.shape[0]

    print("Training")
    for index__ in range(10):
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):

            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]
            batch_indices = all_index_x[batch_indices]
            obs = np.zeros((len(batch_indices), max_his, 64, 64, max_shape), dtype=np.float32)
            l_a = np.zeros((len(batch_indices), max_his, all_last_action.shape[1]), dtype=np.float32)

            inven_0 = np.zeros((len(batch_indices), max_his, all_data_obs[0].shape[1]), dtype=np.float32)
            for j in range(len(batch_indices)):
                index = batch_indices[j]
                c_h = c_history[index]
                counter_i = 0
                for b in main_all_i:
                    n_index = index - b
                    if n_index >= 0:
                        o_h = c_history[n_index]
                        if o_h == c_h:
                            obs[j, counter_i] = all_pov_obs[n_index]
                            l_a[j, counter_i] = all_last_action[n_index]
                            inven_0[j, counter_i] = all_data_obs[0][n_index]
                    counter_i += 1
            obs = obs.transpose(0, 1, 4, 2, 3)
            obs = th.from_numpy(obs).float().cuda()

            obs /= 255.0

            inven_0 = np.concatenate((inven_0, l_a), axis=-1)
            inven_0 = th.from_numpy(inven_0).float().cuda()

            logits = network(obs, inven_0)
            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            actions = all_actions[batch_indices, 0]
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1
            losses.append([loss.item()])
            if (update_count % 1000) == 0:
                mean_loss = np.mean(losses, axis=0)
                tqdm.write("Iteration {}. Loss {:<10.3f}".format(
                    update_count, mean_loss[0]))
                losses.clear()



    th.save(network.state_dict(), 'pen/a_1_2_2.pth')
    th.save(network_1.state_dict(), 'pen/a_2_2.pth')
    th.save(network_2.state_dict(), 'pen/a_3_2.pth')
    th.save(network_3.state_dict(), 'pen/a_4_2.pth')
    del data




network = NatureCNN((max_shape, 64, 64), len(list_key[0])).cuda()
network.load_state_dict(th.load('pen/a_1_2.pth'))

network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1])).cuda()
network_1.load_state_dict(th.load('pen/a_2_2.pth'))

network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2])).cuda()
network_2.load_state_dict(th.load('pen/a_3_2.pth'))

network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3])).cuda()
network_3.load_state_dict(th.load('pen/a_4_2.pth'))


# env = gym.make('MineRLBasaltCreateVillageAnimalPenHighRes-v0')
env = gym.make('MineRLBasaltCreateVillageAnimalPen-v0')

env.seed(2)

rewards = []

for episode in range(20):
    obs = env.reset()


    total_reward = 0

    done = False
    steps = 0
    # BC part to get some logs:

    angle_1 = 0
    angle_2 = 0

    last_attack = 0
    time_add = 0
    a2 = 0
    all_obs = []
    action_before = []
    last_action = None

    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    cv2.moveWindow('image', -30, 30)
    time_wait_equi = 0

    jump = False
    attack = False
    time_now = 0

    stack_all = []
    stack_all_data = []
    stack_all_data_1 = []
    stack_all_data_2 = []
    stack_all_data_3 = []

    stack_all_last = [np.zeros((1, max_last_action * 9))]
    stack_all_a_ = [np.zeros((1, 9)) for _ in range(max_last_action)]

    for i in range(18000):
        new_a = env.action_space.noop()

        final = obs['pov']
        final = np.expand_dims(final, axis=0)
        stack_all.append(final)

        final = np.zeros((max_his, 64, 64, max_shape), dtype=np.float32)
        l_a = np.zeros((max_his,  9 * max_last_action), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                final[counter_i] = stack_all[n_index]
                l_a[counter_i] = stack_all_last[n_index]
            counter_i += 1


        final = final.transpose(0, 3, 1, 2).astype(np.float32)
        final = th.from_numpy(final[None]).float().cuda()
        final /= 255.0


        # -----#
        inven_m = process_inventory(obs, angle_1, angle_2)

        # -----#
        # inven = np.expand_dims(inven_m, axis=0)
        stack_all_data.append(inven_m)

        inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all_data) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                inven[counter_i] = stack_all_data[n_index]
            counter_i += 1
        inven = np.concatenate((inven, l_a), axis=-1)

        inven = th.from_numpy(inven[None]).float().cuda()
        p = network(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[0])), p=probabilities)
        for k, v in to_action_0[0][action].items():
            new_a[k] = v

        # -----#

        # -----#
        x = np.array([new_a['use'], new_a['attack']])
        inven_m = np.concatenate((inven_m, x), axis=0)

        stack_all_data_1.append(inven_m)

        inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all_data_1) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                inven[counter_i] = stack_all_data_1[n_index]
            counter_i += 1
        inven = np.concatenate((inven,l_a), axis=-1)

        inven = th.from_numpy(inven[None]).float().cuda()
        p = network_1(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[1])), p=probabilities)
        for k, v in to_action_0[1][action].items():
            new_a[k] = v
        # -----#

        # -----#

        x = np.array([new_a['jump'], new_a['forward'], new_a['back']
                         , new_a['right'], new_a['left']])

        inven_m = np.concatenate((inven_m, x), axis=0)



        stack_all_data_2.append(inven_m)
        inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all_data_2) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                inven[counter_i] = stack_all_data_2[n_index]
            counter_i += 1
        inven = np.concatenate((inven, l_a), axis=-1)


        inven = th.from_numpy(inven[None]).float().cuda()
        p = network_2(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        new_a['camera'][1] = to_action_0[2][action]

        # -----#


        # -----#
        inven_m = np.concatenate((inven_m, [(new_a['camera'][1] + np.max(range_angle_2))
                                            / range_angle_2_delta]), axis=0)

        stack_all_data_3.append(inven_m)
        inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all_data_3) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                inven[counter_i] = stack_all_data_3[n_index]
            counter_i += 1
        inven = np.concatenate((inven, l_a), axis=-1)


        inven = th.from_numpy(inven[None]).float().cuda()
        p = network_3(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        a1 = to_action_0[3][action]
        if angle_1 - a1 < -70:
            a1 = -70 - angle_1
            if -70 - angle_1 < 0:
                a1 = 0
        elif angle_1 - a1 > 40:
            a1 = 40 - angle_1
            if 40 - angle_1 > 0:
                a1 = 0
        new_a['camera'][0] = a1


        # -----#
        angle_1 -= new_a['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += new_a['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        obs, reward, done, info = env.step(new_a)
        if new_a['use'] == 1:
            print(obs['equipped_items']['mainhand']['type'])
        last_action = new_a
        new_a['c1'] = (new_a['camera'][0] + np.min(range_angle) * -1) / range_angle_delta
        new_a['c2'] = (new_a['camera'][1] + np.min(range_angle_2) * -1) / range_angle_2_delta
        del new_a['camera']
        del new_a['equip']
        del new_a['sprint']
        del new_a['sneak']
        tmp = np.expand_dims(np.array(list(new_a.values())), axis=0)
        stack_all_a_.append(tmp)
        if len(stack_all_a_) > max_last_action:
            del stack_all_a_[0]

        stack_all_last.append(np.ndarray.flatten(np.array(stack_all_a_))[None])

        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
        total_reward += reward
        steps += 1
        if done:
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        # time.sleep(0.01)
    cv2.destroyAllWindows()
    print(obs['equipped_items'])


img = plt.imshow(obs['pov'])
plt.show()
x = final.cpu().numpy()

a = x[0, 17,:]
a = a.transpose(1, 2, 0)
img = plt.imshow(a)
plt.show()
not random