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

with open('trajectory_name_s.pkl', 'rb') as f:
    trajectory_name_s = pickle.load(f)
DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 32
DATA_SAMPLES = 1000000
ENV_RUN = 'MineRLBasaltMakeWaterfall-v0'
class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim, none_dim):
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
            nn.Linear(none_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        # self.flat_cnn = nn.Sequential(
        #     nn.Linear(n_flatten, 512),
        #     nn.ReLU(),
        # )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor, data: th.Tensor) -> th.Tensor:
        x1 = self.cnn(observations)
        x2 = self.linear_stack(data)
        x = torch.cat((x1, x2), dim=1)

        return self.linear(x)




def process_inventory(obs, angle_1, angle_2, angle_2_f, last_action):

    data = np.zeros(16)

    if obs['equipped_items']['mainhand']['type'] in ['cobblestone']:
        data[0] = 1
    if obs['equipped_items']['mainhand']['type'] in ['water_bucket']:
        data[1] = 1
    if obs['equipped_items']['mainhand']['type'] in ['bucket']:
        data[2] = 1
    if obs['equipped_items']['mainhand']['type'] in ['stone_pickaxe']:
        data[3] = 1
    if obs['equipped_items']['mainhand']['type'] in ['stone_shovel']:
        data[4] = 1
    if obs['equipped_items']['mainhand']['type'] in ['snowball']:
        data[5] = 1
    if obs['equipped_items']['mainhand']['type'] in ['other', 'none', 'air']:
        data[6] = 1

    data[7] = np.clip(obs['inventory']['cobblestone'] / 20, 0, 1)
    data[8] = np.clip(obs['inventory']['water_bucket'] / 1, 0, 1)
    data[10] = np.clip(obs['inventory']['bucket'] / 1, 0, 1)

    data[11] = np.clip((angle_1 + 90) / 180, 0, 1)

    if angle_2 > 180:
        angle_2 = 360 - angle_2
        data[13] = 1
    data[12] = np.clip(angle_2/ 180, 0, 1)

    jumping = 0
    attack = 0
    if last_action is not None:
        jumping = last_action['jump']
        attack = last_action['attack']
    data[14] = jumping
    data[15] = attack

    data_f = copy.deepcopy(data)
    data_f[13] = 0
    if angle_2_f > 180:
        angle_2_f = 360 - angle_2_f
        data_f[14] = 1
    data_f[12] = np.clip(angle_2_f/180, 0, 1)

    # print(angle_2, angle_2_f)
    return data, data_f



# range_angle = np.array([0, -15, -10, -7, -5, -3, -1, -0.5, 0.5, 1, 3, 5, 7, 10, 15])
#
# range_angle_2 = np.array([0, -25, -15, -8, -5, -3,-1, 1, 3, 5, 8, 15, 25])


# np.bincount(all_actions[:,0])
range_angle = np.array([-30, -20, -15,  0,  15, 20, 30])
range_angle_2 = np.array([-30, -20, -15,  0,  15, 20, 30])
list_key = [range_angle, range_angle_2,
            ['0', 'j_f', 'f', 'b', 'l', 'r'], ['0', 'equip_cobblestone', 'equip_snowball',
               'equip_stone_pickaxe', 'equip_stone_shovel', 'equip_water_bucket',
               'equip_bucket', 'a', 'use']]

to_action_0 = [range_angle, range_angle_2,
                [{}, {'forward': 1, 'jump': 1}, {'forward': 1},  {'back': 1},  {'left': 1},  {'right': 1}],
               [{}, {'equip': 'cobblestone'}, {'equip': 'snowball'},
               {'equip': 'stone_pickaxe'}, {'equip': 'stone_shovel'}, {'equip': 'water_bucket'},
               {'equip': 'bucket'}, {'attack': 1}, {'use': 1}]]
list_key = np.array(list_key)

delta_a = 5

def dataset_action_batch_to_actions(actions):

    action_with_key = actions
    ks = [0, 0, f'0', f'0']

    a1 =  action_with_key['camera'][0]
    a1 = np.abs(range_angle - a1)
    ks[0] = np.argmin(a1)

    a2 = action_with_key['camera'][1]
    a2 = np.abs(range_angle_2 - a2)
    ks[1] = np.argmin(a2)

    if action_with_key['jump'] == 1:
        ks[2] = f'j_f'
    elif action_with_key['forward'] == 1:
        ks[2] = f'f'
    elif action_with_key['back'] == 1:
        ks[2] = f'b'
    elif action_with_key['left'] == 1:
        ks[2] = f'l'
    elif action_with_key['right'] == 1:
        ks[2] = f'r'

    ks[2] = np.where(np.array(list_key[2]) == ks[2])[0][0]

    if action_with_key['equip'] not in ['none', 'other']:
        equip = action_with_key['equip']
        k = f'equip_{equip}'
        ks[3] = k
    elif action_with_key['use'] == 1:
        k = f'use'
        ks[3] = k
    elif action_with_key['attack'] == 1:
        k = f'a'
        ks[3] = k
    if len(np.where(np.array(list_key[3]) == ks[3])[0]) == 0:
        ks[3] = '0'
    ks[3] = np.where(np.array(list_key[3]) == ks[3])[0][0]

    ks_f = copy.deepcopy(ks)
    a2 = copy.deepcopy(action_with_key['camera'][1]) * -1
    a2 = np.abs(range_angle_2 - a2)
    ks_f[1] = np.argmin(a2)

    return ks, ks_f

max_shape = 3
def train():
    print("Prepare_Data")

    data = minerl.data.make(ENV_RUN,  data_dir='data', num_workers=4)
    trajectory_names = data.get_trajectory_names()

    all_actions = []
    all_pov_obs = []
    all_data_obs = [[], [], [], []]

    print("Loading data")

    a1_0_Index = np.argmin(np.abs(range_angle - 0))
    a2_0_Index = np.argmin(np.abs(range_angle_2 - 0))
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

        angle_1 = 0
        angle_2 = 0
        angle_2_f = 0
        last_action = None

        counter_i_a1 = []
        counter_i_a2 = []
        counter_i = 0
        a1 = []
        a2 = []
        stack_all_ss = {'action': [], "obs": []}

        for obs, action, r, _, _ in trajectory:

            stack_all_ss['action'].append(action)
            stack_all_ss['obs'].append(obs)

            if 15 > np.abs(action['camera'][0]) > 0:
                a1.append(action['camera'][0])
                counter_i_a1.append(counter_i)
                if np.abs(np.sum(a1)) >= 7.5:
                    for j in range(len(a1)):
                        stack_all_ss['action'][j]['camera'][0] = np.sum(a1)
            else:
                a1 = []
                counter_i_a1 = []

            if 15 > np.abs(action['camera'][1]) > 0:
                a2.append(action['camera'][1])
                counter_i_a2.append(counter_i)
                if np.abs(np.sum(a2)) >= 7.5:
                    for j in range(len(a2)):
                        stack_all_ss['action'][j]['camera'][1] = np.sum(a2)
            else:
                a2 = []
                counter_i_a2 = []
            counter_i+=1



        for i in range(len(stack_all_ss['obs'])):
            obs = stack_all_ss['obs'][i]
            action = stack_all_ss['action'][i]

            final = obs['pov']
            final_f = np.fliplr(obs['pov'])
            key, key_f = dataset_action_batch_to_actions(action)
            final_data = []
            after_proces, after_proces_f = process_inventory(obs, angle_1, angle_2, angle_2_f, last_action)
            final_data.append(copy.deepcopy(after_proces))

            angle_2 += action['camera'][1]
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2

            data_l = np.zeros(18)
            data_l[:16] = after_proces
            data_l[16] = 0
            angle_2_fix = copy.deepcopy(angle_2)
            if angle_2_fix > 180:
                angle_2_fix = 360 - angle_2_fix
                data_l[16] = 1
            data_l[17] = np.clip(angle_2_fix / 180, 0, 1)
            final_data.append(copy.deepcopy(data_l))

            angle_1 -= action['camera'][0]
            angle_1 = np.clip(angle_1, -90, 90)

            data_l = np.zeros(19)
            data_l[:18] = final_data[-1]
            data_l[18] = np.clip((angle_1 + 90) / 180, 0, 1)
            final_data.append(copy.deepcopy(data_l))

            data_l = np.zeros(24)
            data_l[:19] = final_data[-1]
            data_l[19] = action['jump']
            data_l[20] = action['forward']
            data_l[21] = action['back']
            data_l[22] = action['left']
            data_l[23] = action['right']
            final_data.append(copy.deepcopy(data_l))



            angle_2_f -= action['camera'][1]
            if angle_2_f > 360:
                angle_2_f = angle_2_f - 360
            elif angle_2_f < 0:
                angle_2_f = 360 + angle_2_f
            last_action = action


            check_add = True

            if check_add:
                all_pov_obs.append(final)
                all_actions.append(key)
                for j in range(4):
                    all_data_obs[j].append(final_data[j])
            # all_pov_obs.append(final_f)
            # all_actions.append(key_f)
            # all_data_obs.append(after_proces_f)




    all_actions = np.array(all_actions)
    # all_pov_obs = np.array(all_pov_obs)
    for j in range(4):
        all_data_obs[j] = np.array(all_data_obs[j])
        print(all_data_obs[j].shape)
    np.sum(all_actions[:,1] == a2_0_Index)
    network = NatureCNN((max_shape, 64, 64), len(list_key[0]), all_data_obs[1].shape[1]).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1]),  all_data_obs[0].shape[1]).cuda()
    optimizer_1 = th.optim.Adam(network_1.parameters(), lr=LEARNING_RATE)
    loss_function_1 = nn.CrossEntropyLoss()

    network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2]),  all_data_obs[2].shape[1]).cuda()
    optimizer_2 = th.optim.Adam(network_2.parameters(), lr=LEARNING_RATE)
    loss_function_2 = nn.CrossEntropyLoss()

    network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3]),  all_data_obs[3].shape[1]).cuda()
    optimizer_3 = th.optim.Adam(network_3.parameters(), lr=LEARNING_RATE)
    loss_function_3 = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []

    print("Training")
    for _ in range(10):
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

            inven = th.from_numpy(all_data_obs[1][batch_indices]).float().cuda()
            logits = network(obs, inven)
            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            actions = all_actions[batch_indices, 0]
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            inven = th.from_numpy(all_data_obs[0][batch_indices]).float().cuda()

            logits = network_1(obs, inven)
            actions = all_actions[batch_indices, 1]
            loss_1 = loss_function_1(logits, th.from_numpy(actions).long().cuda())
            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            inven = th.from_numpy(all_data_obs[2][batch_indices]).float().cuda()
            logits = network_2(obs, inven)
            actions = all_actions[batch_indices, 2]
            loss_2 = loss_function_2(logits, th.from_numpy(actions).long().cuda())
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            inven = th.from_numpy(all_data_obs[3][batch_indices]).float().cuda()
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


    th.save(network.state_dict(), 'pen/a_1.pth')
    th.save(network_1.state_dict(), 'pen/a_2.pth')
    th.save(network_2.state_dict(), 'pen/a_3.pth')
    th.save(network_3.state_dict(), 'pen/a_4.pth')
    del data




network = NatureCNN((max_shape, 64, 64), len(list_key[0])).cuda()
network.load_state_dict(th.load('pen/a_1.pth'))

network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1])).cuda()
network_1.load_state_dict(th.load('pen/a_2.pth'))

network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2])).cuda()
network_2.load_state_dict(th.load('pen/a_3.pth'))

network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3])).cuda()
network_3.load_state_dict(th.load('pen/a_4.pth'))



env = gym.make(ENV_RUN)


env.seed(2)

rewards = []

for episode in range(20):
    obs = env.reset()


    done = False
    steps = 0

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
    for i in range(18000):

        final = obs['pov']

        final_data = []
        final = final.transpose(2, 0, 1).astype(np.float32)
        final = th.from_numpy(final[None]).float().cuda()
        final /= 255.0

        inven,_ = process_inventory(obs, -angle_1, angle_2, 0, last_action)
        final_data.append(copy.deepcopy(inven))


        inven = th.from_numpy(inven[None]).float().cuda()
        new_a = env.action_space.noop()

        probabilities = th.softmax(network_1(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[1])), p=probabilities)
        new_a['camera'][1] = to_action_0[1][action]

        angle_2 += new_a['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        data_l = np.zeros(18)
        data_l[:16] = final_data[-1]
        data_l[16] = 0
        angle_2_fix = copy.deepcopy(angle_2)
        if angle_2_fix > 180:
            angle_2_fix = 360 - angle_2_fix
            data_l[16] = 1
        data_l[17] = np.clip(angle_2_fix / 180, 0, 1)
        final_data.append(copy.deepcopy(data_l))
        inven = th.from_numpy(data_l[None]).float().cuda()

        probabilities = th.softmax(network(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[0])), p=probabilities)
        new_a['camera'][0] =to_action_0[0][action]

        angle_1 -= new_a['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        data_l = np.zeros(19)
        data_l[:18] = final_data[-1]
        data_l[18] = np.clip((angle_1 + 90) / 180, 0, 1)
        final_data.append(copy.deepcopy(data_l))
        inven = th.from_numpy(data_l[None]).float().cuda()

        probabilities = th.softmax(network_2(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        for k, v in to_action_0[2][action].items():
            new_a[k] = v

        data_l = np.zeros(24)
        data_l[:19] = final_data[-1]
        data_l[19] = new_a['jump']
        data_l[20] = new_a['forward']
        data_l[21] = new_a['back']
        data_l[22] = new_a['left']
        data_l[23] = new_a['right']
        final_data.append(copy.deepcopy(data_l))
        inven = th.from_numpy(data_l[None]).float().cuda()

        probabilities = th.softmax(network_3(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        for k, v in to_action_0[3][action].items():
            new_a[k] = v

        obs, reward, done, info = env.step(new_a)
        last_action = new_a

        angle_1 -= new_a['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += new_a['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        # obs, reward, done, info = env.step(new_a)
        # if action not in [0, 7]:
        #     for _ in range(3):
        #         if done:
        #             break
        #         obs, reward, done, info = env.step(env.action_space.noop())
        #
        #         cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        #         cv2.resizeWindow('image', 950, 950)
        #         if cv2.waitKey(10) & 0xFF == ord('o'):
        #             break
        #         # time.sleep(0.05)
        #
        #     print(action)
        if new_a['use'] == 1:
            for _ in range(3):
                obs, reward, done, info = env.step(env.action_space.noop())



        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
        steps += 1
        if done:
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        # time.sleep(0.01)
    cv2.destroyAllWindows()
    print(obs['equipped_items'])



import time
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
for i in range(360,len(all_visibale_obs)):
    image = all_visibale_obs[i]
    print(esp_all[i])
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 1000, 1000)
    time.sleep(0.35)
    if cv2.waitKey(10) & 0xFF == ord('o'):
        break


counter = 0
all_pov = []
all_action = []
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
from pynput.keyboard import Key, Listener

# trajectory_name_s = {}
def on_press(key):
    global trajectory_name_s
    global counter_i
    global trajectory_name
    try:
        key_on_press = key.char

        if key_on_press == 'x':
            if trajectory_name not in trajectory_name_s:
                trajectory_name_s[trajectory_name] = []
            if counter_i not in trajectory_name_s[trajectory_name]:
                trajectory_name_s[trajectory_name].append(counter_i)

    except AttributeError:
        # do something when a certain key is pressed, using key, not key.char
        pass

def on_release(key):

    if key == Key.esc:
        # Stop listener
        print("Stop lister ")
        key_on_release = 'esc'
        return False
key_list.stop()

key_list = Listener(on_press=on_press, on_release=on_release)
key_list.start()
# del trajectory_name_s['v3_wasteful_carrot_vampire-3_37191-38820']
while True:
    if counter == len(trajectory_names):
        break

    trajectory_name = trajectory_names[counter]
    if trajectory_name in trajectory_name_s:
        counter += 1
        continue
    # trajectory_name = 'v3_accomplished_pattypan_squash_ghost-6_44486-45290'
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    angle_1 = 0
    angle_2 = 0
    counter_i = 0
    all_pov = []
    for obs, action, r, _, _ in trajectory:
        all_pov.append(obs['pov'])
        all_action.append(action)

    for i in range(len(all_pov)):
        if i <= len(all_pov)/1.3:
            counter_i += 1
            continue
        obs = all_pov[i]
        cv2.imshow('image', cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 1000, 1000)
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        counter_i += 1
        time.sleep(0.05)
    counter += 1
key_list.stop()

# with open('trajectory_name_s.pkl', 'wb') as f:
#     pickle.dump(trajectory_name_s, f)

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



counter = 1
all_pov = []
all_action = []
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)

while True:
    if counter == len(trajectory_names):
        break

    trajectory_name = trajectory_names[counter]
    # if trajectory_name in trajectory_name_s:
    #     counter += 1
    #     continue
    # trajectory_name = 'v3_accomplished_pattypan_squash_ghost-6_44486-45290'
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    angle_1 = 0
    angle_2 = 0
    counter_i = 0
    for obs, action, r, _, _ in trajectory:
        all_pov.append(obs['pov'])
        all_action.append(action)
        angle_1 -= action['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)
        print(angle_1)
        # if action['equip'] == 'water_bucket':
        #     print("ccc")
        # print(obs['equipped_items']['mainhand']['type'])
        # counter_i += 1
        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 1000, 1000)
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
    counter += 1
