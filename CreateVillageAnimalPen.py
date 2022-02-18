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

with open('a_k_f.pkl', 'rb') as f:
    action_key = pickle.load(f)
list_key_index = np.array(list(action_key.keys()))
print(len(list_key_index))

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
            nn.Linear(15, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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



def process_a(action, action_new):
    key = list_key_index[action]
    action_new = copy.deepcopy(action_new)
    tmp = copy.deepcopy(action_key)
    for k, v in tmp[key].items():
        action_new[k] = v


    return action_new

def dataset_action_batch_to_actions(actions):
    action_with_key = actions
    dict_action = {}
    a1 = action_with_key['camera'][0]
    a2 = action_with_key['camera'][1]
    k = f'0_0'
    if action_with_key['equip'] not in ['none', 'other' ]:
        equip = action_with_key['equip']
        k = f'equip_{equip}'
        dict_action['equip'] = equip
    elif action_with_key['use'] == 1:
        dict_action['use'] = 1
        k = f'use'

    elif action_with_key['jump'] == 1:
        dict_action['jump'] = 1
        dict_action['forward'] = 1
        k = f'j_f'

    elif np.abs(a1) >= 2:
        if a1 > 0:
            dict_action['camera'] = [5, 0]
            k = f'1_0'
        elif a1 < 0 :
            dict_action['camera'] = [-5,0]
            k = f'-1_0'

    elif np.abs(a2) >= 2:
        if a2 > 0:
            dict_action['camera'] = [0, 5]
            k = f'0_1'
        elif a2 < 0:
            dict_action['camera'] = [0, -5]
            k = f'0_-1'
    elif action_with_key['forward'] == 1:
        dict_action['forward'] = 1
        k = f'f'

    elif action_with_key['back'] == 1:
        dict_action['back'] = 1
        k = f'b'
    elif action_with_key['left'] == 1:
        dict_action['left'] = 1
        k = f'l'

    elif action_with_key['right'] == 1:
        dict_action['right'] = 1
        k = f'r'
    elif action_with_key['attack'] == 1:
        dict_action['attack'] = 1
        k = f'a'
    else:
        dict_action['camera'] = [0,0]
        k = f'0_0'
        # if np.abs(a1) >= 0.5:
        #     if a1 > 0:
        #         dict_action['camera'] = [2, 0]
        #         k = f'2_0'
        #     elif a1 < 0:
        #         dict_action['camera'] = [-2, 0]
        #         k = f'-2_0'
        # elif np.abs(a2) >= 0.5:
        #     if a2 > 0:
        #         dict_action['camera'] = [0, 2]
        #         k = f'0_2'
        #     elif a2 < 0:
        #         dict_action['camera'] = [0, -2]
        #         k = f'0_-2'


    return k, dict_action
def dataset_action_batch_to_actions(actions):

    action_with_key = actions
    dict_action = {}
    if action_with_key['equip'] not in ['none', 'other']:
        equip = action_with_key['equip']
        k = f'equip_{equip}'
        dict_action['equip'] = equip
    elif action_with_key['use'] == 1:
        dict_action['use'] = 1
        k = f'use'
    elif action_with_key['jump'] == 1:
        dict_action['jump'] = 1
        dict_action['forward'] = 1
        k = f'j_f'
    elif action_with_key['attack'] == 0:
        k, dict_action = find_key('move', action_with_key, '', dict_action)
    else:
        dict_action['attack'] = 1
        k, dict_action = find_key('jump', action_with_key, 'attack', dict_action)

    return k, dict_action

def find_key(byWhat, action_with_key, key, dict_action):
    first_angle = action_with_key['camera'][0]
    second_angle = action_with_key['camera'][1]
    if byWhat == 'camera':
        camera = []
        if np.abs(first_angle) > 1:
            if first_angle < 0:
                key += '_-1'
                camera.append(-10)
            else:
                key += '_1'
                camera.append(10)
        else:
            key += '_0'
            camera.append(0)

        if np.abs(second_angle) > 1:
            if second_angle < 0:
                key += '_-1'
                camera.append(-10)
            else:
                key += '_1'
                camera.append(10)
        else:
            key += '_0'
            camera.append(0)
        dict_action['camera'] = camera
        return key, dict_action
    elif byWhat == 'jump':
        if action_with_key['jump'] == 1:
            key += '_j'
            dict_action['jump'] = 1
            key, dict_action = find_key('move', action_with_key, key, dict_action)
        elif action_with_key['back'] == 1:
            key += '_b'
            dict_action['back'] = 1
        elif action_with_key['right'] == 1:
            key += '_r'
            dict_action['right'] = 1
        elif action_with_key['left'] == 1:
            key += '_l'
            dict_action['left'] = 1
        else:
            key, dict_action = find_key('move', action_with_key, key, dict_action)
    elif byWhat == 'move':
        if action_with_key['forward'] == 1:
            key += '_f'
            dict_action['forward'] = 1
        elif action_with_key['back'] == 1:
            key += '_b'
            dict_action['back'] = 1
        elif action_with_key['right'] == 1:
            key += '_r'
            dict_action['right'] = 1
        elif action_with_key['left'] == 1:
            key += '_l'
            dict_action['left'] = 1
        key, dict_action = find_key('camera', action_with_key, key, dict_action)
    return key, dict_action


def process_inventory(obs, attack, angle_1, angle_2):

    data = np.zeros(15)

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
    data[13] = (angle_1 + 90) / 180
    angle_2 = int(angle_2)
    data[14] = np.clip(((angle_2 + 180) % 360) / 360, 0, 1)
    return data


def checking_craft_place(obs, action):
    inventory = obs['inventory']
    if action["craft"] == 'planks':
        if inventory['log'] == 0:
            action["craft"] = 'none'

    if action["craft"] == 'stick':
        if inventory['planks'] <= 1:
            action["craft"] = 'none'
    if action["craft"] == 'crafting_table':
        if inventory['planks'] < 4:
            action["craft"] = 'none'
    if action["craft"] == 'torch':
        if inventory['coal'] < 1 or inventory['stick'] < 1:
            action["craft"] = 'none'

    if action["nearbyCraft"] == 'wooden_pickaxe':
        if inventory['stick'] < 2 or inventory['planks'] < 3:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'stone_pickaxe':
        if inventory['stick'] < 2 or inventory['cobblestone'] < 3:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'furnace':
        if inventory['cobblestone'] < 8 and inventory['stone'] < 8:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'iron_pickaxe':
        if inventory['stick'] < 2 or inventory['iron_ingot'] < 3:
            action["nearbyCraft"] = 'none'

    if action["place"] == 'furnace':
        if inventory['furnace'] == 0:
            action["place"] = 'none'
    if action["place"] == 'crafting_table':
        if inventory['crafting_table'] == 0:
            action["place"] = 'none'
        # elif inventory['stick'] < 2 and:
        #     if not (inventory['furnace'] == 0 and inventory['cobblestone'] >= 8):
        #         action["place"] = 'none'
        # else:
        #     if not (inventory['furnace'] == 0 and inventory['cobblestone'] >= 8):
        #         if (inventory['planks'] < 3 and
        #                 inventory['cobblestone'] < 3 and
        #                 (inventory['iron_ingot'] +
        #                  inventory['iron_ore'] < 3)):
        #             action["place"] = 'none'
        #         elif (inventory['cobblestone'] < 3 and
        #               (inventory['iron_ingot'] +
        #                inventory['iron_ore'] < 3)):
        #             action["place"] = 'none'
        #         elif (inventory['iron_ingot'] +
        #               inventory['iron_ore'] < 3):
        #             action["place"] = 'none'

    return action

item_by_attack_index = [0, 1, 3, 7, 9, 12]
none_move_action = np.array(['attack', 'craft_crafting_table',
                             'craft_planks', 'craft_stick', 'equip_iron_pickaxe',
                             'equip_stone_pickaxe', 'equip_wooden_pickaxe',
                             'nearbyCraft_furnace', 'nearbyCraft_iron_pickaxe',
                             'nearbyCraft_stone_pickaxe', 'nearbyCraft_wooden_pickaxe',
                             'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace',
                             'nearbySmelt_coal', 'craft_torch', 'place_torch', 'place_cobblestone',
                             'place_stone', 'place_dirt'])
list_obs = [[-10, 0], [10, 0], [0, -10], [0, 10]]
list_obs = [-2, -4, -6, -8, -10]



def train():
    print("Prepare_Data")

    data = minerl.data.make("MineRLBasaltCreateVillageAnimalPen-v0",  data_dir='data', num_workers=4)

    all_actions = []
    all_pov_obs = []
    all_data_obs = []

    keyss = []
    # action_key = {}

    print("Loading data")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    counter = 0
    # Add trajectories to the data until we reach the required DATA_SAMPLES.


    for trajectory_name in trajectory_names:
        # trajectory_name = 'v3_accomplished_pattypan_squash_ghost-6_44486-45290'
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

        last_inventory = None
        time = 0
        angle_1 = 0
        angle_2 = 0
        time_attack_no_new = 0
        current_item = 0
        stack_grey = []
        stack_index = []
        stack_index_max = []

        counter = 0
        for obs, action, r, _, _ in trajectory:
            counter += 1
            print(obs['inventory'])
            if counter >= 81:
                break
            grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])


            key, dict_action = dataset_action_batch_to_actions(action)
            keyss.append(key)

            # angle_2_fix = (np.round(int(angle_2)/10)*10)
            # angle_1_fix = (np.round(int(angle_1)/10)*10)
            stack_grey.append(grey)
            if len(stack_grey) >= 20:
                del stack_grey[0]
            #
            for location in list_obs:
                if len(stack_grey) >= 1 - location:
                    final = np.concatenate((final, stack_grey[location][:, :, None]), axis=-1)
                else:
                    final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)
            final = final.astype(np.uint8)
            # action_key[key] = dict_action
            # if key not in action_key:
            #     action_key[key] = dict_action
            # if len(action_key.keys()) ==  97:
            #     print("aa")
            a = np.where(list_key_index == key)[0][0]
            after_proces = process_inventory(obs, time_attack_no_new, angle_1, angle_2)

            if action['attack']:
                if time_attack_no_new == 0:
                    current_item = np.sum(np.array(list(obs['inventory'].values())))
                    time_attack_no_new += 1

                else:
                    check_new = np.sum(np.array(list(obs['inventory'].values())))
                    if check_new != current_item:
                        time_attack_no_new = 0
                    else:
                        time_attack_no_new += 1
            else:
                time_attack_no_new = 0

            angle_1 += action['camera'][0]
            angle_1 = np.clip(angle_1, -90, 90)
            angle_2 += action['camera'][1]
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2

            time += 1
            if key != '_0_0':
                all_pov_obs.append(final)
                all_actions.append(a)
                all_data_obs.append(after_proces)

    # img = plt.imshow(obs['pov'])
    # plt.show()



    a = Counter(keyss)
    a['_0_0']

    # key = list(action_key.keys())
    # key.sort()
    # new_dict = {}
    # for j in key:
    #     # if j not in list_key_index:
    #     #     print(j)
    #     new_dict[j] = action_key[j]
    # action_key = new_dict
    # with open('a_k_f.pkl', 'wb') as f:
    #     pickle.dump(action_key, f)



    all_actions = np.array(all_actions)
    # all_pov_obs = np.array(all_pov_obs)
    all_data_obs = np.array(all_data_obs)
    # all_actions_r = np.array(all_actions_r)
    # np.bincount(all_actions)
    # print(len(all_actions)/ 1916597)
    # np.sum(all_actions == 80)

    network = NatureCNN((8, 64, 64), len(list_key_index)).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []

    print("Training")
    for _ in range(15):
        # Randomize the order in which we go over the samples
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):
            # break
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            obs = np.zeros((len(batch_indices), 64, 64, 8), dtype=np.float32)
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

            actions = all_actions[batch_indices]
            # distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
            # actions = np.argmin(distances, axis=0)
            # Obtain logits of each action
            inven = th.from_numpy(all_data_obs[batch_indices]).float().cuda()
            logits = network(obs, inven)

            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())

            # Standard PyTorch update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1
            losses.append(loss.item())
            if (update_count % 1000) == 0:
                mean_loss = sum(losses) / len(losses)
                tqdm.write("Iteration {}. Loss {:<10.3f}".format(update_count, mean_loss))
                losses.clear()



    TRAIN_MODEL_NAME = 'a_k_f_1.pth'  # name to use when saving the trained agent.

    th.save(network.state_dict(), TRAIN_MODEL_NAME)
    del data


network = NatureCNN((8, 64, 64), len(list_key_index)).cuda()
network.load_state_dict(th.load('a_k_f_1.pth'))

env = gym.make('MineRLBasaltCreateVillageAnimalPen-v0')


rewards = []

for episode in range(20):
    obs = env.reset()
    total_reward = 0

    done = False
    steps = 0
    # BC part to get some logs:
    action_list = np.arange(len(list_key_index))

    angle_1 = 0
    angle_2 = 0

    last_attack = 0
    time_add = 0
    a2 = 0
    all_obs = []
    action_before = []
    last_action = 0

    data_obs = []
    time_attack_no_new = 0
    current_item = 0
    counter_0_0 = 0
    check_fix = False
    start_fix = 0

    previous = None
    check_attack = False
    time = 0
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    cv2.moveWindow('image', -30, 30)
    stack_grey = []
    time_wait_equi = 0
    for i in range(18000):

        final = obs['pov']
        grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])
        #
        stack_grey.append(grey)
        if len(stack_grey) >= 20:
            del stack_grey[0]
        #
        for location in list_obs:
            if len(stack_grey) >= 1 - location:
                final = np.concatenate((final, stack_grey[location][:, :, None]), axis=-1)
            else:
                final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)

        final = final.transpose(2, 0, 1).astype(np.float32)
        final = th.from_numpy(final[None]).float().cuda()
        final /= 255.0

        inven = process_inventory(obs, time_attack_no_new, angle_1, angle_2)
        inven = th.from_numpy(inven[None]).float().cuda()

        probabilities = th.softmax(network(final, inven), dim=1)[0]
        # Into numpy
        probabilities = probabilities.detach().cpu().numpy()
        # Sample action according to the probabilities
        action = np.random.choice(action_list, p=probabilities)

        key = list_key_index[action]
        # if key == 'attack_0_0' and (check_attack or time_attack_no_new >= 300):
        #     action = np.random.choice([0, 1, 2, 3, 5, 6, 7, 8], p=np.repeat(0.125, 8))
        #     key = list_key_index[action]
        #     time_attack_no_new = 0
        #     # print("aa")
        #
        # if key in none_move_action:
        #     a = 3
        #     # print(key)

        # action_before.append(action)
        # data_obs.append(inven)
        # last_action = action


        time_add += 1

        # for k,v in action_key.items():
        #     print(k,v)
        # action_key['_f_0_0']
        # _f_0_0



        action = process_a(action, env.action_space.noop())
        # print(key)
        angle_1 += action['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += action['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        obs, reward, done, info = env.step(action)
        # if key in ['place_furnace', 'place_crafting_table']:
        #     for __i in range(2):
        #         action = process_a(4, env.action_space.noop())
        #         obs, reward, done, info = env.step(env.action_space.noop())
        #         total_reward += reward
        #         time_add += 1

        if previous is not None:
            delta = previous - obs['pov']
            delta = np.sum(delta)
            if delta == 0:
                check_attack = True
            else:
                check_attack = False
        previous = obs['pov']
        #
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

    rewards.append(total_reward)

    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

np.mean(rewards)
