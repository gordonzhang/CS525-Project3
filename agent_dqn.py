#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle

import random
import math
import numpy as np
from collections import deque, namedtuple
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN



torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            parameters for neural network
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN, self).__init__(env)

        self.num_episodes = 1000
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE_STEPS = 10000

        self.START_LEARNING = 5000
        self.replay_memory_size = 10000
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.recent_screens = deque(maxlen=4)

        resized_dim = 84
        self.num_screens = 4

        self.env = env
        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(resized_dim, resized_dim, self.num_screens, self.n_actions).to(device)
        self.target_net = DQN(resized_dim, resized_dim, self.num_screens, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0
        self.episode_rewards = []

        self.num_episodes_per_report = 20
        self.num_episodes_per_save = 500
        self.save_path = "C:/Users/gordo/Desktop/saved_DQN"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_net.load_state_dict(torch.load(args.path))
            

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass
    
    
    def make_action(self, state, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        dice_roll = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if dice_roll > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)


    def push(self, *args):
        self.replay_memory.append(Transition(*args))
        
        
    def replay_buffer(self, batch_size):
        return random.sample(self.replay_memory, batch_size)


    def train(self):
        for i_episode in range(self.num_episodes):
            self.env.reset()
            # get initial screen
            screen_init, _, _, _ = self.env.step(0)
            screen_init = self.preprocess_screen(screen_init)
            # prepare the initial state with 4 screen frames
            screen_blank = screen_init - screen_init
            self.recent_screens.extend([screen_blank] * 3)
            self.recent_screens.append(screen_init)
            # dim of state is (BCHW)
            state = self.get_state_from(self.recent_screens)

            done = False
            episode_reward = 0

            while not done:
                # Select and perform an action
                action = self.make_action(state)
                screen_next, reward, done, _ = self.env.step(action.item())

                reward = torch.tensor([reward], device=device)
                episode_reward += reward

                screen_next = self.preprocess_screen(screen_next)
                self.recent_screens.append(screen_next)

                if done:
                    next_state = None
                    self.episode_rewards.append(episode_reward)
                else:
                    next_state = self.get_state_from(self.recent_screens)

                self.push(state, action, next_state, reward)
                state = next_state

                # Perform one step of the optimization
                if len(self.replay_memory) >= self.START_LEARNING:
                    self.optimize_model()

                # Update the target network, copying all weights and biases in DQN
                if self.steps_done % self.TARGET_UPDATE_STEPS == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % self.num_episodes_per_report == 0:
                val = (sum(self.episode_rewards)/len(self.episode_rewards)).item()
                print(f'Average reward {val:.2f} of last {self.num_episodes_per_report} episodes. Last episode: {i_episode}')

                with open(self.save_path + '/avg_rewards.txt', 'a') as reward_file:
                    reward_file.write(f'{i_episode}, {self.steps_done}, {val}\n')

            if i_episode % self.num_episodes_per_save == 0:
                print(f'Saving target to disk at episode {i_episode}')
                self.save_model(self.target_net, f'ep_{i_episode:07}_')


    def get_state_from(self, screens):
        # screen dimension is (CHW) with C=1
        # concatenate all screens in the 0 dimension (channel)
        state = torch.cat(list(screens), 0)
        # add batch dimension to make the output state (BCHW)
        state = state.unsqueeze(0)
        return state.to(device)



    def preprocess_screen(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = cv2.cvtColor(cv2.resize(screen, (84, 110)), cv2.COLOR_BGR2GRAY)
        # do not include top 26 pixels of screen which contains only score
        screen = screen[26:110,:]
        # convert to binary image 0 or 255 with threshold 1
        ret, screen = cv2.threshold(screen,1,255,cv2.THRESH_BINARY)
        screen = torch.from_numpy(screen)
        # output screen dimension: (CHW)
        screen = screen.unsqueeze(0)
        return screen.to(device)


    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        transitions = self.replay_buffer(self.BATCH_SIZE)
        # batch dimension: BCHW
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # state_batch dim: BCHW
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def save_model(self, net, prefix):
        torch.save(net.state_dict(), self.save_path + "/" + prefix + "model.pth")
        # with open(self.save_path + "/" + prefix + 'avg_rewards.data', 'wb') as file_handle:
        #     # store the data as binary data stream
        #     pickle.dump(self.episode_rewards[], file_handle)


    def load_model(self, dqn):
        return dqn.load_state_dict(torch.load(self.model_path))