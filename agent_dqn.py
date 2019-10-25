#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import math
import numpy as np
from collections import deque, namedtuple
import os
import sys
import torchvision.transforms as T
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

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
        super(Agent_DQN,self).__init__(env)

        self.num_episodes = 100
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.replay_memory_size = 10000
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.recent_screens = deque(maxlen=4)

        resized_dim = 84
        self.history = 4
        self.resize = T.Compose([T.ToPILImage(),
                            T.Resize(resized_dim, interpolation=Image.CUBIC),
                            T.ToTensor()])

        self.env = env
        self.
        state0 = self.env.step(0)
        state0 = self.preprocess(state0)

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(resized_dim, resized_dim, self.history, self.n_actions).to(device)
        self.target_net = DQN(resized_dim, resized_dim, self.history, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0


        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################
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
            # Initialize the environment and state
            self.env.reset()
            state_init,_,_,_ = self.env.step(0)
            state_init_p = self.preprocess_screen(state_init)
            current_screen = get_screen()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = select_action(state)
                state_next, reward, done, _ = env.step(action.item())
                print(state_next.shape)
                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(device)


    def preprocess_screen(self, screen):
        screen = cv2.cvtColor(cv2.resize(screen, (84, 110)), cv2.COLOR_BGR2GRAY)
        screen = screen[26:110,:]
        ret, screen = cv2.threshold(screen,1,255,cv2.THRESH_BINARY)
        screen = torch.from_numpy(screen)
        screen = screen.unsqueeze(2)

        return screen


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
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
        for param in self.self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()