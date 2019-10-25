import gym
import math
import random
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# print(list(zip(*[('a', 1), ('b', 2), ('c', 3), ('d', 4)])))
# print(list(zip(('a', 1), ('b', 2), ('c', 3), ('d', 4))))


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
#
# a = [torch.tensor(1),torch.tensor(2),torch.tensor(3),torch.tensor(4)]
# b = [torch.tensor(5),torch.tensor(6),None,torch.tensor(8)]
# c = [torch.tensor(9),torch.tensor(8),torch.tensor(2),torch.tensor(6)]
#
#
# t1 = Transition(*a)
# t2 = Transition(*b)
# t3 = Transition(*c)
#
# # print(t1)
#
# batch = Transition(*zip(*[t1,t2,t3]))
# # print(batch.state)
#
# reward_batch = torch.stack(batch.reward)
# print(reward_batch)
#
# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                       batch.next_state)), device=device, dtype=torch.bool)
# non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
#
# print(non_final_mask)
# # print(non_final_next_states)
#
# next_state_values = torch.zeros(3)
# next_state_values[non_final_mask] = torch.cat([[torch.tensor(9)], [torch.tensor(2)]])
# print(non_final_next_states)
# print(next_state_values)



state = np.array([[[0.,0.,30.,40.,50.],[2.,20.,30.,40.,50.],[20.,30.,40.,50.,60.],[30.,40.,50.,60.,70.]],
                  [[0.,0.,30.,40.,50.],[2.,20.,30.,40.,50.],[20.,30.,40.,50.,60.],[30.,40.,50.,60.,70.]],
                  [[0.,0.,30.,40.,50.],[2.,20.,30.,40.,50.],[20.,30.,40.,50.,60.],[30.,40.,50.,60.,70.]]])
state = state.transpose(1,2,0)
print(state.shape)
# print(state)

state = np.ascontiguousarray(state, dtype=np.float32)
state = cv2.cvtColor(cv2.resize(state, (6, 4)), cv2.COLOR_BGR2GRAY)
ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
state = torch.from_numpy(state)
state = state.unsqueeze(2)
state = torch.cat([state, state], 2)
# state = state.squeeze(2)

print(state)
print(state.shape)



# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(5, interpolation=Image.CUBIC),
#                     T.ToTensor()])
#
# state = torch.from_numpy(state)
# state = resize(state).unsqueeze(0)
# state = torch.cat([state, state], 0)
# print(state.size())
# print(state)





# print(state)
# state = state[26:110, :]
# ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
# np.reshape(state, (84, 84, 1))

