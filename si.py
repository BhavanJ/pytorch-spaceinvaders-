import sys
import argparse
import random
import math
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import cv2
import pdb
#from skimage.transform import resize
#from skimage.color import rgb2gray



# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# ByteTensor = torch.ByteTensor
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))



class Space_Invaders_CNN(nn.Module):
    def __init__(self):
        super(Space_Invaders_CNN, self).__init__()
        self.conv1 = nn.Conv2d(4,16,8,stride=4) #output will be 20x20 feature
        self.conv2 = nn.Conv2d(16,32,4,stride=2) #output will be 9x9
        self.fc1 = nn.Linear(32*81,256)
        self.fc2 = nn.Linear(256,6)

    def forward(self,x):

        #pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,32*81)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearQN(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinearQN, self).__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = self.fc(x)
        return x

class DQN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_out

        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 2*n_hidden)

        self.fc1_adv = nn.Linear(2*n_hidden, n_hidden)
        self.fc1_val = nn.Linear(2*n_hidden, n_hidden)

        self.fc2_adv = nn.Linear(n_hidden, self.n_actions)
        self.fc2_val = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.n_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return x

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent(object):
    def __init__(self, args, render=False):
        self.env = gym.make(args.env)
        self.expt_name = args.name
        # self.env = gym.wrappers.Monitor(self.env, directory='monitors/'+args.env, force=True)
        #n_in = self.env.observation_space.shape[0]
        #n_out = self.env.action_space.n
        self.batch_size = args.batch_size

        # type of function approximator to use
        if args.model_type == 'Space_Invaders_CNN':
            self.model = Space_Invaders_CNN()
        elif args.model_type == 'linear':
            self.model = LinearQN(n_in, n_out)
        elif args.model_type == 'dqn':
            self.model = DQN(n_in, args.n_hidden, n_out)
        else:
            self.model = DuelingDQN(n_in, args.n_hidden, n_out)


        if use_cuda:
            self.model.cuda()

        # should experience replay be used
        if args.exp_replay:
            self.exp_replay = True
            self.memory = ReplayMemory(args.buffer_size)
        else:
            # memory of size 1 is same as using only the immediate transitions
            # this is only to keep the overall api similar for all cases
            self.memory = ReplayMemory(1)
            assert self.batch_size == 1

        # policy type
        if args.eps_greedy:
            self.eps_greedy = True
            self.eps_start = args.eps_start
            self.eps_end = args.eps_end
            self.eps_decay = args.eps_decay
        else:
            self.eps_greedy = False

        if args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters())
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.num_episodes = args.num_episodes
        self.loss_fn = args.loss_fn
        self.steps_done = 0
        self.episode_durations = []
        self.avg_rewards = []
        self.memory_burn_limit = args.memory_burn_limit

        #pdb.set_trace()

    def select_action(self, state, train):

        #pdb.set_trace()
        state = FloatTensor(state)
        if train:
            self.steps_done += 1
        # action will be selected based on the policy type : greedy or epsilon-greedy
        if self.eps_greedy:
            # smoothly decaying the epsilon threshold value as we progress
            if train:
                # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.*(self.steps_done/self.eps_decay))
                eps_threshold = (self.steps_done)*((self.eps_end - self.eps_start)/(self.eps_decay)) + self.eps_start
            else:
                eps_threshold = 0.05
            # explore or exploit?
            if random.random() > eps_threshold:
                return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
                # with torch.no_grad():
                #    action = self.model(Variable(state))
                # return action.data.max(1)[1].view(1,1)
            else:
                return LongTensor([[random.randrange(self.env.action_space.n)]])
                #return LongTensor([[random.randrange(2)]])
        else:
            return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            # with torch.no_grad():
            #    action = self.model(Variable(state))
            # return action.data.max(1)[1].view(1,1)


    def burn_memory(self):

        steps = 0
        state = np.zeros((4,84,84),dtype=np.float64)
        next_state = np.zeros((4,84,84),dtype=np.float64)
        # state = FloatTensor(4,84,84)
        # next_state = FloatTensor(4,84,84)

        state_single = self.env.reset()
        #state_single = rgb2gray(resize(state_single,(84,84)))
        # print("[before] : state_single : ", state_single)
        state_single = self.normalize_image(state_single)
        # print("[after] : state_single : ", state_single)

        state[0,:,:] = state_single
        state[1,:,:] = state_single
        state[2,:,:] = state_single
        state[3,:,:] = state_single

        print('Starting to fill the memory with random policy')
        while steps < self.memory_burn_limit:

            #Executing a random policy
            action = LongTensor([[random.randrange(self.env.action_space.n)]])
            next_state_single, reward, is_terminal, _ = self.env.step(action[0,0])
            #next_state_single = rgb2gray(resize(next_state_single,(84,84)))
            next_state_single = self.normalize_image(next_state_single)

            next_state[0,:,:] = state[1,:,:]
            next_state[1,:,:] = state[2,:,:]
            next_state[2,:,:] = state[3,:,:]
            next_state[3,:,:] = next_state_single

            #self.memory.store(FloatTensor([state]),
            #                  action,
            #                  FloatTensor([next_state]),
            #                  FloatTensor([reward]))


            if is_terminal:
                # store the transition in memory
                # next_state = None
                self.memory.store(np.array([state]),
                                  action.cpu().numpy(),
                                  None,
                                  np.array([reward]))
            else:
                self.memory.store(np.array([state]),
                                  action.cpu().numpy(),
                                  np.array([next_state]),
                                  np.array([reward]))


            steps += 1
            state = next_state


            while steps == self.memory_burn_limit and not is_terminal:
               #Executing a random policy
               action = LongTensor([[random.randrange(self.env.action_space.n)]])
               next_state, reward, is_terminal, _ = self.env.step(action[0,0])


            #If the next_state is terminal, then you reset it
            if is_terminal:
                state_single = self.env.reset()
                #state_single = rgb2gray(resize(state_single,(84,84)))
                state_single = self.normalize_image(state_single)
                print(steps)
                # print("state_single : ", state_single)
                # print("state[0,:,:] : ", state[0,:,:])
                state[0,:,:] = state_single
                state[1,:,:] = state_single
                state[2,:,:] = state_single
                state[3,:,:] = state_single


        print('Memory filled, ready to start training now')
        print("-"*50)

################################################################################################################################################

    def normalize_image(self,state):
        state = cv2.resize(state, (84, 84))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        #state = rgb2gray(resize(state,(84,84)))
        state = state.astype(np.float64)
        state = state/(1.0*np.max(state))
        return state


    def testing_random_play(self):

        state = self.env.reset() #state is 210,160,3
        state = self.normalize_image(state)

        for i in range(1000):
            #action = random.randrange(self.env.action_space.n)
            action = LongTensor([[random.randrange(self.env.action_space.n)]])
            next_state, reward, is_terminal, _ = self.env.step(action[0,0])
            print(reward,is_terminal)
            # self.env.render()
        print('Random play done now')

################################################################################################################################################

    def play_episode(self, e, train=True):

        state_single = self.env.reset()
        #state_single = rgb2gray(resize(state_single,(84,84)))
        state_single = self.normalize_image(state_single)

        state = np.zeros((4,84,84),dtype=np.float64)
        next_state = np.zeros((4,84,84),dtype=np.float64)

        state[0,:,:] = state_single
        state[1,:,:] = state_single
        state[2,:,:] = state_single
        state[3,:,:] = state_single



        steps = 0
        total_reward = 0
        # iterate till the terminal state is reached
        while True:
            # self.env.render()
            action = self.select_action([state],train)
            # print("action: ", action)
            next_state_single, reward, is_terminal, _ = self.env.step(action[0,0])

            #next_state_single = rgb2gray(resize(next_state_single,(84,84)))
            next_state_single = self.normalize_image(next_state_single)


            next_state[0,:,:] = state[1,:,:]
            next_state[1,:,:] = state[2,:,:]
            next_state[2,:,:] = state[3,:,:]
            next_state[3,:,:] = next_state_single

            total_reward += reward

            if is_terminal:
                # store the transition in memory
                # next_state = None
                self.memory.store(np.array([state]),
                                  action.cpu().numpy(),
                                  None,
                                  np.array([reward]))
            else:
                self.memory.store(np.array([state]),
                                  action.cpu().numpy(),
                                  np.array([next_state]),
                                  np.array([reward]))

            if train:
                # backprop and learn; otherwise just play the policy
                self.optimize_model()
            # update state
            state = next_state
            steps += 1
            if is_terminal:
                if train:
                    # backprop and learn; otherwise just play the policy
                    # self.optimize_model()
                    #if steps %20 == 0:
                    print("Episode {} completed after {} steps | Total reward = {}".format(e+1,total_reward,self.steps_done))
                self.episode_durations.append(steps)
                # self.plot_durations()
                return total_reward

    def optimize_model(self):
        # check if enough experience collected so far
        # the agent continues with a random policy without updates till then
        if len(self.memory) < self.batch_size:
            return

        self.optimizer.zero_grad()
        # sample a random batch from the replay memory to learn from experience
        # for no experience replay the batch size is 1 and hence learning online
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # isolate the values
        # non_terminal_mask = np.array(list(map(lambda s: s is not None, batch.next_state)))
        non_terminal_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        # with torch.no_grad():
        #     batch_next_state = Variable(torch.cat([s for s in batch.next_state if s is not None]))
        batch_next_state = Variable(torch.cat([FloatTensor(s) for s in batch.next_state if s is not None]),volatile=True)
        # batch_next_state = Variable(torch.cat(batch.next_state))
        # pdb.set_trace()
        batch_state = Variable(torch.cat([FloatTensor(s) for s in batch.state]))
        batch_action = Variable(torch.cat([LongTensor(s) for s in batch.action]))
        batch_reward = Variable(torch.cat([FloatTensor(s) for s in batch.reward]))

        # batch_state = Variable(torch.cat(batch.state))
        # batch_action = Variable(torch.cat(batch.action))
        # batch_reward = Variable(torch.cat(batch.reward))

        # There is no separate target Q-network implemented and all updates are done
        # synchronously at intervals of 1 unlike in the original paper
        # current Q-values
        current_Q = self.model(batch_state).gather(1, batch_action)
        # expected Q-values (target)
        if use_cuda:
            max_next_Q = Variable(torch.zeros(self.batch_size).type(torch.cuda.FloatTensor))
            max_next_Q[non_terminal_mask] = self.model(batch_next_state).max(1)[0]
            max_next_Q.volatile = False
            expected_Q = (max_next_Q * self.gamma) + batch_reward

        # max_next_Q = self.model(batch_next_state).detach().max(1)[0]
        # # expected_Q = np.array(batch.reward)
        # expected_Q = Variable(torch.from_numpy(np.array(batch.reward)).cuda(), volatile=True)

        # if use_cuda:
        #     expected_Q[non_terminal_mask] += (self.gamma * max_next_Q).cpu().data
        #     # with torch.no_grad():
        #     #     expected_Q = Variable(torch.from_numpy(expected_Q).cuda())
        #     expected_Q = Variable(torch.from_numpy(expected_Q).cuda(), volatile=True)
        # else:
        #     expected_Q[non_terminal_mask] += (self.gamma * max_next_Q).data
        #     expected_Q = Variable(torch.from_numpy(expected_Q), volatile=True)
        # expected_Q = batch_reward + (self.gamma * max_next_Q)
        # expected_Q = batch_reward + (self.gamma * max_next_Q)

        # loss between current Q values and target Q values
        if self.loss_fn == 'l1':
            loss = F.smooth_l1_loss(current_Q, expected_Q)
        else:
            loss = F.mse_loss(current_Q, expected_Q)

        # backprop the loss
        loss.backward()
        self.optimizer.step()

    def plot_durations(self):
        durations = torch.FloatTensor(self.episode_durations)
        plt.figure(1)
        plt.clf()
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations.numpy())
        # Averaging over 100 episodes and plotting those values
        if len(durations) >= 100:
            means = durations.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        # pause so that the plots are updated
        plt.pause(0.001)

    def plot_rewards(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training')
        plt.ylabel('Avg Reward')
        plt.plot(self.avg_rewards)
        # pause so that the plots are updated
        plt.pause(0.001)
        # plt.show()

    def train(self):
        print("Going to be training for a total of {} episodes".format(self.num_episodes))
        for e in range(self.num_episodes):
            # print("----------- Episode {} -----------".format(e))
            self.play_episode(e,train=True)
            if e%20 == 0:
                self.test(10)

    def test(self,num_episodes):
        print("-"*50)
        print("Testing for {} episodes".format(num_episodes))
        rewards = []
        for e in range(num_episodes):
            rewards.append(self.play_episode(e,train=False))
        print("Running policy after training for {} updates".format(self.steps_done))
        print("Avg reward achieved in {} episodes : {}".format(num_episodes, np.mean(rewards)))
        print("-"*50)
        self.avg_rewards.append(np.mean(rewards))
        self.save_plots()
        print("\n\n")
        print("mean : ", np.mean(rewards))
        print("std : ", np.std(rewards))
        print("\n\n")

    def save_plots(self):
        # save durations plot
        durations = torch.FloatTensor(self.episode_durations)
        plt.figure(1)
        plt.clf()
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations.numpy())
        # Averaging over 100 episodes and plotting those values
        if len(durations) >= 100:
            means = durations.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.savefig(self.expt_name+'_durations_{}.png'.format(self.steps_done))
        # save reward plot
        plt.figure(2)
        plt.clf()
        plt.title('Testing')
        plt.xlabel('Test')
        plt.ylabel('Avg Reward')
        plt.plot(self.avg_rewards)
        plt.savefig(self.expt_name+'_rewards_{}.png'.format(self.steps_done))

    # def test(self,num_episodes):
    #     total_reward = 0
    #     print("-"*50)
    #     print("Testing for {} episodes".format(num_episodes))
    #     for e in range(num_episodes):
    #         total_reward += self.play_episode(e,train=False)
    #     print("Running policy after training for {} updates".format(self.steps_done))
    #     print("Avg reward achieved in {} episodes : {}".format(num_episodes, total_reward/num_episodes))
    #     print("-"*50)
    #     self.avg_rewards.append(total_reward/num_episodes)
        # self.plot_rewards()


    def close(self):
        self.env.render(close=True)
        self.env.close()
        plt.ioff()
        plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--name',type=str, default='si_plot', help='name of the expt - also used as folder name for saving outputs')
    parser.add_argument('--env',type=str, default='SpaceInvaders-v0')
    parser.add_argument('--render',type=int,default=0)
    parser.add_argument('--model_type',type=str, default='Space_Invaders_CNN',help ='Model type one of (linear,dqn,duel)')
    parser.add_argument('--exp_replay', type=int, default=1, help='should experience replay be used, default 1')
    parser.add_argument('--num_episodes', type=int, default=5000, help='number of episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay memory buffer size')
    # parser.add_argument('--n_in', type=int, default=4, help='input layer size')
    # parser.add_argument('--n_out', type=int, default=256, help='output layer size')
    parser.add_argument('--loss_fn', type=str, default='l2', help='loss function one of (l1,l2) | Default: l1')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer one of (rmsprop,adam) | Default : rmsprop')
    parser.add_argument('--n_hidden', type=int, default=32, help='hidden layer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--frame_hist_len', type=int, default=4, help='frame history length | Default : 4')
    parser.add_argument('--eps_greedy', type=int, default=1, help='should policy be epsilon-greedy, default 1')
    parser.add_argument('--eps_start', type=float, default=0.5, help='e-greedy threshold start value')
    parser.add_argument('--eps_end', type=float, default=0.05, help='e-greedy threshold end value')
    parser.add_argument('--eps_decay', type=int, default=100000, help='e-greedy threshold decay')
    parser.add_argument('--logs', type=str, default = 'logs',  help='logs path')
    parser.add_argument('--memory_burn_limit', type=int,default=50000, help='Till when to burn memory')
    return parser.parse_args()

def main():
    plt.ion()
    # plt.figure()
    # plt.show()

    args = parse_arguments()
    print(args)
    agent  = Agent(args)

    #agent.testing_random_play()
    #pdb.set_trace()

    agent.burn_memory()
    #pdb.set_trace()
    agent.train()
    print('----------- Completed Training -----------')
    pdb.set_trace()
    agent.test(num_episodes=20)
    print('----------- Completed Testing -----------')

    pdb.set_trace()
    agent.close()

    # plt.ioff()
    # plt.show()

if __name__ == '__main__':
    main()



#TODO
#1) Verify is this correct--->>> rgb2gray(resize(state_single,(84,84)))
#2) Storing none for next_state if it's terminal state during burning
#3) Check is this required in burning memory.......            #while steps == self.memory_burn_limit and not is_terminal:
#4) Take a look at the resized images and crop the center region


#Increase the memory size

#Updates
#1) Changed the image normalization function
#2) Using state.tolist() while storing in memory in burn_memory()
#3) FloatTensor(next_state) instead of FloatTensor([next_state])
