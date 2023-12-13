import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import random

class PrioritizedReplayBuffer:

    def __init__(self, 
                 max_samples = 10000,
                 alpha = 0.6,
                 beta0 = 0.1,
                 beta_rate = 0.999,
                 eps = 1e-6):
        self.max_samples = max_samples
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray)
        self.n_entries = 0
        self.next_index = 0
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate
        self.eps = eps

    def update(self, index , priorities):
        self.memory[index, 1] = priorities

    def store(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[:self.n_entries, 1].max()
        self.memory[self.next_index,1] = priority
        self.memory[self.next_index,0] = np.array(sample,dtype=object)
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def sample(self, batch_size):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        entries = self.memory[:self.n_entries]

        priorities = entries[:, 1] + self.eps
        scaled_priorities = priorities**self.alpha        
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])
        
        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, 0]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, samples_stacks
    
class QNet_Duel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        advantage = self.linear2(hidden)
        value = self.linear3(hidden)
        value = value.expand_as(advantage)
        qvalue = value + advantage - advantage.mean(-1, keepdim=True).expand_as(advantage)
        return qvalue
    
class QTrainer:
    def __init__(self, lr, gamma,input_dim, hidden_dim, output_dim):
        self.gamma = gamma
        self.model = QNet_Duel(input_dim,hidden_dim,output_dim)
        self.target_model = QNet_Duel(input_dim,hidden_dim,output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.copy_model()

    def copy_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, weights, experiences):
        weights = weights
        state, action, reward, next_state, done = experiences

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        #action = torch.unsqueeze(action, -1)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.long)
        weights = torch.tensor(weights, dtype=torch.float)

        Q_value = self.model(state).gather(-1, action)
        
        Q_value_next_index = self.model(next_state).detach().max(-1)[1]
        Q_value_next_index = torch.unsqueeze(Q_value_next_index, -1)
        Q_value_next_target = self.target_model(next_state).detach()
        Q_value_next = Q_value_next_target.gather(-1, Q_value_next_index)

        target =  (reward + self.gamma * Q_value_next * (1 - done))
        td_error = Q_value - target
        
        #loss = (weights * td_error).pow(2).mean()
        loss = self.criterion(Q_value,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        priorities = np.abs(td_error.detach().numpy())
        
        return priorities 

class Agent:
    def __init__(self,state_space, action_space, hidden_dim = 16, max_explore=1000, gamma = 0.9,
                max_memory=5000, lr=0.001):
        self.max_explore = max_explore 
        self.PRB = PrioritizedReplayBuffer(max_samples=max_memory)
        self.nS = state_space  
        self.nA = action_space  
        self.step = 0
        self.n_game = 0
        self.trainer = QTrainer(lr, gamma, self.nS, hidden_dim,self.nA)

    def remember(self, state, action, reward, next_state, done):
        self.PRB.store((state, action, reward, next_state, done)) 

    def train_long_memory(self,batch_size):
        if self.PRB.n_entries > batch_size:
            idxs, weights, experiences = self.PRB.sample(batch_size)
            priorities = self.trainer.train_step(weights,experiences)
            self.PRB.update(idxs, priorities)


    def get_action(self, state, n_game):
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.trainer.model(state).detach().numpy().squeeze()
        
        epsilon = self.max_explore - n_game
        if random.randint(0, self.max_explore) < epsilon:
            final_move = np.random.randint(len(prediction))
        else:
            final_move = prediction.argmax()
        return final_move


    @staticmethod
    def one_hot(x,size):
        result = np.zeros(size)
        result[x] = 1
        return result 
    
def train(env, max_game=5000, max_step=100):
    nS = env.observation_space.n
    agent = Agent(state_space = env.observation_space.n, 
                action_space = env.action_space.n,
                hidden_dim=16,
                max_explore=1000, gamma = 0.9,
                max_memory=50000, lr=0.0005)
    results = []
    state_new, _ = env.reset()
    state_new = Agent.one_hot(state_new,nS)
    done = False
    total_step = 0
    while agent.n_game <= max_game:
        state_old = state_new
        action = agent.get_action(state_old, agent.n_game)
        state_new, reward, done, _, _ = env.step(action)
        state_new = Agent.one_hot(state_new,nS)
        agent.remember(state_old, action, reward, state_new, done)
        agent.train_long_memory(batch_size=256)
        agent.step += 1
        total_step += 1

        if total_step % 10 == 0:
            agent.trainer.copy_model()

        if done or agent.step>max_step:
            results.append(reward>0)
            state_new, _ = env.reset()
            state_new = Agent.one_hot(state_new,nS)
            agent.step = 0
            agent.n_game += 1

            if (agent.n_game>0) and (agent.n_game % 200 ==0):         
                print("Running episode  {}, step {} Reaches goal {:.2f}%. ".format(
                    agent.n_game, total_step,np.sum(results[-100:])))


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1',map_name="8x8")
    train(env, 5000)