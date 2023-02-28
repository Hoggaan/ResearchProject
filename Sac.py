import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque
import random

from ProjectCode import MultiRobotEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, act_dim)
        self.log_std_fc = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Constrain the logarithm of the standard deviation to a reasonable range
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.randn(mean.shape)
        action = torch.tanh(mean + std * normal)
        log_prob = self.compute_log_prob(mean, log_std, action)
        return action, log_prob

    def compute_log_prob(self, mean, log_std, action):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim, buffer_size, batch_size, gamma, tau, alpha, lr):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.q1 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)

        self.q2 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        self.target_q1 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    def update(self, batch):
        obs, acts, rews, next_obs, dones = batch

        obs = torch.FloatTensor(obs).to(self.device)
        acts = torch.FloatTensor(acts).to(self.device)
        rews = torch.FloatTensor(rews).to(self.device).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        next_acts, next_log_probs = self.actor.sample(next_obs)
        target_q1_values = self.target_q1(next_obs, next_acts)
        target_q2_values = self.target_q2(next_obs, next_acts)
        target_q_values = torch.min(target_q1_values, target_q2_values) - self.alpha * next_log_probs
        target_q_values = rews + (1 - dones) * self.gamma * target_q_values

        q1_values = self.q1(obs, acts)
        q2_values = self.q2(obs, acts)
        q1_loss = F.mse_loss(q1_values, target_q_values)
        q2_loss = F.mse_loss(q2_values, target_q_values)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        sampled_acts, log_probs = self.actor.sample(obs)
        min_q_values = torch.min(self.q1(obs, sampled_acts), self.q2(obs, sampled_acts))
        actor_loss = (self.alpha * log_probs - min_q_values).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs)
        return action.cpu().numpy()[0]

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic1.state_dict(), '%s/%s_critic1.pth' % (directory, filename))
        torch.save(self.critic2.state_dict(), '%s/%s_critic2.pth' % (directory, filename))
    
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic1.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, filename)))
        self.critic2.load_state_dict(torch.load('%s/%s_critic2.pth' % (directory, filename)))


# Define SACAgent class and its methods here (same as before)

if __name__ == "__main__":

    # Set hyperparameters
    obs_dim = 4 # observation space for each robot
    act_dim = 2 # action space
    hidden_dim = 256
    buffer_size = int(1e6)
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    lr = 3e-4
    updates_per_step = 1
    start_steps = 10000

    # Create the environment
    env = MultiRobotEnv()

    # Create the agent
    agent = SACAgent(obs_dim, act_dim, hidden_dim, buffer_size, batch_size, gamma, tau, alpha, lr)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    # Create a list to store episode rewards
    episode_rewards = []

    # Set the initial state
    obs = env.reset()

    # Start the training loop
    for t in range(1000000):
        # Sample actions from the agent
        actions = []
        for i in range(len(obs)):
            action = agent.select_action(obs[i], evaluate=False)
            actions.append(action)
        actions = np.array(actions)
        
        # Take a step in the environment
        obs_list, rewards, dones, _ = env.step(actions)
        
        # Store the experience in the replay buffer
        for i in range(len(obs)):
            replay_buffer.push(obs[i], actions[i], rewards[i], obs_list[i], dones[i])
            
        # Update the state
        obs = obs_list
        # Update the agent if enough samples are available
        if len(replay_buffer) > batch_size:
            for i in range(updates_per_step):
                batch = random.sample(replay_buffer, batch_size)
                agent.update(batch)
                
        # Store the episode rewards
        for r in rewards:
            episode_rewards.append(r)
            
        # If the episode is over, reset the environment
        if all(dones):
            obs = env.reset()
            
        # Print the current episode reward and save the model weights
        if t % 1000 == 0:
            print("Episode reward: ", np.mean(episode_rewards[-1000:]))
            agent.save()