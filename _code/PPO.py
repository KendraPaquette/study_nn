############################### Import libraries ###############################
# code reference: Pytorch-PPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


def init_SACD_agent(state_dim, action_dim, action_max, device, embed_dim = -1, summary_model=None, \
                    lr_actor=1e-4, lr_critic=2e-4, lr_summary=-1, gamma = 0.99, weight_decay=1e-3, entropy_regularizer = 0.03):
    state_dim = state_dim  # the dimension of the state space
    action_dim = action_dim
    action_max = action_max
    
    if( embed_dim > 0 ):
        state_dim = embed_dim

    sacd_agent = SACD(
        state_dim=state_dim,
        action_dim=action_dim,
        action_max=action_max,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        weight_decay=weight_decay,
        entropy_regularizer=entropy_regularizer,
        device=device
    )

    return sacd_agent



################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, has_continuous_action_space, action_std_init, device):
        """
        action_dim for continuous action space
        action_max for discrete action space
        """
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        self.state_dim = state_dim

        if has_continuous_action_space:
            # we don't need to care about the action_max in this case
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        else:
            self.action_dim = action_dim
            self.action_max = action_max

        PPO_inter_dim = 128
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, PPO_inter_dim),
                nn.Tanh(),
                nn.Linear(PPO_inter_dim, PPO_inter_dim),
                nn.Tanh(),
                nn.Linear(PPO_inter_dim, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, PPO_inter_dim),
                nn.Tanh(),
                nn.Linear(PPO_inter_dim, PPO_inter_dim),
                nn.Tanh(),
                nn.Linear(PPO_inter_dim, action_max),  # for each action_dim, there are action_max choices
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, PPO_inter_dim),
            nn.Tanh(),
            nn.Linear(PPO_inter_dim, PPO_inter_dim),
            nn.Tanh(),
            nn.Linear(PPO_inter_dim, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        :return:
            action: [action_dim, ]
            action_logprob: []
            state_val: []
        """
        
        state = state.view(-1, self.state_dim)
        if self.has_continuous_action_space:
            action_mean = self.actor(state.float())
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state.float())  # [action_dim ^ action_max, ]
            dist = Categorical(action_probs)  # a multinomial distribution

        action = dist.sample()  # []
        action_logprob = dist.log_prob(action)  # []
        state_val = self.critic(state.float())[:,0]  # []

        return action.detach().view(-1, self.action_dim), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        :param state: shape [n, state_dim]
        :param action: shape [n, action_dim] for continuous action space
                       shape [n, ] for discrete action space
        :return:
            action_logprobs: [n,]
            state_values: [n,]
            dist_entropy: [n,]
        """

        state = state.view(-1, self.state_dim)
        if self.has_continuous_action_space:
            action_mean = self.actor(state.float())  # [n, action_dim]
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state.float())  # [n, action_max]
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)  # shape [n,]
        dist_entropy = dist.entropy()  # shape [n,]
        state_values = self.critic(state.float())[:, 0]  # [n,], choose slice 0 because critic network output dimension 1

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
            self, state_dim, action_dim, action_max, lr_actor, lr_critic, gamma=0.99, K_epochs=40, eps_clip=0.2, entropy_regularizer=0.03,
            device=torch.device('cpu'), weight_decay=1e-3
    ):
        """
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param action_max: maximum value of action, if action is discrete integer
        :param lr_actor: learning rate for the actor
        :param lr_critic: learning rate for the critic
        :param gamma: discount factor
        :param K_epochs: number of epochs per update
        :param eps_clip: clip parameter for PPO
        :param has_continuous_action_space: is action space continuous or discrete
        :param action_std_init: initial std for continuous action space
        :param device: cpu or cuda
        :param summary_model: model to be used for summary. If not None, and lr_summary > 0, summary_model will be updated
        :param lr_summary: learning rate for the summary model
        """

        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        # print("Device set to : " + str(torch.cuda.get_device_name(device)))
        # print("Device set to : " + str(device.type))
        print("Device set to : " + str(device))

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.entropy_regularizer = entropy_regularizer
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_max, has_continuous_action_space, action_std_init, self.device).to \
            (self.device)
        
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        self.q_optimizer = torch.optim.Adam([
            {'params': self.state_embedder.parameters(), 'lr': lr_critic, 'weight_decay': weight_decay},
            {'params': self.policy.online_critic.Q1.parameters(), 'lr': lr_critic, 'weight_decay': weight_decay},
            {'params': self.policy.online_critic.Q2.parameters(), 'lr': lr_critic, 'weight_decay': weight_decay}
        ])
        self.update_summary = False
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'weight_decay': weight_decay}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_max, has_continuous_action_space, action_std_init,
                                      self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def add_to_buffer(self, last_state, last_action, last_logprob, last_state_val, reward, done):
        self.buffer.states.extend(last_state)
        self.buffer.actions.extend(last_action)
        self.buffer.logprobs.extend(last_logprob)
        self.buffer.state_values.extend(last_state_val)

        self.buffer.rewards.extend(reward)
        self.buffer.is_terminals.extend(done)
        
    def select_action(self, state):
        """
        :param state: shape [state_dim]
        returned: action: shape [action_dim]
                  action_logprob: shape []
                  state_val: shape []
        """
        if self.has_continuous_action_space:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            # import ipdb; ipdb.set_trace()
#             self.last_state = state
#             self.last_action = action
#             self.last_logprob = action_logprob
#             self.last_state_val = state_val

            # action = action.detach().flatten()

        else:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

#             self.last_state = state
#             self.last_action = action
#             self.last_logprob = action_logprob
#             self.last_state_val = state_val

        return action, action_logprob, state_val

    def update_base(self):
        if (len(self.buffer.states) == 0):
            # if there is nothing stored in the buffer
            return

        # reset optimizer internal state periodically

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Normalizing the rewards
        rewards = torch.stack(rewards)  # [n, reward_dim]
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = torch.squeeze(rewards)  # [n,]

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).to(self.device)  # [n, state_dim]
        old_states = old_states.view(-1, self.state_dim)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        if self.has_continuous_action_space:
            old_actions = old_actions.view(-1, self.action_dim)
        # discrete: [n,], continuous: [n, action_dim]
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)  # [n,]
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(self.device)  # [n]
        if not self.update_summary:
            old_states = old_states.detach()

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()  # [n]
        return old_states, old_actions, old_logprobs, old_state_values, rewards, advantages

    def update(self):
        """
        If set `optimizer_step` to False, then we will just take the loss.backward(), but don't do the optimizer.step()
        This is for the sake of delayed update in order to mimic larger batch size
        which is a problem for tf_end2end training.

        If we wait X steps before doing optimizer.step(), then it is equivalent to having batch size of X times larger.
        sadly the gradient is calculated to be averaged over the current batch, and summed over multiple backward().
        So for a proper scale of gradient, we need to divide the gradient by X.
        This is what the normalize_factor is used for.
        If we do optimizer.step() every X batch, we will use normalize_factor=X.
        """
        if (len(self.buffer.states) == 0):
            # if there is nothing stored in the buffer
            return

        old_states, old_actions, old_logprobs, old_state_values, rewards, advantages = self.update_base()
        # Optimize policy for K epochs
        for i_K in range(self.K_epochs):  # self.K_epochs
            # Evaluating old actions and values
            # old_states: [n, d]
            # old_actions: [n,] or [n, action_dim]
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # logprobs shape [n,]
            # state_values shape [n]
            # dist_entropy shape [n,]

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())  # [n,]

            # Finding Surrogate Loss
            surr1 = ratios * advantages  # [n,] * [n,] -> [n,]
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # [n,]

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_regularizer * dist_entropy
            # torch.min(surr1, surr2) shape: [n,]
            # self.MseLoss(state_values, rewards) shape: []
            # dist_entropy shape [n,]
            # loss shape: [n,]

            # take gradient step
            # self.optimizer.zero_grad()
            # (loss.mean() / normalize_factor).backward(retain_graph=True)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def clear(self):
        self.buffer.clear()


