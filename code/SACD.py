############################### Import libraries ###############################
# code reference: toshikwa/sac-discrete.pytorch-PPO


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


def init_SACD_agent(state_dim, action_dim, action_max, device, belief_dim=None, summary_model=None, \
                    lr_actor=1e-4, lr_critic=2e-4, lr_summary=5e-5, gamma = 0.95, entropy_regularizer = 0.03):
    state_dim = state_dim  # the dimension of the state space
    action_dim = action_dim
    action_max = action_max

    if belief_dim is not None:
        state_dim = belief_dim
        
    sacd_agent = SACD(
        state_dim=state_dim,
        action_dim=action_dim,
        action_max=action_max,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        entropy_regularizer=entropy_regularizer,
        device=device,
        summary_model=summary_model,
        lr_summary=lr_summary
    )

    return sacd_agent


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_max, inter_dim=256):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(state_dim, inter_dim),
            nn.Tanh(),
            nn.Linear(inter_dim, inter_dim),
            nn.Tanh(),
            nn.Linear(inter_dim, action_max)
        )
        
    def forward(self, states):
        return self.head(states)
        
class DoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_max, inter_dim=256):
        super().__init__()
        self.Q1 = QNetwork(state_dim, action_max, inter_dim)
        self.Q2 = QNetwork(state_dim, action_max, inter_dim)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2
        
def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, device):
        """
        action_dim for continuous action space
        action_max for discrete action space
        """
        super(ActorCritic, self).__init__()

        self.device = device
        self.state_dim = state_dim

        self.action_dim = action_dim
        self.action_max = action_max
        
        # actor
        inter_dim = 256

        self.actor = nn.Sequential(
            nn.Linear(state_dim, inter_dim),
            nn.Tanh(),
            nn.Linear(inter_dim, inter_dim),
            nn.Tanh(),
            nn.Linear(inter_dim, action_max),  # for each action_dim, there are action_max choices
            nn.Softmax(dim=-1)
        )

        # critic
        self.online_critic = DoubleQNetwork(state_dim, action_max, inter_dim)
        self.target_critic = DoubleQNetwork(state_dim, action_max, inter_dim)
        
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        disable_gradients(self.target_critic)

    def forward(self):
        raise NotImplementedError

        
    def act(self, state, explore=False):
        """
        :return:
            action: [action_dim, ]
            action_logprob: []
            state_val: []
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        if( explore ):
            actions = dist.sample().view(-1, self.action_dim)  # []
            z = (action_probs == 0.0).float() * 1e-8
            log_action_probs = torch.log(action_probs + z)
            dist_entropy = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        else:
            actions = torch.argmax(action_probs, dim=-1, keepdim=True)
            dist_entropy = dist.entropy()
            
        return actions, action_probs, dist_entropy

    def act(self, state, explore=False):
        """
        :return:
            action: [action_dim, ]
            action_logprob: []
            state_val: []
        """

        if( explore ):
            action_probs = self.actor(state)
            action_dist = Categorical(action_probs)
            actions = action_dist.sample().view(-1, 1)
            
            # Avoid numerical instability.
            z = (action_probs == 0.0).float() * 1e-8
            log_action_probs = torch.log(action_probs + z)            
            return actions, action_probs, log_action_probs
        
        else:
            action_logits = self.actor(state)
            greedy_actions = torch.argmax(action_logits, dim=-1, keepdim=True)
            # print(action_logits, greedy_actions)
            return greedy_actions

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)            
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones, alpha, gamma=1.0):      
        with torch.no_grad():
            _, action_probs, log_action_probs = self.act(next_states, True)
            
            next_q1, next_q2 = self.target_critic(next_states) 
            
            next_q = (action_probs * torch.min(next_q1, next_q2)).sum(dim=1, keepdim=True)
            # next_q, _ = (torch.min(next_q1, next_q2)).max(dim=1, keepdim=True)
        
        # print(rewards.shape, next_q.shape)
        assert rewards.shape == next_q.shape
        return rewards + gamma * (1.0 - dones) * 1.0 * next_q

    def calc_critic_loss(self, states, actions, rewards, next_states, dones, weights, alpha, gamma=1.0):
        curr_q1, curr_q2 = self.calc_current_q(states, actions, rewards, next_states, dones)
        target_q = self.calc_target_q(states, actions, rewards, next_states, dones, alpha, gamma)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, states, actions, rewards, next_states, dones, weights, alpha):
        # (Log of) probabilities to calculate expectations of Q and entropies.
        actions, action_probs, log_action_probs = self.act(states.detach(), True)
        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(q * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - alpha * entropies)).mean()
                
        return policy_loss, entropies


class SACD:
    def __init__(
            self, state_dim, action_dim, action_max, lr_actor, lr_critic, gamma, entropy_regularizer,
            device=torch.device('cpu'), summary_model=None, lr_summary=-1
    ):
        """
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param action_max: maximum value of action, if action is discrete integer
        :param lr_actor: learning rate for the actor
        :param lr_critic: learning rate for the critic
        :param gamma: discount factor
        :param device: cpu or cuda
        :param summary_model: model to be used for summary. If not None, and lr_summary > 0, summary_model will be updated
        :param lr_summary: learning rate for the summary model
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        print("Device set to : " + str(device))

        self.gamma = gamma
        self.target_entropy = -np.log(1.0 / action_max) * 0.98
        
        self.policy = ActorCritic(state_dim, action_dim, action_max, self.device).to \
            (self.device)
        
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        if summary_model is not None and lr_summary > 0:
            self.q_optimizer = torch.optim.Adam([
                {'params': summary_model.parameters(), 'lr': lr_summary, 'weight_decay': 1e-6},
                {'params': self.policy.online_critic.Q1.parameters(), 'lr': lr_critic, 'weight_decay': 1e-8},
                {'params': self.policy.online_critic.Q2.parameters(), 'lr': lr_critic, 'weight_decay': 1e-8}
            ])
            self.lr_summary = lr_summary
            self.summary_model = summary_model
            self.update_summary = True
        else:
            self.q_optimizer = torch.optim.Adam([
                {'params': self.policy.online_critic.Q1.parameters(), 'lr': lr_critic, 'weight_decay': 1e-8},
                {'params': self.policy.online_critic.Q2.parameters(), 'lr': lr_critic, 'weight_decay': 1e-8}
            ])
            self.update_summary = False

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'weight_decay': 1e-8}
        ])
        
        # We optimize log(alpha), instead of alpha.
        self.alpha = entropy_regularizer
        self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        
    def update_target(self):
        self.policy.target_critic.load_state_dict(self.policy.online_critic.state_dict())
        
    def select_action(self, state, explore=True):
        """
        :param state: shape [state_dim]
        returned: action: shape [action_dim]
                    ??
        """
        return self.policy.act(state, explore)
    

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.policy.calc_critic_loss(states, actions, rewards, next_states, \
                                                                                  dones, 1, self.alpha, self.gamma)

        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()
        self.q_optimizer.zero_grad()
  
        policy_loss, entropies = self.policy.calc_policy_loss(states, actions, rewards, next_states, dones, 1, self.alpha)

        (policy_loss).backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        
    def save_dict(self):
        return self.policy.state_dict()

    def load_dict(self, loaded_dict):
        self.policy.load_state_dict(loaded_dict)
        