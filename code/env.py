import numpy as np
import torch

class Env:
    def __init__(self):
        self.last_probed = 0
        self.last_lapse = 0
        
    def reset(self, cur_env):
        self.t = 0
        self.cur_env = cur_env
        while(self.cur_env[self.t]['type'] == 'query'):
            self.t += 1
        
        self.finished = False
        self.hidden_lapsed = False
        self.last_lapse = 0
        self.last_probed = 0

        self.dummy_obs = self.cur_env[self.t]['obs'] * 0
        return self.dummy_obs, self.cur_env[self.t]['time']
        
    def step(self, action, probe_penalty=0.05):
        if( self.finished ):
            return self.dummy_obs, 0, 1, -1
        
        revealed = False
        reward = 0.
        done = 0
        
        if( self.cur_env[self.t]['type'] == 'survey' ):
            obs = self.cur_env[self.t]['obs'] + 0.0 # + 0.00 not to overload through the pointer
            if( action == 0 ):
                # doesn't reveal the survey
                ret_obs = obs * 0
                # read ema = 1
                if( obs[0] > 0 ):
                    self.hidden_lapsed = True
            
            else:
                # reveal the survey received in-between
                if( self.hidden_lapsed ):
                    obs[0] = 1

                self.hidden_lapsed = False
                reward -= probe_penalty
                ret_obs = obs

            info = 'survey'
            
        else:
            # query time: no survey
            ret_obs = self.dummy_obs + 0.0
            info = {'out':self.cur_env[self.t]['out'], 'feature': self.cur_env[self.t]['feature']}
            y = info['out']
            # get reward depending on whether the guess was right
            if( action == 0 ):
                reward = -1.2 if ( y > 0 ) else 0.03
            else:
                reward = -0.2 if ( y == 0 ) else 0.05
            
        self.t += 1
        if( self.t == len(self.cur_env) ):
            done = 1
            self.finished = True
        else:
            # encode whether this is query or survey in the last coordinate
            ret_obs[-1] = 0 if(self.cur_env[self.t]['type'] == 'query') else 1

        return ret_obs, reward, done, info
        
class VecEnv:
    def __init__(self, env_list, train_index, test_index, num_process = 50):     
        self.env_list = env_list
        self.default_num_process = num_process
        self.venv = []
        for i in range(num_process):
            self.venv.append(Env())
            
        self.train_ids = train_index
        self.test_ids = test_index
        
        self.probe_penalty = 0.00
    
    def set_probe_penalty(self, penalty):
        self.probe_penalty = penalty

    def reset(self, mode = 'train', num_process = -1):
        num_process = self.default_num_process if( num_process < 0 ) else num_process 
        
        if( mode == 'train' ):
            self.m = list(np.random.choice(self.train_ids, num_process, replace=True))
        else:
            num_process = len(self.test_ids)
            self.m = list(self.test_ids)
            
        self.num_process = num_process
        
        obs,info = [], []
        for i in range(num_process):
            obs1, info1 = self.venv[i].reset(self.env_list[self.m[i]])
            obs.append(obs1)
            info.append(info1)            
        
        return torch.from_numpy(np.array(obs)), torch.from_numpy(np.array(info))
        
    def step(self, actions):
        states, rewards, dones, infos = [], [], [], []                
        for i in range(self.num_process):
            env, action = self.venv[i], actions[i]
            obs, reward, done, info = env.step(action, self.probe_penalty)
            
            states.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        # return torch.tensor(states), torch.tensor(rewards), dones, infos
        return torch.from_numpy(np.array(states)), torch.from_numpy(np.array(rewards)), dones, infos
        