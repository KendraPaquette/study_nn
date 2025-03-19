import numpy as np
import torch

# Reward set? 
#1) -1.5, 0.1 / -0.3, 0.2
#2) -1.2, 0.03 / -0.2, 0.05
#3) -1, 0.02 / -0.2, 0.05

sliding_window = [12, 24, 48, 72, 168]

class Env:
    def __init__(self):
        self.last_probed = 0
        self.last_lapse = 0
        self.cur_env = None
        
    def reset(self, cur_env):
        self.t = 0
        self.cur_env = cur_env
        while(self.cur_env[self.t]['type'] == 'query'):
            self.t += 1
        
        self.finished = False
        self.hidden_lapsed = False
        self.last_lapse = 0
        self.last_probed = 0

        # sliding window starting point
        self.js = [0,0,0,0,0]
        self.all_measured = []
        self.all_measured_time = []
        
        # immediate return
        self.dummy_obs = self.cur_env[self.t]['obs'] * 0
        return self.dummy_obs, self.cur_env[self.t]['time']
        
    def step(self, action, probe_penalty=0.05):
        if( self.finished ):
            return self.dummy_obs, 0, 1, -1
        
        reward = 0.
        done = 0
        cur_time = self.cur_env[self.t]['time']
        
        action = action.item()
        if( self.cur_env[self.t]['type'] == 'survey' ):
            obs = self.cur_env[self.t]['obs'] 
            if( action % 2 == 0 ):
                # doesn't reveal the survey
                # read if ema = 1
                if( obs[0] > 0 ):
                    self.hidden_lapsed = True

            else:
                reward -= probe_penalty
                ret_obs = obs + 0.0
                # reveal the survey received in-between
                if( self.hidden_lapsed ):
                    ret_obs[0] = 1

                self.hidden_lapsed = False
                self.all_measured_time.append(cur_time)
                self.all_meausred.append(ret_obs)
                
        else:
            # query time: no survey
            ret_obs = self.dummy_obs
            y = self.cur_env[self.t]['out']
            # get reward depending on whether the guess was right at the query point
#             plist = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.9] #min(max(0.01, 1./9 * action), 0.99)
#             p = plist[action]
#             reward = (5*y * np.log(1-p) + (1 - 5*y) * np.log(p))

#             if( action == 0 ):
#                 # FN : high cost, TN : low reward
#                 reward = -1. if ( y > 0 ) else 0.02
#             else:
#                 # FP : low cost, TP : high reward
#                 reward = -0.1 if ( y == 0 ) else 0.1
        
        info = {'out':self.cur_env[self.t]['out'], 'feature': self.cur_env[self.t]['feature'], \
                'type': self.cur_env[self.t]['type'], 'next_type': 'end'}
                
        # manual feature engineering
        for k in range(5):
            while( self.js[k] < len(self.all_measured_time) and self.all_measured_time[self.js[k]] < cur_time - sliding_window[k] ):
                self.js[k] += 1

        tmp_measure = np.array(self.all_measured)
        raw1 = np.concatenate(([np.median(tmp_measure[self.js[k]:, 1:-5], axis=0) for k in range(5)], \
                    [np.min(tmp_measure[self.js[k]:, 1:-5], axis=0) for k in range(5)] , \
                    [np.max(tmp_measure[self.js[k]:, 1:-5], axis=0) for k in range(5)]), axis=1)
        X1 = np.concatenate(raw1)
        for k in range(2,5):
            if(len(self.all_measured[j[k]:][tmp_measure[j[k]:, -1]==1]) == 0):
                raw2 = np.zeros((9,), dtype=np.float32)
                X1 = np.concatenate((X1, raw2))
                continue
                
            j = self.js[k]
            raw2 = np.concatenate(([np.median(tmp_measure[j:, -5:-1][self.all_measured[j:, -1]==1], axis=0)], \
                    [np.min(tmp_measure[j:, -5:-1][tmp_measure[j:, -1]==1], axis=0)] , \
                    [np.max(tmp_measure[j:, -5:-1][tmp_measure[j:, -1]==1], axis=0)]), axis=1)
            
            X1 = np.concatenate((X1, np.concatenate(raw2)))
        
        sub_mean = np.mean(tmp_measure, axis=0)
        X = np.concatenate((X1, sub_mean))
        X = np.concatenate((X, self.all_measured[-1]))
        
        # End signal
        self.t += 1
        if( self.t == len(self.cur_env) ):
            done = 1
            self.finished = True
        else:
            info['next_type'] = self.cur_env[self.t]['type']
            
        return X, reward, done, info
        
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
        