import numpy as np
import torch
import time

# Reward set? 
#1) -1.5, 0.1 / -0.3, 0.2
#2) -1.2, 0.03 / -0.2, 0.05
#3) -1, 0.02 / -0.2, 0.05

sliding_window = [12, 24, 48, 72, 168]
def calculate_feature(tmp_measure, last_measured, js):
    X1 = None

    # daily survey
    all_feats = []
    j0 = js[-1]
    tp = tmp_measure[j0:, :7]
    for k in range(5):
        j = js[k] - j0
        if(j == tp.shape[0]):
            raw1 = np.zeros((22,), dtype=np.float32)
        else:
            dp = tp[j:]
            count = np.sqrt(dp.shape[0])       
            raw1 = np.concatenate((np.mean(dp, axis=0), np.min(dp, axis=0), np.max(dp, axis=0), [count]))

        all_feats.append(raw1)

    # morning survey
#     for k in range(2,5):
#         j = js[k]
#         if(j == tmp_measure.shape[0] or len(tmp_measure[j:][tmp_measure[j:, -1]==1]) == 0):
#             raw2 = np.zeros((10,), dtype=np.float32)

#         else:
#             dp = tmp_measure[j:, -4:-1][tmp_measure[j:, -1]==1]            
#             count = np.sqrt(dp.shape[0])
#             raw2 = np.concatenate((np.median(dp, axis=0), np.min(dp, axis=0), np.max(dp, axis=0), [count]))
            
#         all_feats.append(raw2)

    sub_mean = np.mean(tmp_measure, axis=0)
    # sub_std = np.std(tmp_measure, axis=0)
    
    X1 = np.concatenate(all_feats)
    X = np.concatenate((X1, sub_mean, last_measured))
    return X


class Env:
    def __init__(self, state_dim):
        self.last_probed = 0
        self.last_lapse = 0
        self.cur_env = None
        self.state_dim = state_dim
        
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
        self.js = [0, 0, 0, 0, 0]
        self.all_measured = []
        self.all_measured_time = []
        self.cumulated_measures = []
        
        # immediate return
        self.dummy_obs = np.zeros((self.state_dim,))
        return self.dummy_obs, self.cur_env[self.t]['time']
        
    def step(self, action, probe_penalty=0.05):
        if( self.finished ):
            return self.dummy_obs, self.dummy_obs, 0, 1, -1
        
        reward = 0.
        done = 0
        lapse = self.cur_env[self.t]['out']
        cur_time = self.cur_env[self.t]['time']
        
        action = action.item()
        obs = self.cur_env[self.t]['obs'] 

        if( self.cur_env[self.t]['type'] == 'survey' ):
            self.cumulated_measures.append(obs)
            # first 5 asks the most intense moment
            ret_obs = np.max(self.cumulated_measures, axis = 0)
            # since survey asks first / last lapse
            if( np.sum(self.cumulated_measures, axis=0)[0] > 1 ):
                ret_obs[0] += 1
            # this only relies on the current moment
            ret_obs[5:] = obs[5:] + 0.0

            if( action % 2 == 0 ):
                # doesn't reveal the survey
                pass

                # should add penalty about how different my prediction is? 
                # defer this to prediction network
            else:
                reward -= probe_penalty
                
                # reveal the survey received in-between
                self.cumulated_measures = []
                self.all_measured_time.append(cur_time)
                self.all_measured.append(ret_obs)
                
                
                
        else:
            # query time: no survey
            ret_obs = obs + 0.0
            
            # get reward depending on whether the guess was right at the query point            
            # defer the job to prediction network

        info = {'feature': ret_obs[:-4], 'lapse': lapse, 'type': self.cur_env[self.t]['type'], 'next_type': 'end'}
        
        # End signal
        self.t += 1
        if( self.t == len(self.cur_env) - 1 ):
            done = 1
            self.finished = True
        else:
            info['next_type'] = self.cur_env[self.t]['type']
            cur_time = self.cur_env[self.t]['time']
            
        # manual feature engineering
        for k in range(5):
            while( self.js[k] < len(self.all_measured_time) and self.all_measured_time[self.js[k]] < cur_time - sliding_window[k] ):
                self.js[k] += 1
                                
        # next X
        X = calculate_feature(np.array(self.all_measured), self.all_measured[-1], self.js)
        return X, self.cur_env[self.t]['fullX'], reward, done, info
        
class VecEnv:
    def __init__(self, env_list, train_index, test_index, num_process = 50, state_dim=154):     
        self.env_list = env_list
        self.default_num_process = num_process
        self.venv = []
        for i in range(num_process):
            self.venv.append(Env(state_dim))
            
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
        
        return torch.from_numpy(np.array(obs,dtype=np.float32)), info
        
    def step(self, actions):
        states, hstates, rewards, dones, infos = [], [], [], [], []         
        for i in range(self.num_process):
            env, action = self.venv[i], actions[i]
            obs, hobs, reward, done, info = env.step(action, self.probe_penalty)
            
            states.append(obs)
            hstates.append(hobs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        # return torch.tensor(states), torch.tensor(rewards), dones, infos
        return torch.from_numpy(np.array(states,dtype=np.float32)), torch.from_numpy(np.array(hstates,dtype=np.float32)), \
                torch.from_numpy(np.array(rewards, dtype=np.float32)), dones, infos
        