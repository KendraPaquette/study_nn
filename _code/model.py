import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nano_gpt2 import GPT2Model, GPT2Config, LayerNorm

def build_model(embed_dim, state_dim, action_dim, feature_dim, action_max, y_max, model_family='gru', device=None):
    model = Summary_Network(state_dim, action_dim, feature_dim, action_max, y_max, n_embd=embed_dim)
    return model


def get_configuration(n_embd, n_layer, n_head):
    configuration = {}
    configuration['input_size'] = n_embd
    configuration['hidden_size'] = n_embd
    configuration['num_layers'] = n_layer
    configuration['bias'] = True
    configuration['batch_first'] = True
    configuration['dropout'] = 0.0

    return configuration


class Prediction_Network(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.inSize = inSize
        self.outSize = outSize

        self.lin1 = nn.Linear(inSize, 256)
        self.lin2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, outSize)

        self.activate = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.activate(self.lin1(x))
        x = self.activate(self.lin2(x))
        return self.output(x)
        

class Summary_Network(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, action_max, y_max, n_embd=128):
        """
        """
        super(Summary_Network, self).__init__()
        self.n_embd = n_embd  # d
        self.embed = nn.Linear(state_dim, n_embd)
        
        # initialize the transformer backbone
        self.y_predict = Prediction_Network(n_embd, y_max)
        self.f_predict = Prediction_Network(n_embd, feature_dim)
        
    def forward(
            self, X
    ):
        """
        :param prefix_state: [B, n, d]
        :param prefix_action: [B, n] or [B, n-1]
        :param prefix_reward: [B, n] or [B, n-1]
        :param state: [B, n_append, d]
        :param action: [B, n_append]
        :param reward: [B, n_append]
        :return:
        """
        # process the prefix data
        return self.embed(X)
    
    def predict_forward(self, embed):
        return self.y_predict(embed), self.f_predict(embed)
