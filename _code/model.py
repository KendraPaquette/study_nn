import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nano_gpt2 import GPT2Model, GPT2Config, LayerNorm

def build_model(embed_dim, state_dim, action_dim, feature_dim, action_max, y_max, model_family='gru', device=None):
    model = Summary_Network(state_dim, action_dim, feature_dim, action_max, y_max, n_embd=embed_dim, model_family=model_family)
    return model


def get_configuration(n_embd, n_layer, n_head, model_family='gru'):
    if model_family == 'gpt2':
        configuration = GPT2Config()
        configuration.block_size = 100
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.n_embd = n_embd
        configuration.dropout = 0.0
        configuration.bias = True
        configuration.dropout = 0.

    elif model_family == 'gru':
        configuration = {}
        configuration['input_size'] = n_embd
        configuration['hidden_size'] = n_embd
        configuration['num_layers'] = n_layer
        configuration['bias'] = True
        configuration['batch_first'] = True
        configuration['dropout'] = 0.0
        
    return configuration

def get_input_layers(n_embd, state_dim, action_max):       
    state_embd_layer = nn.Linear(state_dim, n_embd)
    action_embd_layer = nn.Embedding(action_max, n_embd)

    return state_embd_layer, action_embd_layer

def input_forward(state_embd_layer, action_embd_layer, state, action):
    state_embeds = state_embd_layer(state)  # [B, n, d_state] -> [B, n, d]
    action_embeds = action_embd_layer(action.int()[:, :, 0])  # [B, n, 1] -> [B, n] -> [B, n, d]

    return state_embeds, action_embeds

def get_output_layers(n_embd, y_max, feature_dim):
    to_ys = nn.Linear(n_embd, y_max)
    to_feature = nn.Linear(n_embd, feature_dim)
    return to_ys, to_feature




class Prediction_Network(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.inSize = inSize
        self.outSize = outSize

        self.lin1 = nn.Linear(inSize, 256)
        self.lin2 = nn.Linear(256, outSize)

        self.activate = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.activate(self.lin1(x))
        x = self.activate(self.lin2(x))
        
        return x, None

    
class Summary_Network(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, action_max, y_max, n_embd=128, n_layer=1, n_head=4, model_family='gru'):
        """
        """
        super(Summary_Network, self).__init__()
        
        self.model_family = model_family
        self.configuration = get_configuration(n_embd, n_layer, n_head, model_family=model_family)
        self.predict_configuration = get_configuration(n_embd, n_layer, n_head, model_family='gru')
        
        self.n_embd = n_embd  # d
        self.n_layer = n_layer

        # initialize the transformer backbone
        # self._summarize_backbone = GPT2Model(self.configuration)
        self._summarize_backbone = self._get_backbone()
        
        self._prediction_backbone = Prediction_Network(n_embd, n_embd)
        # self._next_word_backbone = nn.GRU(**self.predict_configuration)

        # initialize the state, action, reward embedding layers
        self.state_embd, self.action_embd = get_input_layers(n_embd, state_dim, action_max)
        self.y_embd, self.feature_embd = get_output_layers(n_embd, y_max, feature_dim)

        
    def _get_backbone(self):
        if self.model_family == 'gpt2':
            return GPT2Model(self.configuration)
        elif self.model_family == 'gru':
            return nn.GRU(**self.configuration)
        
        raise NotImplementedError

    def _backbone_forward(self, backbone_model, input_embeds, kv_cache_list=None):
        f_output, _ = backbone_model(input_embeds)
        return f_output

    def _combine(self, state, action, mode='as'):
        """
        everything should be in shape [B, n, d]
        :return: shape [B, 2n, d]
        """
        B, n, d = state.shape
        if mode == 'sa':
            zs = torch.stack((state, action), dim=2)
        elif mode == 'as':
            zs = torch.stack((action, state), dim=2)
        else:
            raise NotImplementedError

        return zs.view(B, 2 * n, d)
    
    def forward(
            self, prefix_state, prefix_action, inits=None, inita=None, output_embedding = False
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
        B, n, d = prefix_state.shape

        e_prefix_state, e_prefix_action = input_forward(
            self.state_embd, self.action_embd, prefix_state, prefix_action)  # all [B, n, D]

        prefix_embeds = self._combine(e_prefix_state, e_prefix_action)
        f_summary = self._backbone_forward(self._summarize_backbone, prefix_embeds)
        if( output_embedding ):
            return f_summary[:, -1]
        
        return f_summary[:, 1::2]

    def predict_forward(self, beliefs):
        belief_vecs = torch.flatten(beliefs, start_dim = 1).view(-1, self.n_embd) # B * n, n_embd
        f_output = self._backbone_forward(self._prediction_backbone, belief_vecs)
        return self.y_embd(f_output), self.feature_embd(f_output)