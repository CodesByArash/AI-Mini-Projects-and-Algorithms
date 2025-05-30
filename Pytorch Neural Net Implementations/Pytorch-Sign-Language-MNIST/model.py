import torch.nn as nn
import torch.nn.functional as F
from utils import *

class MLP(nn.Module):


    def __init__(self, units: list, hidden_layer_activation='relu', init_type=None , dropout=False):
        super(MLP, self).__init__()
        self.units = units
        self.n_layers = len(units) # including input and output layers

        #####################################################################################
        # use nn.Sequential() to stack layers in a for loop                                 #
        # It can be summarized as: ***[LINEAR -> ACTIVATION]*(L-1) -> LINEAR -> SOFTMAX***  #
        # Use nn.Linear() as fully connected layers                                         #
        #####################################################################################

        valid_activations = {'relu': nn.ReLU(),
                             'tanh': nn.Tanh(),
                             'sigmoid': nn.Sigmoid(),}

        if hidden_layer_activation is not None :
            self.activation = valid_activations[hidden_layer_activation]

        self.mlp = nn.Sequential()
        # # input layer creation

        if dropout:
            self.mlp.add_module(f'dropout{0}', nn.Dropout(p=0.1))
        self.mlp.add_module(f'ff{0}', nn.Linear(units[0],units[1]))

        # other layers creation
        out = units[1]
        for i in range(1, len(units)):
            if hidden_layer_activation is not None:
                self.mlp.add_module(f'activation{i-1}', self.activation)
            if dropout:
                self.mlp.add_module(f'dropout{i}', nn.Dropout(p=0.1))
            self.mlp.add_module(f'ff{i}', nn.Linear(out, units[i]))
            out = units[i]


        self.mlp.apply(lambda net:init_weights(net ,init_type=init_type))

        #for loop creates last layer but activation is softmax and will be applied in
        #the forward function

    def forward(self, X):
        #####################################################################################
        # First propagate the input and then apply a softmax layer                          #
        #####################################################################################
        out = self.mlp(X)

        return out

        

