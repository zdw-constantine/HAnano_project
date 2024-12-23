import os
import sys
import torch
from torch import nn

import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from model.modle_attention import  AttentionLayer
from model.model_cnn import CNNLayer
from utils.classes import BaseModelImpl

from  utils.bonito import BonitoLSTM





class Basedmodel(BaseModelImpl):
    
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, *args, **kwargs):
        super(Basedmodel, self).__init__(*args, **kwargs)
        
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        
        if load_default:
            self.load_default_configuration()

    

    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 324, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 324,
            'encoder_output_size': 324,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        
        self.cnn_stride = self.get_defaults()['cnn_stride']
        
        self.decoder = self.build_decoder(encoder_output_size = 384, decoder_type = 'crf')
        self.decoder_type = 'crf'




class HAnano(Basedmodel,BaseModelImpl):

    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, *args, **kwargs):
        super(HAnano, self).__init__(decoder = 'crf', *args, **kwargs)

        use_connector=True
        self.cnn_type = "causalcall"
        self.encoder_type = "lstm5"
        self.decoder_type = "crf"
        self.use_connector = use_connector
        self.use_convolution_connector = False

        self.cnn_stride = None
        self.convolution = self.build_cnn()
        self.encoder = self.build_encoder()
        self.dropout=nn.Dropout1d(p=0.1)
        if use_connector:
            self.connector = self.build_connector()
            if self.cnn_stride == 1:
                self.convolution_connector = self.build_convolution_connector()
                self.use_convolution_connector = True
                
        self.decoder = self.build_decoder()
        self.TAA=AttentionLayer(K=18, d=18, kernel_size=3)

    def forward(self, x):
        
        # [batch, channels, len]
        x = self.convolution(x)
        #print(x.shape)
        x=self.dropout(x)
        x=self.TAA(x)
        x=self.dropout(x)
        if self.use_convolution_connector:
            x = self.convolution_connector(x)

        # [batch, channels, len]
        if self.use_connector:
            x = x.permute(0, 2, 1)
            # [batch, len, channels]
            x = self.connector(x)
            x = x.permute(0, 2, 1)
            # [batch, channels, len]
        #print(x.shape)
        

        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.encoder(x)

        # get rid of RNN hidden states
        if isinstance(x, tuple):
            x = x[0]
        x = self.decoder(x)
        return x

    def build_cnn(self):

        defaults = Basedmodel.get_defaults(self)
        cnn = CNNLayer()
        
        self.cnn_output_size = defaults['cnn_output_size']
        self.cnn_output_activation = defaults['cnn_output_activation']
        self.cnn_stride = defaults['cnn_stride']

        return cnn

    def build_encoder(self):

        
        defaults = {'encoder_input_size': 256, 'encoder_output_size': 512}
        num_layers = int(list(self.encoder_type)[-1])
        if self.use_connector:
            input_size = defaults['encoder_input_size']
        else:
            input_size = self.cnn_output_size

        encoder =  nn.LSTM(input_size = input_size, hidden_size = 256, num_layers = 3, bidirectional = True)
        
        

        self.encoder_input_size = defaults['encoder_input_size']
        self.encoder_output_size = defaults['encoder_output_size']
        return encoder

    def build_connector(self):
        if self.cnn_output_activation == 'relu':

            return nn.Sequential(nn.Linear(self.cnn_output_size, self.encoder_input_size), nn.ReLU())
        elif self.cnn_output_activation == 'silu':
            return nn.Sequential(nn.Linear(self.cnn_output_size, self.encoder_input_size), nn.SiLU())
        elif self.cnn_output_activation is None:
            return nn.Sequential(nn.Linear(self.cnn_output_size, self.encoder_input_size))

    def build_convolution_connector(self):
        self.cnn_stride = 5
        if self.cnn_output_activation == 'relu':
            return nn.Sequential(
                nn.Conv1d(
                    self.cnn_output_size, 
                    self.cnn_output_size,
                    kernel_size = 19,
                    stride = 5,
                    padding = 19//2,
                ), 
                nn.ReLU()
            )
        elif self.cnn_output_activation == 'silu':
            return nn.Sequential(
                nn.Conv1d(
                    self.cnn_output_size, 
                    self.cnn_output_size,
                    kernel_size = 19,
                    stride = 5,
                    padding = 19//2,
                ), 
                nn.SiLU()
            )
        elif self.cnn_output_activation is None:
            return nn.Sequential(
                nn.Conv1d(
                    self.cnn_output_size, 
                    self.cnn_output_size,
                    kernel_size = 19,
                    stride = 5,
                    padding = 19//2,
                ), 
            )
        
                

    def build_decoder(self):
        return BaseModelImpl.build_decoder(self, encoder_output_size = self.encoder_output_size, decoder_type = self.decoder_type)     