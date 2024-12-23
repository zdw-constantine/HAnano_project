import os
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader
from abc import abstractmethod
import numpy as np
import random
from pathlib import Path
from fast_ctc_decode import beam_search, viterbi_search, crf_greedy_search, crf_beam_search
import uuid
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.tools import read_metadata, time_limit, TimeoutException
from utils.read import read_fast5
from utils.normalization import normalize_signal_from_read_data, med_mad
from utils.constants import CTC_BLANK, BASES_CRF, S2S_PAD, S2S_EOS, S2S_SOS, S2S_OUTPUT_CLASSES
from utils.constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE, CRF_N_BASE, BASES
from utils.constants import STICH_ALIGN_FUNCTION, STICH_GAP_OPEN_PENALTY, STICH_GAP_EXTEND_PENALTY, RECURRENT_DECODING_DICT, MATRIX

from utils.evaluation import alignment_accuracy, make_align_arr, elongate_cigar

from utils.bonito import CTC_CRF, BonitoLinearCRFDecoder

class BaseModel(nn.Module):
    """Abstract class for basecaller models

    It contains some basic methods: train, validate, predict, ctc_decode...
    Since most models follow a similar style.
    """
    
    def __init__(self, device, dataloader_train, dataloader_validation, 
                 optimizer = None, schedulers = dict(), criterions = dict(), clipping_value = 2, scaler = None, use_amp = False, use_sam = False):
        super(BaseModel, self).__init__()
        
        self.device = device
        
        # data
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation

        # optimization
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterions = criterions
        self.clipping_value = clipping_value
        self.scaler = scaler
        self.use_sam = use_sam
        if self.scaler is not None:
            self.use_amp = True
        else:
            self.use_amp = use_amp
        
        self.init_weights()
        self.stride = self.get_stride()
        self.dummy_batch = None
        
    @abstractmethod
    def forward(self, batch):
        """Forward through the network
        """
        raise NotImplementedError()
    
    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        y = batch['y'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            p = self.forward(x) # forward through the network
            # 400 * 16 * 1280
            loss, losses = self.calculate_loss(y, p)

        self.optimize(loss)
        
        return losses, p
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1) # add channels dimension
            y = batch['y'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                p = self.forward(x) # forward through the network
                _, losses = self.calculate_loss(y, p)
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                p = self.forward(x)
            
        return p
    
    @abstractmethod    
    def decode(self, p, greedy = True):
        """Abstract method that is used to call the decoding approach for
        evaluation metrics during training and evaluation. 
        For example, it can be as simple as argmax for class prediction.
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        raise NotImplementedError()

    @abstractmethod    
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels
            p (tensor): tensor with predictions
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        raise NotImplementedError()
        return loss, losses
    
    
    def optimize(self, loss):
        """Optimizes the model by calculating the loss and doing backpropagation
        
        Args:
            loss (float): calculated loss that can be backpropagated
        """
        
        if self.use_sam: 
            raise NotImplementedError()
            # TODO
            # it is tricky how to use this SAM thing (https://github.com/davda54/sam)
            # because we have to calculate the loss twice, so we have to find a way
            # to make this general
            # also, it is unclear where to put the gradient clipping

            # loss.backward()
            # self.optimizer.first_step(zero_grad=True)
            # loss, losses = self.calculate_loss(y, p)
            # self.optimizer.second_step(zero_grad=True)
        elif self.scaler is not None:

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()
            
        for scheduler in self.schedulers.values():
            if scheduler:
                scheduler.step()
            
        return None

    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [batch, len] in key 'y'
            predictions (list): list of predicted sequences as strings
        """
        y = batch['y'].cpu().numpy()
        y_list = self.dataloader_train.dataset.encoded_array_to_list_strings(y)
        accs = list()
        for i, sample in enumerate(y_list):
            accs.append(alignment_accuracy(sample, predictions[i]))
            
        return {'metric.accuracy': accs}
    
    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, checkpoint_file):
        """Save the model state
        """
        if self.scaler is not None:
            scaler_dict = self.scaler.state_dict()
            #scaler_dict = None
        else:
            scaler_dict = None
        #print(self.optimizer.state_dict())    
        save_dict = {'model_state': self.state_dict(), 
                     'optimizer_state': self.optimizer.state_dict(),
                     'scaler': scaler_dict}
        
        for k, v in self.schedulers.items():
            #print(v.state_dict())
            save_dict[k + '_state'] = v.state_dict()
        torch.save(save_dict, checkpoint_file)
    
    def load(self, checkpoint_file, initialize_lazy = True):
        """Load a model state from a checkpoint file

        Args:
            checkpoint_file (str): file were checkpoints are saved
            initialize_lazy (bool): to do a forward step before loading model,
                this solves the problem for layers that are initialized lazyly
        """

        if initialize_lazy:
            if self.dummy_batch is None:
                dummy_batch = {'x': torch.randn([16, 2000], device = self.device)}
            else:
                dummy_batch = self.dummy_batch
            self.predict_step(dummy_batch)

        checkpoints = torch.load(checkpoint_file)
        self.load_state_dict(checkpoints['model_state'],strict=False)
        #print(checkpoints['model_state'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoints['optimizer_state'])
            #print(checkpoints['optimizer_state'])
           
        if self.scaler is not None:
            self.optimizer.load_state_dict(checkpoints['scaler'])
            
            
        if 'lr_scheduler' in list(self.schedulers.keys()):
            self.schedulers['lr_scheduler'].load_state_dict(checkpoints['lr_scheduler_state'])
            

        
    def get_stride(self):
        """Gives the total stride of the model
        """
        return None
        
    @abstractmethod
    def load_default_configuration(self, default_all = False):
        """Method to load default model configuration
        """
        raise NotImplementedError()

class BaseModelCTC(BaseModel):
    
    def __init__(self, blank = CTC_BLANK, *args, **kwargs):
        """
        Args:   
            blank (int): class index for CTC blank
        """
        super(BaseModelCTC, self).__init__(*args, **kwargs)

        self.criterions['ctc'] = nn.CTCLoss(blank = blank, zero_infinity = True).to(self.device)

    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if not isinstance(p, np.ndarray):
            p = p.cpu().numpy()

        if greedy:
            return self.decode_ctc_greedy(p, *args, **kwargs)
        else:
            return self.decode_ctc_beamsearch(p, *args, **kwargs)

    def decode_ctc_greedy(self, p, qstring = False, qscale = 1.0, qbias = 1.0, collapse_repeats = True, return_path = False, *args, **kwargs):
        """Predict the bases in a greedy approach
        Args:
            p (tensor): [len, batch, classes]
            qstring (bool): whether to return the phredq scores
            qscale (float)
            qbias (float)
        """
        
        alphabet = BASES_CRF
        decoded_predictions = list()
        
        for i in range(p.shape[1]):
            seq, path = viterbi_search(p[:, i, :], alphabet, qstring = qstring, qscale = qscale, qbias = qbias, collapse_repeats = collapse_repeats)
            if return_path:
                decoded_predictions.append((seq, path))
            else:
                decoded_predictions.append(seq)

        return decoded_predictions

    def decode_ctc_beamsearch(self, p, beam_size = 5, beam_cut_threshold = 0.1, collapse_repeats = True, *args, **kwargs):

        alphabet = BASES_CRF
        decoded_predictions = list()
        for i in range(p.shape[1]):
            seq, _ = beam_search(p[:, i, :], alphabet, beam_size = beam_size, beam_cut_threshold = beam_cut_threshold, collapse_repeats = collapse_repeats)
            decoded_predictions.append(seq)

        return decoded_predictions
            
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_ctc_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.ctc': loss.item()}

        return loss, losses

    def calculate_ctc_loss(self, y, p):
        """Calculates the ctc loss
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
        """
        
        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        p_len = torch.full((p.shape[1], ), p.shape[0]).to(self.device)
        
        loss = self.criterions["ctc"](p, y, p_len, y_len)
        
        return loss

class BaseModelCRF(BaseModel):
    
    def __init__(self, state_len = 4, alphabet = BASES_CRF, *args, **kwargs):
        """
        Args:
            state_len (int): k-mer length for the states
            alphabet (str): bases available for states, defaults 'NACGT'
        """
        super(BaseModelCRF, self).__init__(*args, **kwargs)

        self.alphabet = alphabet
        self.state_len = alphabet
        self.seqdist = CTC_CRF(state_len = state_len, alphabet = alphabet)
        self.criterions = {'crf': self.seqdist.ctc_loss}

        
    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_crf_greedy(p, *args, **kwargs)
        else:
            return self.decode_crf_beamsearch(p, *args, **kwargs)

    def compute_scores(self, probs, use_fastctc = False):
        """
        Args:
            probs (cuda tensor): [length, batch, channels]
            use_fastctc (bool)
        """
        if use_fastctc:
            scores = probs.cuda().to(torch.float32)
            betas = self.seqdist.backward_scores(scores.to(torch.float32))
            trans, init = self.seqdist.compute_transition_probs(scores, betas)
            trans = trans.to(torch.float32).transpose(0, 1)
            init = init.to(torch.float32).unsqueeze(1)
            return (trans, init)
        else:
            scores = self.seqdist.posteriors(probs.cuda().to(torch.float32)) + 1e-8
            tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
            return tracebacks

    def _decode_crf_greedy_fastctc(self, tracebacks, init, qstring, qscale, qbias, return_path):
        #print("_decode_crf_greedy_fastctc")
        """
        Args:
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            qstring (bool)
            qscale (float)
            qbias (float)
            return_path (bool)
        """

        seq, path = crf_greedy_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = BASES_CRF, 
            qstring = qstring, 
            qscale = qscale, 
            qbias = qbias
        )
        if return_path:
            return seq, path
        else:
            return seq
    
    def decode_crf_greedy(self, probs, use_fastctc = False, qstring = False, qscale = 1.0, qbias = 1.0, return_path = False, *args, **kwargs):
        #print("decode_crf_greedy")
        """Predict the sequences using a greedy approach
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        if use_fastctc:
            tracebacks, init = self.compute_scores(probs, use_fastctc)
            return self._decode_crf_greedy_fastctc(tracebacks, init, qstring, qscale, qbias, return_path)
        
        else:
            return [self.seqdist.path_to_str(y) for y in self.compute_scores(probs, use_fastctc).cpu().numpy()]

    def _decode_crf_beamsearch_fastctc(self, tracebacks, init, beam_size, beam_cut_threshold, return_path):
        #print("_decode_crf_beamsearch_fastctc")
        """
        Args
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            beam_size (int)
            beam_cut_threshold (float)
            return_path (bool)
        """
        seq, path = crf_beam_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = BASES_CRF, 
            beam_size = beam_size,
            beam_cut_threshold = beam_cut_threshold
        )
        if return_path:
            return seq, path
        else:
            return seq

    def decode_crf_beamsearch(self, probs, beam_size = 5, beam_cut_threshold = 0.1, return_path = False, *args, **kwargs):
        """Predict the sequences using a beam search
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        tracebacks, init = self.compute_scores(probs, use_fastctc = True)
        return self._decode_crf_beamsearch_fastctc(tracebacks, init, beam_size, beam_cut_threshold, return_path)

    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_crf_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.crf': loss.item()}

        return loss, losses

    def calculate_crf_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """

        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        loss = self.criterions['crf'](scores = p, 
                                      targets = y, 
                                      target_lengths = y_len, 
                                      loss_clip = 10, 
                                      reduction='mean', 
                                      normalise_scores=True)
        return loss

class BaseModelImpl(BaseModelCTC, BaseModelCRF):

    def __init__(self, decoder_type = 'ctc', *args, **kwargs):
        super(BaseModelImpl, self).__init__(*args, **kwargs)

        valid_decoder_types = ['ctc', 'crf']
        if decoder_type not in valid_decoder_types:
            raise ValueError('Given decoder_type: ' + str(decoder_type) + ' is not valid. Valid options are: ' + str(valid_decoder_types))
        self.decoder_type = decoder_type

    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            and logprobabilities
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """

        if self.decoder_type == 'ctc':
            p = p.exp().detach().cpu().numpy()
            return BaseModelCTC.decode(self, p.astype(np.float32), greedy = greedy, *args, **kwargs)
        if self.decoder_type == 'crf':
            #print("greedy_ctc")
            return BaseModelCRF.decode(self, p, greedy, *args, **kwargs)
        
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        if self.decoder_type == 'ctc':
            return BaseModelCTC.calculate_loss(self, y, p)
        if self.decoder_type == 'crf':
            return BaseModelCRF.calculate_loss(self, y, p)

    def build_decoder(self, encoder_output_size, decoder_type):

        if decoder_type == 'ctc':
            decoder = nn.Sequential(nn.Linear(encoder_output_size, len(BASES)+1), nn.LogSoftmax(-1))
        elif decoder_type == 'crf':
            decoder = BonitoLinearCRFDecoder(
                insize = encoder_output_size, 
                n_base = CRF_N_BASE, 
                state_len = CRF_STATE_LEN, 
                bias=CRF_BIAS, 
                scale= CRF_SCALE, 
                blank_score= CRF_BLANK_SCORE
            )
        else:
            raise ValueError('decoder_type should be "ctc" or "crf", given: ' + str(decoder_type))
        return decoder

class BaseModelS2S(BaseModel):
    
    def __init__(
        self, 
        convolution = None, 
        encoder = None, 
        decoder = None, 
        scheduled_sampling = 0, 
        max_len = None,
        token_sos = S2S_SOS,
        token_eos = S2S_EOS,
        token_pad = S2S_PAD,
        out_classes = S2S_OUTPUT_CLASSES,
        *args, **kwargs):

        """Base model for Seq2Seq

        Args:
            convolution (nn.Module): convolution model that is applied before the encoder
            encoder (nn.Module): rnn for the encoder
            decoder (nn.Module): s2s decoder with embedding, rnn, attention and linear output
            scheduled_sampling (float): probability of using the correct token during training
            token_sos (int): value that indicates start of sentence
            token_eos (int): value that indicates end of sentence
            token_pad (int): value used for padding sentences

        The forward is done using a for loop, it stops if all the samples of the
        batch predict an eos_token. Max length is not defined and it goes at least
        as many times as the amount of timesteps in the output of the encoder.
        """

        super(BaseModelS2S, self).__init__(*args, **kwargs)

        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder

        self.token_sos = token_sos
        self.token_eos = token_eos
        self.token_pad = token_pad
        self.out_classes = out_classes

        class_weights = torch.ones((out_classes,), device = self.device)
        class_weights[self.token_eos] = 3 # add weight to eos token, it is critical that we stop predicting
        self.criterions['ce'] = nn.NLLLoss(weight=class_weights)
        self.scheduled_sampling = scheduled_sampling
        self.max_len = max_len
        self.decoder_type = 'seq2seq'

    def forward(self, x, y = None, forced_teaching = 0, max_len = None):
        """Forward through the network

        Args:
            x (tensor): tensor with raw signal in shape [batch, channels, len]
            y (tensor): tensor with the encoded labels as integers [batch, len]
            forced_teaching (float): float between 0 and 1 that determines
                the probability of using the true label for a step during decoding

            Returns:
                tensor with shape
        """

        if forced_teaching > 0 and y is None:
            raise ValueError('y must be given if forced_teaching > 0')

        if max_len is None:
            max_len = x.shape[-1]
        x = self.convolution(x)
        x = x.permute(2, 0, 1)
        enc_out, hidden = self.encoder(x) 

        if hidden[0] is not None:
        # get the relevant hidden states
            hidden = self._concat_hiddens(hidden)

        if not isinstance(hidden, tuple):
            hidden = (hidden)

        # keep track of which samples have predicted eos
        ended_sequences = torch.ones(enc_out.shape[1], device = self.device)
        encoder_timesteps = enc_out.shape[0]

        ## [len, batch, classes]
        outputs = torch.zeros(max_len, enc_out.shape[1], self.out_classes).to(self.device)
        outputs[1:, :, self.token_pad] = 1 # fill with padding predictions
        outputs[0, :, self.token_sos] = 1 # first token is always SOS

        # first input is the SOS token
        dec_in = torch.full((enc_out.shape[1], ), fill_value = self.token_sos, device = self.device)
        # initial attention is all zeros except first timepoint so that it can look
        # at all the timepoints
        last_attention = torch.zeros((encoder_timesteps, enc_out.shape[1]), device = self.device)
        last_attention[0, :] = 1

        for t in range(1, max_len):

            dec_out, hidden, last_attention = self.decoder(dec_in, hidden, enc_out, last_attention)
            outputs[t, :, :] = dec_out

            teacher_force = random.random() < forced_teaching
            #if teacher forcing, use actual next token as next input
            #if not, use predicted tokens                
            if teacher_force:
                dec_in = y[:, t] # [batch]
            else:   
                dec_in = dec_out.argmax(2).squeeze(0) # [batch]
                
            ended_sequences[torch.where(dec_out.argmax(2).squeeze(0) == self.token_eos)[0]] = 0
            if torch.sum(ended_sequences) == 0:
                break    

        return outputs        

    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        y = batch['y'].to(self.device)
        y = y.to(int)

        # during training shorten max_len based on training to speed up training
        max_len = int(torch.max(torch.sum(y != 0, axis = 1)) + 10)

        p = self.forward(x, y, forced_teaching = self.scheduled_sampling, max_len = max_len) # forward through the network
        
        loss, losses = self.calculate_loss(y, p.permute(1, 2, 0))
        self.optimize(loss)
        
        return losses, p
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1) # add channels dimension
            y = batch['y'].to(self.device)
            y = y.to(int)
            p = self.forward(x, None, forced_teaching = 0.0, max_len = self.max_len) # forward through the network
            
            _, losses = self.calculate_loss(y, p.permute(1, 2, 0))
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1)
            p = self.forward(x, None, forced_teaching = 0.0, max_len = self.max_len)
            
        return p

    def calculate_loss(self, y, p):
        """Calculate CE loss

        Args:
            y (tensor): shape [batch, len]
            p (tensor): shape [batch, channels, len]
        """

        loss = self.criterions['ce'](p, y[:, :p.shape[2]])
        losses = {'loss.global': loss.item(), 'loss.ce': loss.item()}

        return loss, losses
     
    def decode(self, p, decoding_dict = None, greedy = False):
        """Decode the predictions into sequences

        Args:
            p (tensor): tensor of predictions with shape [timesteps, batch, channels]
            greedy (bool): whether to use greedy decoding or beam search

        Returns:
            A `list` with the decoded sequences as strings.
        """

        if decoding_dict is None:
            try:
                decoding_dict = self.dataloader_train.dataset.decoding_dict
            except:
                raise ValueError('Missing decoding_dict')

        if greedy:
            return self.decode_greedy(p, decoding_dict)
        else:
            return self.decode_beamsearch(p, decoding_dict)

    def decode_greedy(self, p, decoding_dict):
        #print("decoder_greedy")
        p = p.detach()
        p = p.permute(1, 0, 2) # [batch, len, channels]
        p = p.argmax(-1) # get most probable 
        p = p.cpu().numpy()
    
        decoded_sequences = list()
        for sample in p:
            seq = ''
            has_stop_codon = False
            for base in sample:
                if base == self.token_sos or base == self.token_pad:
                    continue
                if base == self.token_eos:
                    has_stop_codon = True
                    break
                seq += decoding_dict[base]
            
            if not has_stop_codon:
                if len(seq) > 0:
                    last_base = seq[-1]
                    seq = np.array(list(seq))
                    pos = np.where(seq != last_base)[0]
                    if len(pos) == 0:
                        seq = seq[:6]
                    else:
                        seq = seq[:pos[-1] + 2]
                    seq = "".join(seq.tolist())

            decoded_sequences.append(seq)

        return decoded_sequences

    def decode_beamsearch(self):
        raise NotImplementedError()

    def _concat_hiddens(self, hidden):
        """Concatenates the hidden states output of an RNN

        The output of the RNN in the encoder outputs the last hidden states
        of all the RNN layers. We only want the last (or two last if bidirectional)
        This function extracts the relevant hidden states, and cell states
        for LSTM, to be used as input for the decoder.

        Args:
            hidden (tuple, tensor): with shape [len, batch, hidden]

        Returns:
            Same shaped tensor with last two hidden states concatenated in 
            the hidden dimension
        """

        if isinstance(hidden, tuple):
            return (self._concat_hiddens(hidden[0]), self._concat_hiddens(hidden[1]))

        if self.encoder.bidirectional:
            return torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim = -1).unsqueeze(0)
        else:
            return hidden[-1, :, :].unsqueeze(0)

class BaseNanoporeDataset(Dataset):
    """Base dataset class that contains Nanopore data
    
    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of 
    byte-strings with the bases appended with ''.
    
    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.
    
    Args:
        data (str): dir with the npz files
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        s2s (bool): whether to encode for s2s models
        token_sos (int): value used for encoding start of sequence
        token_eos (int): value used for encoding end of sequence
        token_pad (int): value used for padding all the sequences (s2s and not s2s)
    """

    def __init__(self, data_dir, decoding_dict, encoding_dict, 
                 split = 0.95, shuffle = True, seed = None,
                 s2s = False, token_sos = S2S_SOS, token_eos = S2S_EOS, token_pad = S2S_PAD):
        super(BaseNanoporeDataset, self).__init__()
        
        self.data_dir = data_dir
        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
        self.files_list = self._find_files()
        self.num_samples_per_file = self._get_samples_per_file()
        self.total_num_samples = np.sum(np.array(self.num_samples_per_file))
        self.train_files_idxs = set()
        self.validation_files_idxs = set()
        self.train_idxs = list()
        self.validation_idxs = list()
        self.train_sampler = None
        self.validation_sampler = None
        self._split_train_validation()
        self._get_samplers()
        
        self.loaded_train_data = None
        self.loaded_validation_data = None
        self.current_loaded_train_idx = None
        self.current_loaded_validation_idx = None

        self.s2s = s2s
        self.token_sos = token_sos
        self.token_eos = token_eos
        self.token_pad = token_pad

        self._check()
    
    def __len__(self):
        """Number of samples
        """
        return self.total_num_samples
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.
        
        Returns a dictionary
        """
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            return self.get_data(data_dict = self.loaded_train_data, idx = idx[1])
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            return self.get_data(data_dict = self.loaded_validation_data, idx = idx[1])
        else:
            raise IndexError('Given index not in train or validation files indices: ' + str(idx[0]))
    
    def _check(self):
        """Check for possible problems
        """

        # check that the encoding dict does not conflict with S2S tokens
        if self.s2s:
            s2s_tokens = (self.token_eos, self.token_sos, self.token_pad)
            for v in self.encoding_dict.values():
                assert v not in s2s_tokens

    def _find_files(self):
        """Finds list of files to read
        """
        l = list()
        for f in os.listdir(self.data_dir):
            if f.endswith('.npz'):
                l.append(f)
        l = sorted(l)
        return l
    
    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name
        """
        l = list()
        for f in self.files_list:
            metadata = read_metadata(os.path.join(self.data_dir, f))
            l.append(metadata[0][1][0]) # [array_num, shape, first elem shape]
        return l
    
    def _split_train_validation(self):
        """Splits datafiles and idx for train and validation according to split
        """
        
        # split train and validation data based on files
        num_train_files = int(len(self.files_list) * self.split)
        num_validation_files = len(self.files_list) - num_train_files
        
        files_idxs = list(range(len(self.files_list)))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(files_idxs)
            
        self.train_files_idxs = set(files_idxs[:num_train_files])
        self.validation_files_idxs = set(files_idxs[num_train_files:])
        
        # shuffle indices within each file and make a list of indices (file_idx, sample_idx)
        # as tuples that can be iterated by the sampler
        for idx in self.train_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.train_idxs.append((idx, i))
        
        for idx in self.validation_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.validation_idxs.append((idx, i))
                
        return None
    
    def _get_samplers(self):
        """Add samplers
        """
        self.train_sampler = IdxSampler(self.train_idxs, data_source = self)
        self.validation_sampler = IdxSampler(self.validation_idxs, data_source = self)
        return None
            
    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
        arr = np.load(os.path.join(self.data_dir, self.files_list[idx]))
        x = arr['x']
        y = arr['y']
        return self.process({'x':x, 'y':y})
    
    def get_data(self, data_dict, idx):
        """Slices the data for given indices
        """
        return {'x': data_dict['x'][idx], 'y': data_dict['y'][idx]}
    
    def process(self, data_dict):
        """Processes the data into a ready for training format
        """
        
        y = data_dict['y']
        if y.dtype != 'U1':
            y = y.astype('U1')
        if self.s2s:
            y = self.encode_s2s(y)
        else:
            y = self.encode(y)
        data_dict['y'] = y
        return data_dict
    
    def encode(self, y_arr):
        """Encode the labels
        """
        
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        for k, v in self.encoding_dict.items():
            new_y[y_arr == k] = v
        return new_y

    def encode_s2s(self, y_arr):
    
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        # get the length of each sample to add eos token at the end
        sample_len = np.sum(y_arr != '', axis = 1)
        # array with sos_token to append at the begining
        sos_token = np.full((y_arr.shape[0], 1), self.token_sos, dtype=int)
        # replace strings for integers according to encoding dict
        for k, v in self.encoding_dict.items():
            if v is None:
                continue
            new_y[y_arr == k] = v
        # replace first padding for eos token
        for i, s in enumerate(sample_len):
            new_y[i, s] = self.token_eos
        # add sos token and slice of last padding to keep same shape
        new_y = np.concatenate([sos_token, new_y[:, :-1]], axis = 1)
        return new_y
    
    def encoded_array_to_list_strings(self, y):
        """Convert an encoded array back to a list of strings

        Args:
            y (array): with shape [batch, len]
        """

        y = y.astype(str)
        if self.s2s:
            # replace tokens with nothing
            for k in [self.token_sos, self.token_eos, self.token_pad]:
                y[y == str(k)] = ''
        else:
            y[y == str(self.token_pad)] = ''
        # replace predictions with bases
        for k, v in self.decoding_dict.items():
            y[y == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in y.tolist()]
        return decoded_sequences


class IdxSampler(Sampler):
    """Sampler class to not sample from all the samples
    from a dataset.
    """
    def __init__(self, idxs, *args, **kwargs):
        super(IdxSampler, self).__init__(*args, **kwargs)
        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
        
class BaseFast5Dataset(Dataset):
    """Base dataset class that iterates over fast5 files for basecalling
    """

    def __init__(self, 
        data_dir = None, 
        fast5_list = None, 
        recursive = True, 
        buffer_size = 100,
        window_size = 2000,
        window_overlap = 400,
        trim_signal = True,
        ):
        """
        Args:
            data_dir (str): dir where the fast5 file
            fast5_list (str): file with a list of files to be processed
            recursive (bool): if the data_dir should be searched recursively
            buffer_size (int): number of fast5 files to read 

        data_dir and fast5_list are esclusive
        """
        
        super(BaseFast5Dataset, self).__init__()
    
        self.data_dir = data_dir
        self.recursive = recursive
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.trim_signal = trim_signal

        if fast5_list is None:
            self.data_files = self.find_all_fast5_files()
        else:
            self.data_files = self.read_fast5_list(fast5_list)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        return self.process_reads(self.data_files[idx])

        
    def find_all_fast5_files(self):
        """Find all fast5 files in a dir recursively
        """
        # find all the files that we have to process
        files_list = list()
        for path in Path(self.data_dir).rglob('*.fast5'):
            files_list.append(str(path))
        files_list = self.buffer_list(files_list, self.buffer_size)
        return files_list

    def read_fast5_list(self, fast5_list):
        """Read a text file with the reads to be processed
        """

        if isinstance(fast5_list, list):
            return self.buffer_list(fast5_list, self.buffer_size)

        files_list = list()
        with open(fast5_list, 'r') as f:
            for line in f:
                files_list.append(line.strip('\n'))
        files_list = self.buffer_list(files_list, self.buffer_size)
        return files_list

    def buffer_list(self, files_list, buffer_size):
        buffered_list = list()
        for i in range(0, len(files_list), buffer_size):
            buffered_list.append(files_list[i:i+buffer_size])
        return buffered_list

    def trim(self, signal, window_size=40, threshold_factor=2.4, min_elements=3):
        """

        from: https://github.com/nanoporetech/bonito/blob/master/bonito/fast5.py
        """

        min_trim = 10
        signal = signal[min_trim:]

        med, mad = med_mad(signal[-(window_size*100):])

        threshold = med + mad * threshold_factor
        num_windows = len(signal) // window_size

        seen_peak = False

        for pos in range(num_windows):
            start = pos * window_size
            end = start + window_size
            window = signal[start:end]
            if len(window[window > threshold]) > min_elements or seen_peak:
                seen_peak = True
                if window[-1] > threshold:
                    continue
                return min(end + min_trim, len(signal)), len(signal)

        return min_trim, len(signal)

    def chunk(self, signal, chunksize, overlap):
        """
        Convert a read into overlapping chunks before calling

        The first N datapoints will be cut out so that the window ends perfectly
        with the number of datapoints of the read.
        """
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)

        T = signal.shape[0]
        if chunksize == 0:
            chunks = signal[None, :]
        elif T < chunksize:
            chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
        else:
            stub = (T - overlap) % (chunksize - overlap)
            chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
        
        return chunks.unsqueeze(1)
    
    def normalize(self, read_data):
        return normalize_signal_from_read_data(read_data)

    def process_reads(self, read_list):
        """
        Args:
            read_list (list): list of files to be processed

        Returns:
            two arrays, the first one with the normalzized chunked data,
            the second one with the read ids of each chunk.
        """
        chunks_list = list()
        id_list = list()
        l_list = list()

        for read_file in read_list:
            reads_data = read_fast5(read_file)

            for read_id in reads_data.keys():
                read_data = reads_data[read_id]
                norm_signal = self.normalize(read_data)

                if self.trim_signal:
                    trim, _ = self.trim(norm_signal[:8000])
                    norm_signal = norm_signal[trim:]

                chunks = self.chunk(norm_signal, self.window_size, self.window_overlap)
                num_chunks = chunks.shape[0]
                
                uuid_fields = uuid.UUID(read_id).fields
                id_arr = np.zeros((num_chunks, 6), dtype = np.int)
                for i, uf in enumerate(uuid_fields):
                    id_arr[:, i] = uf
                
                id_list.append(id_arr)
                l_list.append(np.full((num_chunks,), len(norm_signal)))
                chunks_list.append(chunks)
        
        out = {
            'x': torch.vstack(chunks_list).squeeze(1), 
            'id': np.vstack(id_list),
            'len': np.concatenate(l_list)
        }
        return out


class BaseBasecaller():

    def __init__(self, dataset, model, batch_size, output_file, n_cores = 4, chunksize = 2000, overlap = 400, stride = None, beam_size = 1, beam_threshold = 0.1):

        assert isinstance(dataset, BaseFast5Dataset)

        self.dataset = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 2)
        self.model = model
        self.batch_size = batch_size
        self.output_file = output_file
        self.n_cores = n_cores
        self.chunksize = chunksize
        self.overlap = overlap
        if stride is None:
            self.stride = self.model.cnn_stride
        else:
            self.stride = stride
        self.beam_size = beam_size
        self.beam_threshold = beam_threshold

    def stich(self, method, *args, **kwargs):
        """
        Stitch chunks together with a given overlap
        
        Args:
            chunks (tensor): predictions with shape [samples, length, classes]
        """

        if method == 'stride':
            return self.stich_by_stride(*args, **kwargs)
        elif method == 'alignment':
            return self.stich_by_alignment(*args, **kwargs)
        else:
            raise NotImplementedError()

    def basecall(self, verbose = True):
        raise NotImplementedError()
    
    def stitch_by_stride(self, chunks, chunksize, overlap, length, stride, reverse=False):
        """
        Stitch chunks together with a given overlap
        
        This works by calculating what the overlap should be between two outputed
        chunks from the network based on the stride and overlap of the inital chunks.
        The overlap section is divided in half and the outer parts of the overlap
        are discarded and the chunks are concatenated. There is no alignment.
        
        Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
        Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
        Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
        
        Args:
            chunks (tensor): predictions with shape [samples, length, *]
            chunk_size (int): initial size of the chunks
            overlap (int): initial overlap of the chunks
            length (int): original length of the signal
            stride (int): stride of the model
            reverse (bool): if the chunks are in reverse order
            
        Copied from https://github.com/nanoporetech/bonito
        """

        if isinstance(chunks, np.ndarray):
            chunks = torch.from_numpy(chunks)

        if chunks.shape[0] == 1: return chunks.squeeze(0)

        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (length - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

        if reverse:
            chunks = list(chunks)
            return torch.cat([
                chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
            ])
        else:
            return torch.cat([
                chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
            ])

    def stich_by_alignment(self, preds, qscores_list, num_patch_bases = 10):

        consensus = list()
        phredq_consensus = list()
        for i in range(0, len(preds) - 2, 2):
            
            if i == 0:
                ref1 = preds[i]
                ref1_phredq = qscores_list[i]

            ref2 = preds[i+2]
            ref2_phredq = qscores_list[i+2]
            que = preds[i+1]
            que_phredq = qscores_list[i+1]

            alignment = STICH_ALIGN_FUNCTION(que, ref1+ref2, open = STICH_GAP_OPEN_PENALTY, extend = STICH_GAP_EXTEND_PENALTY, matrix = MATRIX)

            decoded_cigar = alignment.cigar.decode.decode()
            long_cigar, _, _ = elongate_cigar(decoded_cigar)
            align_arr = make_align_arr(long_cigar, ref1+ref2, que, phredq = que_phredq, phredq_ref = ref1_phredq+ref2_phredq)


            n_gaps = 0
            st_first_segment = len(ref1) - num_patch_bases
            while True:
                n_gaps_new = np.sum(align_arr[0][:st_first_segment] == '-')
                if n_gaps_new > n_gaps:
                    st_first_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            n_gaps = 0
            nd_first_segment = st_first_segment + num_patch_bases
            while True:
                n_gaps_new = np.sum(align_arr[0][st_first_segment:nd_first_segment] == '-')
                if n_gaps_new > n_gaps:
                    nd_first_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            st_second_segment = nd_first_segment
            nd_second_segment = st_second_segment + num_patch_bases

            n_gaps = 0
            while True:
                n_gaps_new = np.sum(align_arr[0][st_second_segment:nd_second_segment] == '-')
                if n_gaps_new > n_gaps:
                    nd_second_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            segment1_patch = "".join(align_arr[2][st_first_segment:nd_first_segment].tolist()).replace('-', '')
            segment2_patch = "".join(align_arr[2][st_second_segment:nd_second_segment].tolist()).replace('-', '')
            segment1_patch_phredq = "".join(align_arr[3][st_first_segment:nd_first_segment].tolist()).replace(' ', '')
            segment2_patch_phredq = "".join(align_arr[3][st_second_segment:nd_second_segment].tolist()).replace(' ', '')

            new_ref1 = ref1[:-num_patch_bases] + segment1_patch
            new_ref1_phredq = ref1_phredq[:-num_patch_bases] + segment1_patch_phredq
            ref1 = segment2_patch + ref2[num_patch_bases:] 
            ref1_phredq = segment2_patch_phredq + ref2_phredq[num_patch_bases:] 
            assert len(new_ref1) == len(new_ref1_phredq)

            consensus.append(new_ref1)
            
            phredq_consensus.append(new_ref1_phredq)

        return "".join(consensus), "".join(phredq_consensus), '+'  

    def stich_by_alignment2(self, preds, qscores_list, chunk_size, chunk_overlap):
        """Stich basecalled sequences via alignment

        Works the same as stich_by_stride, but instead on overlaping the windows
        based on the stride. We overlap the basecalled sequences based on 
        alignment.

        Args:
            preds(list): list of predicted sequences as strings
            qscores(list): list of phredq scores as chars
            chunk_size (int): length of the raw data window
            chunk_overlap (int): length of the overlap between windows
        """
        pre_st = 0
        consensus = list()
        phredq_consensus = list()
        unmasked_fraction = round(chunk_overlap/chunk_size, 1)
        min_length_align = int(chunk_overlap*unmasked_fraction/20)
        difficult_alignments = 0

        for i in range(len(preds)-1):

            que = preds[i+1]
            ref = preds[i]
            ref_phredq = qscores_list[i]
            que_phredq = qscores_list[i+1]
            right = len(ref) - int(len(ref)*unmasked_fraction)
            left = int(len(ref)*unmasked_fraction)

            cut_ref = ref[-left:]
            cut_que = que[:int(len(que)*unmasked_fraction)]
            cut_ref_phredq = ref_phredq[-left:]
            cut_que_phredq = que_phredq[:int(len(que)*unmasked_fraction)]

            if len(cut_ref) <=  min_length_align or len(cut_que) <= min_length_align:
                pre_st = 0
                consensus.append(ref)
                phredq_consensus.append(ref_phredq)
                difficult_alignments += 1
                continue

            alignment = STICH_ALIGN_FUNCTION(cut_que, cut_ref, open = STICH_GAP_OPEN_PENALTY, extend = STICH_GAP_EXTEND_PENALTY, matrix = MATRIX)

            decoded_cigar = alignment.cigar.decode.decode()
            long_cigar, _, _ = elongate_cigar(decoded_cigar)
            align_arr = make_align_arr(long_cigar, cut_ref, cut_que, cut_que_phredq, cut_ref_phredq)
            matches = np.where(align_arr[1] == '|')[0]
            missmatches = np.where(align_arr[1] == '.')[0]

            st = min(np.concatenate([matches, missmatches]))
            nd = max(np.concatenate([matches, missmatches])) + 1
            mid = int((nd - st) /2)
            
            mini1 = align_arr[0][:st + mid]
            mini2 = align_arr[2][(st + mid):nd]
            mini1 = "".join(mini1.tolist()).replace('-', '')
            mini2 = "".join(mini2.tolist()).replace('-', '')

            mini1_phredq = align_arr[4][:st + mid]
            mini2_phredq = align_arr[3][(st + mid):nd]
            mini1_phredq = "".join(mini1_phredq.tolist()).replace(' ', '')
            mini2_phredq = "".join(mini2_phredq.tolist()).replace(' ', '')

            pre_st = nd - st

            consensus_seq = ref[pre_st:right]+mini1+mini2
            consensus_phredq = ref_phredq[pre_st:right]+mini1_phredq+mini2_phredq
            assert len(consensus_seq) == len(consensus_phredq)
            consensus.append(consensus_seq)
            phredq_consensus.append(consensus_phredq)
            
        if difficult_alignments > 0:
            direction = 'difficult_' + str(difficult_alignments)
        else:
            direction = '+'

        return "".join(consensus), "".join(phredq_consensus), direction 

class BasecallerCTC(BaseBasecaller):
    """A base Basecaller class that is used to basecall complete reads
    """
    
    def __init__(self, *args, **kwargs):
        """
        Args:
            model (nn.Module): a model that has the following methods:
                predict, decode
            chunk_size (int): length of the chunks that a read will be divided into
            overlap (int): amount of overlap between consecutive chunks
            batch_size (int): batch size to forward through the network
        """
        super(BasecallerCTC, self).__init__(*args, **kwargs)


    def decode_process(self, probs_stack, read_len, read_id):
    
        probs_stack = self.stitch_by_stride(
            chunks = probs_stack, 
            chunksize = self.chunksize, 
            overlap = self.overlap, 
            length = read_len, 
            stride = self.stride, 
            reverse = False,
        )
        probs_stack = probs_stack.unsqueeze(1)

        if self.beam_size == 1:
            greedy = True
        else:
            greedy = False

        seq = self.model.decode(
            probs_stack, 
            greedy = greedy, 
            qstring = True, 
            collapse_repeats = True, 
            return_path = True,
            beam_size = self.beam_size,
            beam_cut_threshold = self.beam_threshold,
            read_len = read_len,
            chunksize = self.chunksize, 
            overlap = self.overlap, 
            stride = self.stride
        )

        if isinstance(seq[0], tuple):

            fastq_string = '@'+str(read_id)+'\n'
            fastq_string += seq[0][0][:len(seq[0][1])] + '\n'
            fastq_string += '+\n'
            fastq_string += seq[0][0][len(seq[0][1]):] + '\n'
        
        else:

            fastq_string = '@'+str(read_id)+'\n'
            fastq_string += seq[0] + '\n'
            fastq_string += '+\n'
            fastq_string += '?'*len(seq[0]) + '\n'
    
        return fastq_string

    def basecall(self, verbose = True):

        # iterate over the data
        for batch in tqdm(self.dataset, disable = not verbose):
            
            x = batch['x'].squeeze(0)
            l = x.shape[0]
            ss = torch.arange(0, l, self.batch_size)
            nn = ss + self.batch_size

            p_list = list()
            for s, n in zip(ss, nn):
                
                p = self.model.predict_step({'x':x[s:n, :]})
                p_list.append(p)
                
            p = torch.hstack(p_list)

            ids = batch['id'][0]
            ids_arr = np.zeros((ids.shape[0], ), dtype = 'U36')
            for i in range(ids.shape[0]):
                ids_arr[i] = str(uuid.UUID(fields=ids[i].tolist()))


            for read_id in np.unique(ids_arr):
                w = np.where(ids_arr == read_id)[0]
                read_stacks = p[:, w, :].permute(1, 0, 2)
                read_len = batch['len'][0, w[0]].item()

                fastq_string = self.decode_process(read_stacks, read_len, read_id)
                with open(self.output_file, 'a') as f:
                    f.write(str(fastq_string))
                    f.flush()
            

        return None
    


class BasecallerCRF(BaseBasecaller):

    def __init__(self, *args, **kwargs):
        super(BasecallerCRF, self).__init__(*args, **kwargs)


    def basecall(self, verbose = True, qscale = 1.0, qbias = 1.0):
        
        # iterate over the data

        assert self.dataset.dataset.buffer_size == 1
        
        for batch in tqdm(self.dataset, disable = not verbose):

            ids = batch['id'].squeeze(0)
            ids_arr = np.zeros((ids.shape[0], ), dtype = 'U36')
            for i in range(ids.shape[0]):
                ids_arr[i] = str(uuid.UUID(fields=ids[i].tolist()))

            assert len(np.unique(ids_arr)) == 1
            read_id = np.unique(ids_arr)[0]
            
            x = batch['x'].squeeze(0)
            l = x.shape[0]
            ss = torch.arange(0, l, self.batch_size)
            nn = ss + self.batch_size

            transition_scores = list()
            for s, n in zip(ss, nn):
                #print("@"+str(x[s:n, :].shape))
                p = self.model.predict_step({'x':x[s:n, :]})
                scores = self.model.compute_scores(p, use_fastctc=True)
                transition_scores.append(scores[0].cpu())
            init = scores[1][0, 0].cpu()

            stacked_transitions = self.stitch_by_stride(
                chunks = np.vstack(transition_scores), 
                chunksize = self.chunksize, 
                overlap = self.overlap, 
                length = batch['len'].squeeze(0)[0].item(), 
                stride = self.stride
            )

            #print(self.beam_size)
            if self.beam_size == 1:
                #print("greedy decoder")
                seq, path = self.model._decode_crf_greedy_fastctc(
                    tracebacks = stacked_transitions.numpy(), 
                    init = init.numpy(), 
                    qstring = True, 
                    qscale = qscale, 
                    qbias = qbias,
                    return_path = True,
                    
                )
                
                fastq_string = '@'+str(read_id)+'\n'
                fastq_string += seq[:len(path)] + '\n'
                fastq_string += '+\n'
                fastq_string += seq[len(path):] + '\n'
                
            else:
                seq = self.model._decode_crf_beamsearch_fastctc(
                    tracebacks = stacked_transitions.numpy(), 
                    init = init.numpy(), 
                    beam_size = self.beam_size, 
                    beam_cut_threshold = self.beam_threshold, 
                    return_path = False
                )
                
                fastq_string = '@'+str(read_id)+'\n'
                fastq_string += seq + '\n'
                fastq_string += '+\n'
                fastq_string += '?'*len(seq) + '\n'
            
            with open(self.output_file, 'a') as f:
                f.write(str(fastq_string))
                f.flush()

            
class BasecallerSeq2Seq(BaseBasecaller):

    def __init__(self, *args, **kwargs):
        super(BasecallerSeq2Seq, self).__init__(*args, **kwargs)

        #if self.beam_size > 1:
        #    raise NotImplementedError('No beam search yet, only greedy')

    def basecall(self, verbose = True, qscale = 1.0, qbias = 1.0):

        for batch in tqdm(self.dataset, disable = not verbose):
            try:
                with time_limit(seconds = 30):
                    ids = batch['id'].squeeze(0)
                    ids_arr = np.zeros((ids.shape[0], ), dtype = 'U36')
                    for i in range(ids.shape[0]):
                        ids_arr[i] = str(uuid.UUID(fields=ids[i].tolist()))

                    assert len(np.unique(ids_arr)) == 1
                    read_id = str(np.unique(ids_arr)[0])
                    
                    x = batch['x'].squeeze(0)
                    l = x.shape[0]
                    ss = torch.arange(0, l, self.batch_size)
                    nn = ss + self.batch_size

                    predictions = list()
                    for s, n in zip(ss, nn):
                        predictions.append(self.model.predict_step({'x':x[s:n, :]}))
                    all_predictions = torch.cat(predictions, dim = 1).cpu()
                    # calculate qscores as integers
                    qscores = (torch.round(-10*(torch.log10(1 - all_predictions.exp().max(-1).values)) * qscale + qbias) + 33).permute(1, 0).numpy().astype(int)

                    preds = self.model.decode(all_predictions, decoding_dict = RECURRENT_DECODING_DICT, greedy = True)

                    # translate qscores to ascii characters
                    qscores_list = list()
                    for p, qscore in zip(preds, qscores):
                        qscore_str = ""
                        for q in qscore:
                            # padding qscores are negative
                            if q < 0:
                                continue
                            qscore_str += chr(q).encode('ascii').decode("ascii") 
                        # cut qscores for predictions after stop token
                        qscores_list.append(qscore_str[:len(p)])

                    if self.overlap > 0:
                        pred_seq, pred_phredq, direction = self.stich_by_alignment(preds, qscores_list)
                    else:
                        pred_seq = "".join(preds)
                        pred_phredq = "".join(qscores_list)
                        direction = '+'
            except TimeoutException:
                pred_seq = ''
                pred_phredq = ''
                direction = 'timeout'
                
            assert len(pred_seq) == len(pred_phredq)
            fastq_string = '@'+str(read_id)+'\n'
            fastq_string += pred_seq + '\n'
            fastq_string += direction + "\n"
            fastq_string += pred_phredq + '\n'

            with open(self.output_file, 'a') as f:
                f.write(str(fastq_string))
                f.flush()
                




class BasecallerImpl(BasecallerCTC, BasecallerCRF, BasecallerSeq2Seq):

    def __init__(self, *args, **kwargs):
        super(BasecallerImpl, self).__init__(*args, **kwargs)

    def basecall(self, verbose, *args, **kwargs):

        if self.model.decoder_type == 'ctc':
            return BasecallerCTC.basecall(self, verbose = verbose, *args, **kwargs)

        if self.model.decoder_type == 'crf':
            return BasecallerCRF.basecall(self, verbose = verbose, *args, **kwargs)

        if self.model.decoder_type == 'seq2seq':
            return BasecallerSeq2Seq.basecall(self, verbose = verbose, *args, **kwargs)