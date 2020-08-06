import copy 
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import os 
import math

class PositionalEncoding(nn.Module):
	"""docstring for PositionalEncoding"""
	def __init__(self, d_model = 512, max_positions = 50):
		super(PositionalEncoding, self).__init__()
		self.embedd = torch.zeros(max_positions, d_model, requires_grad = False)

		for pos in range(max_positions):
			for i in range(int(d_model / 2)):
				self.embedd[pos][2*i] = math.sin(pos/(10000**((2*i)/d_model)))
				self.embedd[pos][2*i + 1] = math.cos(pos/(10000**((2*i)/d_model)))
	
	def get(self, max_postions, frm = 0):
		return self.embedd[frm:max_postions, :].cuda()

class TransformerEncoderLayer(nn.Module):
	def __init__(self, d_model = 512, nhead = 8, dim_feedforward=2048, dropout=0.1, activation = "relu"):
		super(TransformerEncoderLayer, self).__init__()
		# there are total 8 attentions, so each attention head dim will be d_model // nhead,
		# MultiheadAttention will take care of that
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

		## Feed Forward layer is nothing but 2 linear layers that take the dims from 512->2048->512
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)

		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerEncoderLayer, self).__setstate__(state)

	def forward(self, input):
		x = self.norm1(input) # x -> 128x40x512
		
		x = torch.transpose(x, 0, 1)
		x = self.self_attn(x, x, x)[0]
		x = torch.transpose(x, 0, 1)

		input = input + self.dropout1(x) # x -> [128x40x64;128x40x64;... ;128x40x64] -> 128x40x512
		x = self.norm2(input)  # 128x40x512 
		x = self.linear2(self.dropout(self.activation(self.linear1(x)))) # feedforward layer # 128x40x2048 --> 128x40x512
		input = input + self.dropout2(x) # 128x40x512
		return input #128x40x512

class TransformerEncoder(nn.Module):
	"""docstring for TransformerEncoder"""
	def __init__(self, encoder_layer, num_layers, norm=None):
		super(TransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, scr: Tensor):
		output = scr
		for mod in self.layers:
			output = mod(output)

			if self.norm is not None:
				output = self.norm(output)

		return output #128x40x512

class TransformerDecoderLayer(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, dropout = 0.3, activation = "relu"):
		super(TransformerDecoderLayer, self).__init__()
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.nhead = nhead
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

		self.masked_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerDecoderLayer, self).__setstate__(state)

	def forward(self, input, output, mask):
		# input to the decoder layer
		# output : The ouput of encoder from last layer
		# mask: The mask for the target
		# memory_mask: memory_mask(optional)
	
		x = self.norm1(input) # 128x50x512
		# multihead output is attn, attn_wts
		# print(mask.shape)
		# print(x.shape)

		# Two types of mask key_padding_mask and attn mask, key_padding because encoded keys can be of variable length
		# takes input in shape (Lengthxbatch_sizexdims) and output is also (lengthxbatch_sizexdims)
		# mask = mask.unsqueeze(0).repeat(self.nhead*x.shape[0], 1, 1)
		x = torch.transpose(x, 0, 1)
		# print(x.shape)
		x = self.masked_attn(x, x, x, attn_mask=mask)[0] # 128x50x512
		x = torch.transpose(x, 0, 1)
		
		input = input + self.dropout1(x) # 128x50x512
		x = self.norm2(input) # 128x50x512
		
		x = torch.transpose(x, 0, 1)
		output = torch.transpose(output, 0, 1)
		x = self.attn(x, output, output)[0] # softmax((128x50x512)x(128x40x512).T / sqrt(512))x(128x40x512)
		output = torch.transpose(output, 0, 1)
		x = torch.transpose(x, 0, 1)

		input = input + self.dropout2(x) #128X50X512
		x = self.norm3(input) #128X50X512
		x = self.linear2(self.dropout(self.activation(self.linear1(x)))) #128X50X512
		input = input + self.dropout3(x) #128X50X512
		return input #128X50X512
		

class TransformerDecoder(nn.Module):
	def __init__(self, decoder_layer, num_layers, norm = None):
		super(TransformerDecoder, self).__init__()
		self.layers = _get_clones(decoder_layer, num_layers) 
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, tgt: Tensor, memory:Tensor, tgt_mask: Tensor):
		output = tgt
		for mod in self.layers:
			output = mod(output, memory, tgt_mask)

			if self.norm is not None:
				output = self.norm(output)

		return output #128X50X512



class Transformer(nn.Module):
	def __init__(self, vocab_size: int, max_len: int, encoded_feats_dim: int = 2048, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, \
	dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"):
		super(Transformer, self).__init__()
		
		self.max_len = max_len #max_len of a sentence
		self.sos_id = 1
		self.eos_id = 0     
		self.transform_2D = nn.Linear(encoded_feats_dim, d_model)
		self.transform_3D = nn.Linear(encoded_feats_dim, d_model)
		self.transform = nn.Linear(2*d_model, d_model)

		self.pos_embedd = PositionalEncoding(d_model)
		encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
		self.Encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
		self.norm_encoder = nn.LayerNorm(d_model)
		
		decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
		self.Decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

		self._reset_parameters()
		self.embedding = nn.Embedding(vocab_size, d_model)
		self.Linear_layer = nn.Linear(d_model, vocab_size)


	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask.cuda()


	def _reset_parameters(self):
		r"""Initiate parameters in the transformer model."""

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, feats_2D: Tensor, feats_3D: Tensor, tgt: Tensor = None, mode = 'train', opt = {}):
		seq_logprobs = []
		seq_preds = []
		until_now = []
		sample_max = opt.get('sample_max', 1)
		beam_size = opt.get('beam_size', 1)
		temperature = opt.get('temperature', 1.0)

		if mode == 'train':
			
			feats_2D = self.transform_2D(feats_2D) #128x40x2048 -> 128x40x512
			feats_3D = self.transform_3D(feats_3D) #128x40x2048 -> 128x40x512
			encoded_feats = torch.cat([feats_2D, feats_3D], dim = 2) #128x40x1024
			encoded_feats = self.transform(encoded_feats) #128x40x512 
			
			batch_size, max_len_encoder, _ = encoded_feats.shape

			pos_encoder = self.pos_embedd.get(max_len_encoder, 0) # 40x512
			pos_encoder = pos_encoder.unsqueeze(0).repeat(batch_size, 1, 1) # 128x40x512
			encoded_feats = encoded_feats + pos_encoder # 128x40x512
			encoded_feats = self.norm_encoder(self.Encoder(encoded_feats)) # 128x40x512

			tgt = self.embedding(tgt) # 128x50x512
			_, max_len_decoder, _ = tgt.shape # tgt -> 128x50x512
			tgt = tgt[:, :max_len_decoder-1, :]
			pos_decoder = self.pos_embedd.get(max_len_decoder - 1, 0) # 50x512
			pos_decoder = pos_decoder.unsqueeze(0).repeat(batch_size, 1, 1) # 128x50x512
			tgt = tgt + pos_decoder #128x50x512
			tgt_mask = self._generate_square_subsequent_mask(max_len_decoder - 1)
			logprobs = F.log_softmax(self.Linear_layer(self.Decoder(tgt, encoded_feats, tgt_mask)), dim = 2) #128X50X512 -> #128X50X11K -> #128X50X11K
			seq_logprobs = logprobs

		elif mode == 'inference':
			
			feats_2D = self.transform_2D(feats_2D) #128x40x2048 -> 128x40x512
			feats_3D = self.transform_3D(feats_3D) #128x40x2048 -> 128x40x512
			encoded_feats = torch.cat([feats_2D, feats_3D], dim = 2) #128x40x1024
			encoded_feats = self.transform(encoded_feats) #128x40x512 

			batch_size, max_len_encoder, _ = encoded_feats.shape
			
			pos_encoder = self.pos_embedd.get(max_len_encoder, 0) # 40x512
			pos_encoder = pos_encoder.unsqueeze(0).repeat(batch_size, 1, 1) # 128x40x512
			encoded_feats = encoded_feats + pos_encoder # 128x40x512
			encoded_feats = self.norm_encoder(self.Encoder(encoded_feats)) # 128x40x512

			# if beam_size > 1:
				# return self.sample_beam(encoder_outputs, decoder_hidden, opt)

			for t in range(self.max_len - 1):
				if t == 0:  # input <bos>
					it = torch.LongTensor([self.sos_id] * batch_size).cuda()
				elif sample_max:
					sampleLogprobs, it = torch.max(logprobs, 1)
					seq_logprobs.append(sampleLogprobs.view(-1, 1))
					it = it.view(-1).long()

				else:
					# sample according to distribuition
					if temperature == 1.0:
						prob_prev = torch.exp(logprobs)
					else:
						# scale logprobs by temperature
						prob_prev = torch.exp(torch.div(logprobs, temperature))
					it = torch.multinomial(prob_prev, 1).cuda()
					sampleLogprobs = logprobs.gather(1, it)
					seq_logprobs.append(sampleLogprobs.view(-1, 1))
					it = it.view(-1).long()

				seq_preds.append(it.view(-1, 1))

				xt = self.embedding(it) #128x512
				xt = xt.unsqueeze(1) #128x1x512
				until_now.append(xt)
				pos_decoder = self.pos_embedd.get(t+1, 0) # curr_lenx512
				pos_decoder = pos_decoder.unsqueeze(0).repeat(batch_size, 1, 1) # 128xcurr_lenx512
				tgt = torch.cat(until_now, 1) #128xcurr_lenx512
				tgt =  tgt + pos_decoder # 128xcurr_lenx512
				logprobs = F.log_softmax(self.Linear_layer(self.Decoder(tgt, encoded_feats, None)), dim = 2)[:, -1, :] #128x11k
				
			seq_logprobs = torch.cat(seq_logprobs, 1)
			seq_preds = torch.cat(seq_preds[1:], 1)

		return seq_logprobs, seq_preds

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
	if(activation == "relu"):
		return F.relu

	raise RuntimeError("activation shud be relu, not {}".format(activation))
