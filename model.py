import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from embedding import PositionalEmbedding

class TransformerModel(nn.Module):
	def __init__(self, n_src_vocab, n_tgt_vocab, d_model, nhead, nhid, nlayers, dropout=0.1):
		super(TransformerModel, self).__init__()
		try:
			from torch.nn import TransformerEncoder, TransformerEncoderLayer, \
				TransformerDecoder, TransformerDecoderLayer
		except:
			raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
		self.model_type = 'transformer'
		self.d_model = d_model

		self.src_embedding = PositionalEmbedding(n_src_vocab, d_model)
		self.tgt_embedding = PositionalEmbedding(n_tgt_vocab, d_model)

		encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		decoder_layers = TransformerDecoderLayer(d_model, nhead, nhid, dropout)
		self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

		# 生成概率输出
		self.generator = nn.Linear(d_model, n_tgt_vocab)

		self._reset_parameters()

	def _reset_parameters(self):
		r"""Initiate parameters in the transformer model."""
		for p in self.parameters():
			if p.dim() > 1:
				xavier_uniform_(p)

	def pad_masking(self, x):
		pad_mask = x == 0  # (batch_size, seq_len)
		return pad_mask  # (batch_size, seq_len)


	# 简单的上三角矩阵
	def subsequent_masking(self, sz):
		# x: (batch_size, seq_len - 1)
		subsequent_mask=torch.triu(torch.ones(sz, sz), diagonal=1) == 1
        return subsequent_mask

	def encode(self, sources):
		# embedding
		src = self.src_embedding(sources)  # (N, S, E)
		src = src.transpose(0, 1)  # (S, N, E)
		# get mask
		batch_size, sources_len = sources.size()
		src_key_padding_mask = self.pad_masking(source)  # (N, S)
		memory_key_padding_mask = src_key_padding_mask  # (N, S)
		memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
		return memory, memory_key_padding_mask

	def decode(self, targets, memory, memory_key_padding_mask):
		# (N, T)
		batch_size, targets_len = targets.size()
		# embedding
		tgt = self.tgt_embedding(targets)
		tgt = tgt.transpose(0, 1)  # (T, N, E)
		# get mask
		tgt_key_padding_mask = self.pad_masking(targets)  # (N, T)
		tgt_mask = self.subsequent_masking(targets_len)
		dec_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask,
											  tgt_key_padding_mask=tgt_key_padding_mask,
											  memory_key_padding_mask=memory_key_padding_mask)  # (T, N, E)
		return dec_output.transpose(0, 1)

	def forward(self, sources, targets):
		r"""
		:param sources: (N, S)
		:param targets: (N, T)
		:return:	(N, T, FS)

		FS: targets词汇表的大小
		T: targets的长度
		S: src的长度
		"""
		memory, memory_key_padding_mask = self.encode(sources)
		dec_output = self.decode(targets, memory, memory_key_padding_mask)
		output = self.generator(dec_output)

		return memory, dec_output, output

class Model_with_Proof(nn.Module):
	def __init__(self,
					n_src_vocab,
					n_tgt_vocab,
					d_model,
					nhead,
					nlayers,
					nhid,
					dropout,
					d_block,
					P_node_hid,
					P_edge_hid,
					loss_weight):

		super(ModelwithProof, self).__init__()

		self.d_model=d_model

		self.transformer=TransformerModel(n_src_vocab=n_src_vocab,
										n_tgt_vocab=n_tgt_vocab,
										d_model=d_model,
										nhead=nhead,
										nlayers=nlayers,
										nhid=nhid,
										dropout=dropout)

		self.blockleft=nn.Linear(d_model, d_block, bias=True)
		self.blockright=nn.Linear(d_model, d_block, bias=True)
		self.P_node=nn.Sequential(nn.Linear(d_model*2, P_node_hid, bias=True),
									nn.ReLU(),
									nn.Linear(P_node_hid, 3, bias=True),
									nn.Softmax(dim=-1))
		self.P_edge=nn.Sequential(nn.Linear(d_model*6, P_edge_hid, bias=True),
									nn.ReLU(),
									nn.Linear(P_edge_hid, 2, bias=True),
									nn.Softmax(dim=-1))

		self.logits_loss=nn.CrossEntropyLoss(reduction=mean, ignore_index=0)
		self.block_loss=nn.CrossEntropyLoss(reduction=mean, ignore_index=-1)
		self.node_loss=nn.CrossEntropyLoss(reduction=mean, ignore_index=-1)
		self.edge_loss=nn.CrossEntropyLoss(reduction=mean, ignore_index=-1)
		self.loss_weight=loss_weight

	def forward(self,
				source,
				source_len,
				right_pos_truth=None,
				target=None,
				target_len=None,
				target_offset=None,
				node_label=None,
				edge_index=None,
				edge_label=None):

		batch_size=source.size()[0]
		device=encode_output.device()

		encode_output, decode_output, output_logits=self.transformer(source, target[:-1])
		loss_logits=self.logits_loss(output_logits, target[1:])

		source_pad_mask=self.transformer.pad_masking(source)

		left_encode=self.blockleft(encode_output[1:])
		right_encode=self.blockright(encode_output[1:])
		block_match=torch.matmul(BL, BR.transpose(1, 2))

		loss_block=self.block_loss(block_match, right_pos_truth)
		block_embedding=torch.zeros((batch_size, source_len.max(), self.d_model)).to(device)

		for i in range(batch_size):
			for j in range(source_len[i]):
				block_embedding[i][j]=encode_output[i][1+j:1+right_pos_truth[i][j]].mean(dim=-1)

		trace_embedding=torch.zeros((batch_size, target_len.max(), self.d_model)).to(device)

		for i in range(batch_size):
			for j in range(target_len[i]):
				block_embedding[i][j]=decode_output[i][target_offset[i][j]:target_offset[i][-1]].mean(dim=-1)

		node_embedding=torch.cat((torch.repeat_interleave(block_embedding, target_len.max(), dim=1),
								torch.repeat_interleave(trace_embedding, source_len.max(), dim=1)),
								dim=-1)

		node=self.P_node(node_embedding)
		loss_node=self.node_loss(node, node_label)

		n_node=source_len.max()*target_len.max()

		edge_embedding=torch.zeros((batch, n_node*6, self.d_model*6)).to(device)

		for i in range(batch_size):
			for j in range(n_node):
				head=node_embedding[i][j].repeat(6, 1)
				tail=node_embedding[i].index_select(dim=0, edge_index[i][j])
				edge_embedding[i][j*6:(j+1)*6]=torch.cat((head, tail, head-tail), dim=-1)

		edge=self.P_edge(edge_embedding)
		loss_edge=self.edge_loss(edge, edge_label)

		return output_logits, torch.dot(torch.cat((loss_logits, loss_block, loss_node, loss_edge)), loss_weight)
