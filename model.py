import math
import torch
import torch.nn as nn
import nn.functional as F
from nn.init import xavier_uniform_
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
		batch_size, sources_len = sources.shape
		src_key_padding_mask = self.pad_masking(source)  # (N, S)
		memory_key_padding_mask = src_key_padding_mask  # (N, S)
		memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
		return memory, memory_key_padding_mask

	def decode(self, targets, memory, memory_key_padding_mask):
		# (N, T)
		batch_size, targets_len = targets.shape
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
				right_pos_truth,
				target,
				state_len,
				target_offset,
				node_label,
				edge_index,
				edge_label):

		batch_size=source.shape[0]
		device=source.device()

		encode_output, decode_output, output_logits=self.transformer(source, target[:, :-1])
		loss_logits=self.logits_loss(output_logits, target[:, 1:])

		left_encode=self.blockleft(encode_output[:, 1:])
		right_encode=self.blockright(encode_output[:, 1:])
		block_match=torch.matmul(left_encode, right_encode.transpose(1, 2))

		loss_block=self.block_loss(block_match, right_pos_truth)
		block_embedding=torch.zeros((batch_size, source_len.max(), self.d_model)).to(device)

		for i in range(batch_size):
			for j in range(source_len[i]):
				block_embedding[i][j]=encode_output[i][1+j:1+right_pos_truth[i][j]].mean(dim=-1)

		trace_embedding=torch.zeros((batch_size, state_len.max(), self.d_model)).to(device)

		for i in range(batch_size):
			for j in range(state_len[i]):
				block_embedding[i][j]=decode_output[i][target_offset[i][j]:target_offset[i][-1]].mean(dim=-1)

		node_embedding=torch.cat((torch.repeat_interleave(block_embedding, state_len.max(), dim=1),
								torch.repeat_interleave(trace_embedding, source_len.max(), dim=1)),
								dim=-1)

		node=self.P_node(node_embedding)
		loss_node=self.node_loss(node, node_label)

		n_node=source_len.max()*state_len.max()

		edge_embedding=torch.zeros((batch, n_node*6, self.d_model*6)).to(device)

		for i in range(batch_size):
			for j in range(n_node):
				head=node_embedding[i][j].repeat(6, 1)
				tail=node_embedding[i].index_select(dim=0, edge_index[i][j])
				edge_embedding[i][j*6:(j+1)*6]=torch.cat((head, tail, head-tail), dim=-1)

		edge=self.P_edge(edge_embedding)
		loss_edge=self.edge_loss(edge, edge_label)

		return output_logits, torch.dot(torch.cat((loss_logits, loss_block, loss_node, loss_edge)), loss_weight)

class Evaluator:
	def __init__(self,
					model,
					n_src_vocab,
					n_tgt_vocab,
					d_model，
					n_beam,
					max_seq_len,
					src_pad_idx,
					tgt_pad_idx,
					tgt_sos_idx,
					tgt_semicolon_idx,
					tgt_eos_idx,
					len_penalty):

		super(Evaluator, self).__init__()

		self.model=model
		self.model.eval()
		self.n_src_vocab=n_src_vocab
		self.n_tgt_vocab=n_tgt_vocab
		self.d_model=d_model
		self.n_beam=n_beam,
		self.max_seq_len=max_seq_len
		self.src_pad_idx=src_pad_idx
		self.tgt_pad_idx=tgt_pad_idx
		self.tgt_sos_idx=tgt_sos_idx
		self.tgt_semicolon_idx=tgt_semicolon_idx
		self.tgt_eos_idx=tgt_eos_idx
		self.len_penalty=len_penalty

	def beam_search(self, source):
		batch_size=source.shape[0]
		device=source.device()
		source_beam=source_beam.repeat_interleave(self.n_beam, dim=0)

		with torch.no_grad():
			scores=torch.zeros(batch_size*self.n_beam).to(device)
			done=torch.full(batch_size*self.n_beam, False, dtype=torch.bool, device=device)

			gen_seq=torch.full((batch_size*n_beam, 1), self.tgt_sos_idx, dtype=torch.int32, device=device)

			for i in range(self.max_seq_len):
				if done.sum().item()==self.batch_size*self.n_beam:
					break

				_, _, output_logits=self.model.transformer(source_beam, gen_seq)
				best_k2_probs, best_k2_idx=output_logits[:, -1, :].topk(self.n_beam)
				scores=F.log_softmax(best_k2_probs, dim=-1).masked_fill(done[:, None], 0)+scores[:, None]

				neg_inf_mask = (torch.BoolTensor([False]+[True]*(self.n_beam-1)).to(device)).repeat(batch_size*self.n_beam, 1).logical_and(done[:, None])
				scores=scores+torch.zeros(neg_inf_mask.shape, device=neg_inf_mask.device).masked_fill(neg_inf_mask, -1e6)

				scores, best_k_idx_in_k2=torch.topk(scores.view(batch_size, -1), self.n_beam, dim=-1, largest=True, sorted=True)

				scores=scores.view(-1)
				best_k_idx=best_k2_idx.view(batch_size, -1)[best_k_idx_in_k2].view(-1) #token
				best_r_idx=best_k_idx_in_k2 // self.n_beam #beam resort
				done=done[best_r_idx]

				best_k_idx=best_k_idx.masked_fill(done, self.tgt_pad_idx)
				done=done.logical_or(best_k_idx==self.tgt_eos_idx)

				gen_seq=torch.cat((gen_seq[best_r_idx, :], best_k_idx.unsquenze(dim=1)), dim=-1)

			target=gen_seq[torch.arange(0, batch_size*self.n_beam, self.n_beam).to(device), :]

			state_len=[]
			for x in target:
				target_offset.append([0])
				x_list=""
				for y in x:
					if y==self.tgt_eos_idx:
						break
					if y==self.tgt_semicolon_idx:
						x_list+=';'
					x_list+='#'

				for y in x.split(';'):
					target_offset[-1].append(target_offset[-1][-1]+len(y)+1)
				target_offset[-1][-1]-=1
				state_len.append(len(target_offset[-1])-1)

			max_target_len=max(target)
			max_state_len=max(state_len)+1
			for i in range(batch_size):
				target_offset[i]+=[0]*(max_state_len-len(target_offset[i]))
				
			return target, torch.IntTensor(state_len).to(device), torch.IntTensor(target_offset).to(device)

	def gen_edge_index(self, source_len, state_len, right_pos):
		batch_size=source_len.shape[0]
		n_node=source_len.max()*state_len.max()
		edge_index_list=[]

		for batch_index in range(batch_size):
			edge_index_list.append([[0]*6 for _ in range(n_node)])
			for suffix in range(state_len[batch_index]):
				for left in range(source_len[batch_index]):
					index=suffix*source_len[batch_index]+left
					cnt=0
					for i in range(2):
						if suffix+i<state_len[batch_index]:
							edge_index_list[-1][index][cnt]=(suffix+i)*state_len[batch_index]+left
							cnt+=1
							if left+1<source_len[batch_index]:
								edge_index_list[-1][index][cnt]=(suffix+i)*state_len[batch_index]+left+1
								cnt+=1
								if right_pos[index][left+1]+1<source_len[batch_index]:
									edge_index_list[-1][index][cnt]=(suffix+i)*state_len[batch_index]+right_pos[batch_index][left+1]+1
									cnt+=1
		return torch.IntTensor(edge_index_list)

	def run(self, source, source_len):

		target, state_len, target_offset=self.beam_search(source)
		batch_size=source.shape[0]
		device=source.device()

		with torch.no_grad():
			encode_output, decode_output, output_logits=self.model.transformer(source, target[:, :-1])

			left_encode=self.model.blockleft(encode_output[:, 1:])
			right_encode=self.model.blockright(encode_output[:, 1:])
			block_match=torch.matmul(left_encode, right_encode.transpose(1, 2))

			right_pos=block_match.argmax(dim=-1)

			n_node=source_len.max()*state_len.max()

			edge_index=self.gen_edge_index(source_len, state_len, right_pos)

			block_embedding=torch.zeros((batch_size, source_len.max(), self.d_model)).to(device)

			for i in range(batch_size):
				for j in range(source_len[i]):
					block_embedding[i][j]=encode_output[i][1+j:1+right_pos[i][j]].mean(dim=-1)

			trace_embedding=torch.zeros((batch_size, state_len.max(), self.d_model)).to(device)

			for i in range(batch_size):
				for j in range(state_len[i]):
					block_embedding[i][j]=decode_output[i][target_offset[i][j]:target_offset[i][-1]].mean(dim=-1)

			node_embedding=torch.cat((torch.repeat_interleave(block_embedding, state_len.max(), dim=1),
									torch.repeat_interleave(trace_embedding, source_len.max(), dim=1)),
									dim=-1)

			node=self.model.P_node(node_embedding).argmax(dim=-1)
			node_mask=node==2

			edge_embedding=torch.zeros((batch, n_node*6, self.d_model*6)).to(device)

			for i in range(batch_size):
				for j in range(n_node):
					head=node_embedding[i][j].repeat(6, 1)
					tail=node_embedding[i].index_select(dim=0, edge_index[i][j])
					edge_embedding[i][j*6:(j+1)*6]=torch.cat((head, tail, head-tail), dim=-1)

			edge, edge_choice=torch.topk(self.model.P_edge(edge_embedding), 2, dim=-1, largest=True, sorted=True)
			edge_choice=edge_index[edge_idx]

		max_source_len=source_len.max()

		proof=[]
		for i in range(batch_size):
			sub_proof=[]
			for j in range(n_node):
				if node_mask[i][j]!=2:
					for k in range(2):
						if node_mask[i][edge_choice[i][j][k]]!=2:
							tmp=edge_choice[i][j][k]
							x=[tmp/max_source_len, [tmp%max_source_len, right_pos[tmp%max_source_len]], node_mask[i][tmp]]
							x[0], x[1][0], x[1][1], x[2]=x[0].item(), x[1][0].item(), x[1][1].item(), x[2].item()
							sub_proof.append(x)
			proof.append(sub_proof)

		return target.cpu().detach().numpy().tolist(), proof