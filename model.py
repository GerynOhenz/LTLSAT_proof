import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class PositionalEmbedding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim=None, dropout_prob=0., padding_idx=0, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        if dim is None:
            dim = embedding_dim

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embbedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embbedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x

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
	def subsequent_masking(self, x):
		# x: (batch_size, seq_len - 1)
		sz=x.shape[-1]
		subsequent_mask=torch.triu(torch.ones((sz, sz), device=x.device), diagonal=1) == 1
		return subsequent_mask

	def encode(self, sources):
		# embedding
		src = self.src_embedding(sources)  # (N, S, E)
		src = src.transpose(0, 1)  # (S, N, E)
		# get mask
		batch_size, sources_len = sources.shape
		src_key_padding_mask = self.pad_masking(sources)  # (N, S)
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
		tgt_mask = self.subsequent_masking(targets)
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

		return memory.transpose(0, 1), dec_output, output

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
					d_block_hid,
					P_node_hid,
					P_edge_hid,
					loss_weight):

		super(Model_with_Proof, self).__init__()

		self.d_model=d_model
		self.n_tgt_vocab=n_tgt_vocab

		self.transformer=TransformerModel(n_src_vocab=n_src_vocab,
										n_tgt_vocab=n_tgt_vocab,
										d_model=d_model,
										nhead=nhead,
										nlayers=nlayers,
										nhid=nhid,
										dropout=dropout)

		self.blockleft=nn.Sequential(nn.Linear(d_model, d_block_hid, bias=True),
										nn.Tanh(),
										nn.Linear(d_block_hid, d_block, bias=True),
										nn.Tanh())
		self.blockright=nn.Sequential(nn.Linear(d_model, d_block_hid, bias=True),
										nn.Tanh(),
										nn.Linear(d_block_hid, d_block, bias=True),
										nn.Tanh())
		self.P_node=nn.Sequential(nn.Linear(d_model*2, P_node_hid, bias=True),
									nn.Tanh(),
									nn.Linear(P_node_hid, 3, bias=True))
		
		self.P_edge=nn.Sequential(nn.Linear(d_model*6, P_edge_hid, bias=True),
									nn.Tanh(),
									nn.Linear(P_edge_hid, 2, bias=True))
		

		self.logits_loss=nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
		self.block_loss=nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
		self.node_loss=nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
		self.edge_loss=nn.CrossEntropyLoss(reduction="mean", ignore_index=-1, weight=torch.tensor([1.0, 4.0]))
		self.loss_weight=loss_weight[:-1]

	def _block_match_(self, encode_output):
		left_encode=self.blockleft(encode_output[:, 1:-1:1])
		right_encode=self.blockright(encode_output[:, 1:-1:1])
		block_match=torch.matmul(left_encode, right_encode.transpose(1, 2))
		return block_match

	def _block_embedding_(self, source_len, encode_output, right_pos_truth):
		batch_size=source_len.shape[0]
		block_embedding=torch.zeros((batch_size, source_len.max(), self.d_model), device=source_len.device)

		for i in range(batch_size):
			for j in range(source_len[i]):
				block_embedding[i][j]=encode_output[i][1+j:1+right_pos_truth[i][j]+1].mean(dim=0)

		return block_embedding

	def _trace_embedding_(self, state_len, decode_output, target_offset):
		batch_size=state_len.shape[0]
		trace_embedding=torch.zeros((batch_size, state_len.max(), self.d_model), device=state_len.device)

		for i in range(batch_size):
			eos=target_offset[i][state_len[i]]
			for j in range(state_len[i]):
				trace_embedding[i][j]=decode_output[i][target_offset[i][j]:eos].mean(dim=0)

		return trace_embedding
	
	def _edge_embedding_(self, n_node, node_embedding, edge_index):
		batch_size=node_embedding.shape[0]
		hid_dim=node_embedding.shape[-1]

		# print(node_embedding.shape)
		# print(n_node)
		# print(edge_index)

		head=torch.repeat_interleave(node_embedding, 6, dim=1)
		tail=torch.gather(node_embedding, 1, torch.repeat_interleave(edge_index, hid_dim, dim=-1).reshape(batch_size, -1, hid_dim))
		edge_embedding=torch.cat((head, tail, head-tail), dim=-1).reshape(batch_size, n_node, 6, -1)

		return edge_embedding
	

	def forward(self,
				source,
				source_len,
				right_pos_truth,
				target,
				state_len,
				target_offset,
				node_label,
				edge_index,
				edge_label,
				log_file):

		batch_size=source.shape[0]
		max_source_len=source_len.max()
		max_state_len=state_len.max()
		device=source.device

		encode_output, decode_output, output_logits=self.transformer(source, target[:, :-1])
		loss_logits=self.logits_loss(output_logits.reshape(-1, self.n_tgt_vocab), target[:, 1:].reshape(-1))

		block_match=self._block_match_(encode_output)

		loss_block=self.block_loss(block_match.reshape(-1, block_match.shape[-1]), right_pos_truth.reshape(-1))

		#print("encode_output", encode_output, file=log_file)

		#print("decode_output", decode_output, file=log_file)

		block_embedding=self._block_embedding_(source_len, encode_output, right_pos_truth)

		#print("block_embedding", block_embedding, file=log_file)

		#print("target_offset", target_offset, file=log_file)

		trace_embedding=self._trace_embedding_(state_len, decode_output, target_offset)

		#print("trace_embedding", trace_embedding, file=log_file)

		node_embedding=torch.cat((torch.repeat_interleave(block_embedding, max_state_len, dim=1),
								torch.repeat_interleave(trace_embedding, max_source_len, dim=1)),
								dim=-1)

		node=self.P_node(node_embedding)
		loss_node=self.node_loss(node.reshape(-1, 3), node_label.reshape(-1))

		#edge_embedding=self._edge_embedding_(max_source_len*max_state_len, node_embedding, edge_index)

		#edge=self.P_edge(edge_embedding)
		#loss_edge=self.edge_loss(edge.reshape(-1, 2), edge_label.reshape(-1))

		#loss_total=torch.cat((loss_logits[None], loss_block[None], loss_node[None], loss_edge[None]))
		loss_total=torch.cat((loss_logits[None], loss_block[None], loss_node[None]))

		return output_logits, torch.dot(loss_total, self.loss_weight), loss_total, node_embedding, node

class Evaluator:
	def __init__(self,
					model,
					n_src_vocab,
					n_tgt_vocab,
					d_model,
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
		self.n_beam=n_beam
		self.max_seq_len=max_seq_len
		self.src_pad_idx=src_pad_idx
		self.tgt_pad_idx=tgt_pad_idx
		self.tgt_sos_idx=tgt_sos_idx
		self.tgt_semicolon_idx=tgt_semicolon_idx
		self.tgt_eos_idx=tgt_eos_idx
		self.len_penalty=len_penalty

	def beam_search(self, source):
		batch_size=source.shape[0]
		device=source.device
		source_beam=source.repeat_interleave(self.n_beam, dim=0)

		with torch.no_grad():
			scores=torch.zeros(batch_size*self.n_beam, device=device)
			done=torch.BoolTensor([False]*batch_size*self.n_beam, device=device)

			gen_seq=torch.full((batch_size*self.n_beam, 1), self.tgt_sos_idx, dtype=torch.long, device=device)

			for i in range(self.max_seq_len):
				if done.sum().item()==batch_size*self.n_beam:
					break

				_, _, output_logits=self.model.transformer(source_beam, gen_seq)
				best_k2_probs, best_k2_idx=output_logits[:, -1, :].topk(self.n_beam)
				scores=F.log_softmax(best_k2_probs, dim=-1).masked_fill(done[:, None], 0)+scores[:, None]

				neg_inf_mask = torch.BoolTensor([False]+[True]*(self.n_beam-1), device=device).repeat(batch_size*self.n_beam, 1).logical_and(done[:, None])
				scores=scores+torch.zeros(neg_inf_mask.shape, device=neg_inf_mask.device).masked_fill(neg_inf_mask, -1e6)

				scores, best_k_idx_in_k2=torch.topk(scores.view(batch_size, -1), self.n_beam, dim=-1, largest=True, sorted=True)

				scores=scores.view(-1)
				best_k_idx=torch.gather(best_k2_idx.view(batch_size, -1), 1, best_k_idx_in_k2).view(-1) #token
				best_r_idx=best_k_idx_in_k2.view(-1) // self.n_beam #beam resort
				done=done[best_r_idx]

				best_k_idx=best_k_idx.masked_fill(done, self.tgt_pad_idx)
				done=done.logical_or(best_k_idx==self.tgt_eos_idx)

				gen_seq=torch.cat((gen_seq[best_r_idx, :], best_k_idx.unsqueeze(dim=1)), dim=-1)

			target=gen_seq[torch.arange(0, batch_size*self.n_beam, self.n_beam).to(device), :]

			state_len=[]
			loop_start=[]
			target_offset=[]
			for x in target:
				target_offset.append([0])
				x_list=""
				for y in x:
					if y==self.tgt_eos_idx:
						break
					if y==self.tgt_semicolon_idx:
						x_list+=';'
					x_list+='#'

				loop_start.append(0)
				for y in x_list.split(';'):
					if '{' in y:
						loop_start[-1]=len(target_offset[-1])-1

					target_offset[-1].append(target_offset[-1][-1]+len(y)+1)
				target_offset[-1][-1]-=1
				state_len.append(len(target_offset[-1])-1)

			max_state_len=max(state_len)+1
			for i in range(batch_size):
				target_offset[i]+=[0]*(max_state_len-len(target_offset[i]))
				
			return target,\
					torch.tensor(state_len, dtype=torch.long, device=device),\
					torch.tensor(target_offset, dtype=torch.long, device=device),\
					torch.tensor(loop_start, dtype=torch.long, device=device)

	def gen_edge_index(self, source_len, state_len, right_pos, loop_start):
		batch_size=source_len.shape[0]
		max_source_len=source_len.max().item()
		max_state_len=state_len.max().item()
		n_node=max_source_len*max_state_len
		edge_index_list=[]

		# print(n_node)
		# print(state_len)
		# print(source_len)
		# print(right_pos)

		for batch_index in range(batch_size):
			edge_index_list.append([[0]*6 for _ in range(n_node)])
			for suffix in range(state_len[batch_index]):
				for left in range(source_len[batch_index]):
					index=suffix*max_source_len+left
					cnt=0
					for i in range(2):
						next_suffix=(suffix+i if suffix+i<state_len[batch_index] else loop_start[batch_index])
						
						edge_index_list[-1][index][cnt]=next_suffix*max_source_len+left
						cnt+=1
						if left+1<source_len[batch_index]:
							edge_index_list[-1][index][cnt]=next_suffix*max_source_len+left+1
							cnt+=1
							if right_pos[batch_index][left+1]+1<source_len[batch_index]:
								edge_index_list[-1][index][cnt]=next_suffix*max_source_len+right_pos[batch_index][left+1]+1
								cnt+=1
		return torch.tensor(edge_index_list, dtype=torch.long, device=source_len.device)

	def run(self, source, source_len):

		with torch.no_grad():
			target, state_len, target_offset, loop_start=self.beam_search(source)
			batch_size=source.shape[0]
			max_source_len=source_len.max()
			max_state_len=state_len.max()
			device=source.device

			encode_output, decode_output, output_logits=self.model.transformer(source, target[:, :-1])

			block_match=self.model._block_match_(encode_output)

			right_pos=block_match.argmax(dim=-1)

			edge_index=self.gen_edge_index(source_len, state_len, right_pos, loop_start)

			block_embedding=self.model._block_embedding_(source_len, encode_output, right_pos)

			trace_embedding=self.model._trace_embedding_(state_len, decode_output, target_offset)

			node_embedding=torch.cat((torch.repeat_interleave(block_embedding, max_state_len, dim=1),
									torch.repeat_interleave(trace_embedding, max_source_len, dim=1)),
									dim=-1)

			node=self.model.P_node(node_embedding).argmax(dim=-1)
			node_mask=node==2

			edge_embedding=self.model._edge_embedding_(max_source_len*max_state_len, node_embedding, edge_index)
			edge_choice_p=self.model.P_edge(edge_embedding).index_select(-1, torch.tensor(0, device=device)).squeeze(dim=-1)

			edge, edge_choice=torch.topk(edge_choice_p, 2, dim=-1, largest=True, sorted=True)
			edge_choice=torch.gather(edge_index, -1, edge_choice)

		max_source_len=source_len.max()
		n_node=source_len.max()*state_len.max()

		proof=[]
		for batch_index in range(batch_size):
			sub_proof=[]
			q_node=[0]
			q_head=0
			while q_head<len(q_node):
				cur=q_node[q_head]
				q_head+=1
				if node_mask[batch_index][cur]!=2:
					for i in range(2):
						son=edge_choice[batch_index][cur][i]
						if node_mask[batch_index][son]!=2 and son not in q_node:
							q_node.append(son)
							x=[son//max_source_len, [son%max_source_len, right_pos[batch_index][son%max_source_len]], node_mask[batch_index][son]]
							x[0], x[1][0], x[1][1], x[2]=x[0].item(), x[1][0].item(), x[1][1].item(), x[2].item()
							sub_proof.append(x)
			proof.append(sub_proof)

		return target.cpu().detach().numpy().tolist(), proof