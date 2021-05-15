import os
import json
import torch
from torch.utils.data import Dataset

def convert_to_cuda(x, device):
	return {key: value.to(device) for key, value in x.items()}

class LTL_Dataset(Dataset):
	def __init__(self, data_file, LTL_dict, trace_dict):
		super(LTL_Dataset, self).__init__()

		self.LTL_dict=LTL_dict
		self.trace_dict=trace_dict
		self.raw_data=[]
		'''
		for file_name in os.listdir(data_dir):
			with open(os.path.join(data_dir, file_name), "r") as f:
				self.raw_data.extend(json.load(f))
		'''
		with open(data_file, "r") as f:
			self.raw_data=json.load(f)

	def __getitem__(self, item):
		cur=self.raw_data[item]
		ret={}
		ret["id"]=item
		ret["source"]=[self.LTL_dict["[CLS]"]]+sentence_to_index(cur["ltl_pre"], self.LTL_dict)+[self.LTL_dict["[EOS]"]]
		ret["source_len"]=len(ret["source"])-2
		ret["right_pos_truth"]=[pos for pos in cur["pair_set"]]
		ret["target"]=[self.trace_dict["[SOS]"]]+sentence_to_index(cur["trace"], self.trace_dict)+[self.trace_dict["[EOS]"]]
		ret["loop_start"]=0

		ret["target_offset"]=[0]
		for state in cur["trace"].split(";"):
			if '{' in state:
				ret["loop_start"]=len(ret["target_offset"])-1

			ret["target_offset"].append(ret["target_offset"][-1]+len(state)+1)
		ret["target_offset"][-1]-=1

		ret["state_len"]=len(ret["target_offset"])-1
		ret["proof"]=cur["proof"]

		return ret

	def __len__(self):
		return len(self.raw_data)

def input_collate_fn_train(batch_data):
	ret={"id":[],
			"source":[],
			"source_len":[],
			"right_pos_truth":[],
			"target":[],
			"state_len":[],
			"target_offset":[],
			"node_label":[],
			"edge_index":[],
			"edge_label":[]
			}

	source_maxlen=0
	target_maxlen=0
	state_maxlen=0

	for cur in batch_data:
		source_maxlen=max(source_maxlen, cur["source_len"])
		target_maxlen=max(target_maxlen, len(cur["target"]))
		state_maxlen=max(state_maxlen, cur["state_len"])

	node_maxlen=source_maxlen*state_maxlen
	
	for cur in batch_data:
		ret["id"].append(cur["id"])
		ret["source"].append(cur["source"]+[0]*(source_maxlen+2-len(cur["source"])))
		ret["source_len"].append(cur["source_len"])
		ret["right_pos_truth"].append(cur["right_pos_truth"]+[-1]*(source_maxlen-len(cur["right_pos_truth"])))
		ret["target"].append(cur["target"]+[0]*(target_maxlen-len(cur["target"])))
		ret["state_len"].append(cur["state_len"])
		ret["target_offset"].append(cur["target_offset"]+[0]*(state_maxlen+1-len(cur["target_offset"])))
		ret["node_label"].append([-1]*node_maxlen)
		ret["edge_index"].append([[(0, 0)]*6 for _ in range(node_maxlen)])
		ret["edge_label"].append([[-1]*6 for _ in range(node_maxlen)])

		for x, y in cur["proof"]:
			x_index=x[0]*source_maxlen+x[1][0]
			y_index=y[0]*source_maxlen+y[1][0]
			ret["node_label"][-1][x_index]=x[2]
			ret["node_label"][-1][y_index]=y[2]

		for suffix in range(cur["state_len"]):
			for left in range(cur["source_len"]):
				index=suffix*source_maxlen+left
				
				if ret["node_label"][-1][index]==-1:
					ret["node_label"][-1][index]=2
					continue

				cnt=0
				for i in range(2):
					next_suffix=(suffix+i if suffix+i<cur["state_len"] else cur["loop_start"])

					ret["edge_index"][-1][index][cnt]=(next_suffix, left)
					cnt+=1
					if left+1<cur["source_len"]:
						ret["edge_index"][-1][index][cnt]=(next_suffix, left+1)
						cnt+=1
						if cur["right_pos_truth"][left+1]+1<cur["source_len"]:
							ret["edge_index"][-1][index][cnt]=(next_suffix, cur["right_pos_truth"][left+1]+1)
							cnt+=1
				ret["edge_label"][-1][index][:cnt]=[0]*cnt

		for x, y in cur["proof"]:
			y_index=y[0]*source_maxlen+y[1][0]
			ret["edge_label"][-1][y_index][ret["edge_index"][-1][y_index].index((x[0], x[1][0]))]=1

		for node_index in range(len(ret["edge_index"][-1])):
			for son_index in range(len(ret["edge_index"][-1][node_index])):
				x, y=ret["edge_index"][-1][node_index][son_index]
				ret["edge_index"][-1][node_index][son_index]=x*source_maxlen+y

	return {key:torch.tensor(value, dtype=torch.long) for key, value in ret.items()}

def input_collate_fn_test(batch_data):
	ret={"id":[], "source":[], "source_len":[]}

	source_maxlen=0

	for cur in batch_data:
		source_maxlen=max(source_maxlen, len(cur["source"]))
	
	for cur in batch_data:
		ret["id"].append(cur["id"])
		ret["source"].append(cur["source"]+[0]*(source_maxlen-len(cur["source"])))
		ret["source_len"].append(cur["source_len"])

	return {key:torch.tensor(value, dtype=torch.long) for key, value in ret.items()}

def index_to_sentence(target, index_to_trace):
	ret=[]
	for x in target:
		sub=[]
		for y in x:
			if index_to_trace[y]=="[EOS]": break
			sub.append(index_to_trace[y])
		ret.append("".join(sub))

	return ret

def sentence_to_index(sentence, sentence_to_index):
	ret=[]
	rest=""
	for x in sentence:
		rest+=x
		if rest in sentence_to_index:
			ret.append(sentence_to_index[rest])
			rest=""

	assert(rest=="")

	return ret

def Accuracy(outputs, targets, pad_index):
    batch_size, seq_len, vocabulary_size = outputs.size()

    outputs = outputs.reshape(batch_size * seq_len, vocabulary_size)
    targets = targets.reshape(batch_size * seq_len)

    predicts = outputs.argmax(dim=1)
    corrects = predicts == targets

    corrects.masked_fill_((targets == pad_index), 0)

    correct_count = corrects.sum().item()
    count = (targets != pad_index).sum().item()

    return correct_count, count