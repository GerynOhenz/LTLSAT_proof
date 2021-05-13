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
		ret["source"]=[self.LTL_dict["CLS"]]+[self.LTL_dict[x] for x in cur["ltl_pre"]]+[self.LTL_dict["EOS"]]
		ret["source_len"]=len(cur["ltl_pre"])
		ret["right_pos_truth"]=[pos for pos in cur["pair_set"]]
		ret["target"]=[self.trace_dict["SOS"]]+[self.trace_dict[x] for x in cur["trace"]]+[self.trace_dict["EOS"]]

		ret["target_offset"]=[0]
		for state in range(cur["trace"].split(";")):
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
			"target_len":[],
			"target_offset":[],
			"node_label":[],
			"edge_index":[],
			"edge_label":[]
			}

	source_maxlen=0
	target_maxlen=0
	state_maxlen=0

	for cur in batch_data:
		source_maxlen=max(source_maxlen, len(cur["source"]))
		target_maxlen=max(target_maxlen, len(cur["target"]))
		state_maxlen=max(state_maxlen, cur["state_len"])
	
	state_maxlen+=1 #EOS
	node_maxlen=source_maxlen*state_maxlen
	
	for cur in batch_data:
		ret["id"].append(cur["id"])
		ret["source"].append(cur["source"]+[0]*(source_maxlen-len(cur["source"])))
		ret["source_len"].append(cur["source_len"])
		ret["right_pos_truth"].append(cur["right_pos_truth"]+[-1]*(source_maxlen-len(cur["right_pos_truth"])))
		ret["target"].append(cur["target"]+[0]*(target_maxlen-len(cur["target"])))
		ret["state_len"].append(cur["state_len"])
		ret["target_offset"].append(cur["target_offset"]+[0]*(state_maxlen-len(cur["target_offset"])))
		ret["node_label"].append([-1]*node_maxlen)
		ret["edge_index"].append([[0]*6 for _ in range(node_maxlen)])
		ret["edge_label"].append([[-1]*6 for _ in range(node_maxlen)])

		for suffix in range(cur["state_len"]):
			for left in range(cur["source_len"]):
				index=suffix*source_maxlen+left
				ret["node_label"][-1][index]=2
				cnt=0
				for i in range(2):
					if suffix+i<cur["state_len"]:
						ret["edge_index"][-1][index][cnt]=(suffix+i)*source_maxlen+left
						cnt+=1
						if left+1<cur["source_len"]:
							ret["edge_index"][-1][index][cnt]=(suffix+i)*source_maxlen+left+1
							cnt+=1
							if cur["right_pos_truth"][left+1]+1<cur["source_len"]:
								ret["edge_index"][-1][index][cnt]=(suffix+i)*source_maxlen+cur["right_pos_truth"][left+1]+1
								cnt+=1
				ret["edge_label"][-1][index][:cnt]=[0]*cnt

		for x, y in cur["proof"]:
			x_index=x[0]*source_maxlen+x[1][0]
			y_index=y[0]*source_maxlen+y[1][0]
			ret["node_label"][-1][x_index]=x[2]
			ret["node_label"][-1][y_index]=y[2]
			ret["edge_label"][-1][y_index][ret["edge_index"][-1][y_index].index(x_index)]=1

	return {key:torch.IntTensor(value) for key, value in ret.items()}

def input_collate_fn_test(batch_data):
	ret={"id":[], "source":[], "source_len":[]}

	source_maxlen=0

	for cur in batch_data:
		source_maxlen=max(source_maxlen, len(cur["source"]))
	
	for cur in batch_data:
		ret["id"].append(cur["id"])
		ret["source"].append(cur["source"]+[0]*(source_maxlen-len(cur["source"])))
		ret["source_len"].append(cur["source_len"])

	return {key:torch.IntTensor(value) for key, value in ret.items()}

def index_to_sentence(target, index_to_trace):
	ret=[]
	for x in target:
		sub=[]
		for y in x:
			if index_to_trace[y]=="EOS": break
			sub.append(index_to_trace[y])
		ret.append("".join(sub))

	return ret

def Accuracy(outputs, targets, pad_index):
    batch_size, seq_len, vocabulary_size = outputs.size()

    outputs = outputs.view(batch_size * seq_len, vocabulary_size)
    targets = targets.view(batch_size * seq_len)

    predicts = outputs.argmax(dim=1)
    corrects = predicts == targets

    corrects.masked_fill_((targets == pad_index), 0)

    correct_count = corrects.sum().item()
    count = (targets != pad_index).sum().item()

    return correct_count, count