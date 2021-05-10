import os
import json
import torch
from torch.utils.data import Dataset

def convert_to_cuda(x, device):
	return {key: value.to(device) for key, value in x.items()}

class LTL_Dataset(Dataset):
	def __init__(self, data_dir, LTL_dict, trace_dict):
		super(LTL_Dataset, self).__init__()

		self.LTL_dict=LTL_dict
		self.trace_dict=trace_dict
		self.raw_data=[]

		for file_name in os.listdir(data_dir):
			with open(os.path.join(data_dir, file_name), "r") as f:
				lines=json.load(f)
			self.raw_data.extend([{"ltl_pre":line["ltl_pre"],
									"trace":line["trace"],
									"proof":line["proof"],
									"pair_set":line["pair_set"]}
									for line in lines])

	def __getitem__(self, item):
		cur=self.raw_data[item]
		ret={}
		ret["source"]=[self.LTL_dict["CLS"]]+[self.LTL_dict[x] for x in cur["ltl_pre"]]+[self.LTL_dict["EOS"]]
		ret["source_len"]=len(cur["source"])
		ret["right_pos_truth"]=[pos for pos in cur["pair_set"]]
		ret["target"]=[self.trace_dict["CLS"]]+[self.trace_dict[x] for x in cur["trace"]]+[self.trace_dict["EOS"]]

		ret["target_offset"]=[0]
		for state in range(cur["trace"].split(";")):
			ret["target_offset"].append(len(state)+1)
		ret["target_offset"][-1]-=1

		ret["target_len"]=len(ret["target_offset"])-1
		ret["proof"]=cur["proof"]

		return ret

	def __len__(self):
		return len(self.raw_data)

def input_collate_fn(batch_data):
	ret={"source":[],
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
	node_maxlen=source_maxlen*target_maxlen

	for cur in batch_data:
		source_maxlen=max(source_maxlen, len(cur["source"]))
		target_maxlen=max(target_maxlen, len(cur["target"]))
		state_maxlen=max(state_maxlen, cur["target_len"])
	
	for cur in batch_data:
		ret["source"].append(cur["source"]+[0]*(source_maxlen-len(cur["source"])))
		ret["source_len"].append(cur["source_len"])
		ret["right_pos_truth"].append(cur["right_pos_truth"]+[-1]*(source_maxlen-len(cur["right_pos_truth"])))
		ret["target"].append(cur["target"]+[0]*(target_maxlen-len(cur["target"])))
		ret["target_len"].append(cur["target_len"])
		ret["target_offset"].append(cur["target_offset"]+[0]*(state_maxlen-len(cur["target_offset"])))
		ret["node_label"].append([-1]*node_maxlen)
		ret["edge_index"].append([[0]*6 for _ in range(node_maxlen)])
		ret["edge_label"].append([[-1]*6 for _ in range(node_maxlen)])

		for suffix in range(cur["target_len"]):
			for left in range(cur["source_len"]):
				index=suffix*cur["source_len"]+left
				ret["node_label"][-1][index]=2
				cnt=0
				for i in range(2):
					if suffix+i<cur["target_len"]:
						ret["edge_index"][-1][index][cnt]=(suffix+i)*cur["target_len"]+left
						cnt+=1
						if left+1<cur["source_len"]:
							ret["edge_index"][-1][index][cnt]=(suffix+i)*cur["target_len"]+left+1
							cnt+=1
							if cur["right_pos_truth"][left+1]+1<cur["source_len"]:
								ret["edge_index"][-1][index][cnt]=(suffix+i)*cur["target_len"]+cur["right_pos_truth"][left+1]+1
								cnt+=1
				ret["edge_label"][-1][index][:cnt]=[0]*cnt

		for x, y in cur["proof"]:
			x_index=x[0]*cur["source_len"]+x[1][0]
			y_index=y[0]*cur["source_len"]+y[1][0]
			ret["node_label"][-1][x_index]=x[2]
			ret["node_label"][-1][y_index]=y[2]
			ret["edge_label"][-1][y_index][ret["edge_index"][-1][y_index].index(x_index)]=1

	return {key:torch.IntTensor(value) for key, value in ret.items()}