import os
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model import Model_with_Proof, Evaluator
from utils import LTL_Dataset, input_collate_fn_train, input_collate_fn_test, convert_to_cuda, Accuracy, index_to_sentence
from tqdm import tqdm

def run_train(config):
	with open(config["LTL_vocab"], "r") as f:
		index_to_LTL=[x.strip() for x in f]
	with open(config["trace_vocab"], "r") as f:
		index_to_trace=[x.strip() for x in f]

	LTL_to_index={x:i for i, x in enumerate(index_to_LTL)}
	trace_to_index={x:i for i, x in enumerate(index_to_trace)}

	device=config["device"]
	model_path=config["model_path"]
	log_file=open(os.path.join(model_path, "model.log"), "w")

	train_data=LTL_Dataset(config["data_file"], LTL_to_index, trace_to_index)
	'''
	train_loader=DataLoader(train_data, batch_size=config["batch_size"], shuffle=False, collate_fn=input_collate_fn_train)

	for data in train_loader:
		pass

	exit(0)
	'''
	
	model=Model_with_Proof(n_src_vocab=len(LTL_to_index),
							n_tgt_vocab=len(trace_to_index),
							d_model=config["d_model"],
							nhead=config["nhead"],
							nlayers=config["nlayers"],
							nhid=config["nhid"],
							dropout=config["dropout"],
							d_block=config["d_block"],
							P_node_hid=config["P_node_hid"],
							P_edge_hid=config["P_edge_hid"],
							loss_weight=torch.FloatTensor(config["loss_weight"]))

	if config["model_file"] is not None:
		model_dict=model.state_dict()
		pretrained_dict=torch.load(config["model_file"])
		pretrained_dict={key:value for key, value in pretrained_dict.items() if key in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		
	model.to(device)

	optimizer=Adam(model.parameters(), lr=config["lr"])

	epochs=config["epochs"]
	batch_size=config["batch_size"]

	model.train()

	for epoch in range(epochs):
		print("epoch: ", epoch)
		print("epoch: ", epoch, file=log_file)
		
		train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=input_collate_fn_train)
		acc_count=0
		count=0

		for data in train_loader:
			cuda_data=convert_to_cuda(data, device)
			output, loss=model(cuda_data["source"],
								cuda_data["source_len"],
								cuda_data["right_pos_truth"],
								cuda_data["target"],
								cuda_data["state_len"],
								cuda_data["target_offset"],
								cuda_data["node_label"],
								cuda_data["edge_index"],
								cuda_data["edge_label"])

			x, y=Accuracy(output, cuda_data["target"][:, 1:], trace_to_index["[PAD]"])
			acc_count+=x
			count+=y
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		torch.save(model.state_dict(), os.path.join(model_path, "model"+str(epoch)+".pkl"))
		print("accuracy: ", acc_count/count)
		print("accuracy: ", acc_count/count, file=log_file)

def run_val(config):
	with open(config["LTL_vocab"], "r") as f:
		index_to_LTL=[x.strip() for x in f]
	with open(config["trace_vocab"], "r") as f:
		index_to_trace=[x.strip() for x in f]

	LTL_to_index={x:i for i, x in enumerate(index_to_LTL)}
	trace_to_index={x:i for i, x in enumerate(index_to_trace)}

	device=config["device"]
	model_file=config["model_file"]

	test_data=LTL_Dataset(config["data_file"], LTL_to_index, trace_to_index)
	
	model=Model_with_Proof(n_src_vocab=len(LTL_to_index),
							n_tgt_vocab=len(trace_to_index),
							d_model=config["d_model"],
							nhead=config["nhead"],
							nlayers=config["nlayers"],
							nhid=config["nhid"],
							dropout=config["dropout"],
							d_block=config["d_block"],
							P_node_hid=config["P_node_hid"],
							P_edge_hid=config["P_edge_hid"],
							loss_weight=torch.FloatTensor(config["loss_weight"]).to(device))

	model.load_state_dict(torch.load(model_file))
	model.to(device)

	evaluator=Evaluator(model=model,
					n_src_vocab=len(index_to_LTL),
					n_tgt_vocab=len(index_to_trace),
					d_model=config["d_model"],
					n_beam=config["n_beam"],
					max_seq_len=config["max_seq_len"],
					src_pad_idx=LTL_to_index["[PAD]"],
					tgt_pad_idx=trace_to_index["[PAD]"],
					tgt_sos_idx=trace_to_index["[SOS]"],
					tgt_semicolon_idx=trace_to_index[";"],
					tgt_eos_idx=trace_to_index["[EOS]"],
					len_penalty=1.0)

	batch_size=config["batch_size"]
		
	test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=input_collate_fn_test)

	output=[]
	
	for data in test_loader:
		cuda_data=convert_to_cuda(data, device)
		target, proof=evaluator.run(source=cuda_data["source"], source_len=cuda_data["source_len"])
		target=index_to_sentence(target, index_to_trace)
		output.append([{"ltl_pre":test_data.raw_data[x]["ltl_pre"], "trace":y, "proof":z} for x, y, z in zip(data["id"], target, proof)])

	with open(os.path.join(config["result_path"], "res-"+os.path.basename(model_file).split(".")[0]+"-"+os.path.basename(config["data_file"])), "w") as f:
		json.dump(output, fp=f, indent=4)

if __name__=="__main__":

	parser = ArgumentParser(description='Proof')

	parser.add_argument('--data_file', type=str, required=True)
	parser.add_argument('--LTL_vocab', type=str, default='LTL_vocab.txt')
	parser.add_argument('--trace_vocab', type=str, default='trace_vocab.txt')
	parser.add_argument('--model_path', type=str, default='model')
	parser.add_argument('--result_path', type=str, default='result')
	parser.add_argument('--model_file', type=str, default=None)

	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

	parser.add_argument('--d_model', type=int, default=128)
	parser.add_argument('--nlayers', type=int, default=8)
	parser.add_argument('--nhead', type=int, default=4)
	parser.add_argument('--nhid', type=int, default=512)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--d_block', type=int, default=256)
	parser.add_argument('--P_node_hid', type=int, default=512)
	parser.add_argument('--P_edge_hid', type=int, default=512)
	parser.add_argument('--n_beam', type=int, default=5)
	parser.add_argument('--loss_weight', type=int, nargs='+', default=[3, 3, 2, 1])

	parser.add_argument('--lr', type=float, default=0.00025)

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--epochs', type=int, default=150)
	parser.add_argument('--is_train', type=int, required=True)
	parser.add_argument('--max_seq_len', type=int, default=100)
	
	args = parser.parse_args()

	config=vars(args)

	if config["is_train"]==1:
		run_train(config)
	else:
		run_val(config)
