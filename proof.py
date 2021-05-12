import os
from utils import LTL_Dataset, input_collate_fn, convert_to_cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model import Model_with_Proof
from tqdm import tqdm

def run_train(config):
	with open(config["LTL_vocab"], "r") as f:
		index_to_LTL=[x.strip() for x in f]
	with open(config["trace_vocab"], "r") as f:
		index_to_trace=[x.strip() for x in f]

	LTL_to_index={x:i for i, x in enumerate(index_to_LTL)}
	trace_to_index={x:i for i, x in enumerate(trace_to_LTL)}

	device=config["device"]
	log_file=open(os.path.join(model_path, "model.log"), "w")
	model_path=config["model_path"]

	train_data=LTL_Dataset(config["data_file"], LTL_to_index, trace_to_index)
	
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
	model.to(device)

	optimizer=Adam(model.parameters(), lr=config["lr"])

	epochs=config["epochs"]
	batch_size=config["batch_size"]

	model.train()

	for epoch in range(epochs):
		print("epoch: ", epoch)
		print("epoch: ", epoch, file=log_file)
		
		train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=input_collate_fn)
		
		for data in train_loader:
			cuda_data=convert_to_cuda(data, device)
			logits, loss=model(cuda_data["source"],
								cuda_data["source_len"],
								cuda_data["right_pos_truth"],
								cuda_data["target"],
								cuda_data["state_len"],
								cuda_data["target_offset"],
								cuda_data["node_label"],
								cuda_data["edge_index"],
								cuda_data["edge_label"])
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		torch.save(model.state_dict(), os.path.join(model_path, "model"+str(epoch)+".pkl"))
		print("loss: ", loss.data)
		print("loss: ", loss, file=log_file)
		print("label_acc: ", label_acc)
		print("label_acc: ", label_acc, file=log_file)

def run_val(config):
	pass

if __name__=="__main__":

	parser = ArgumentParser(description='Proof')

	parser.add_argument('--data_file', type=str, require=True)
	parset.add_argument('--LTL_vocab', type=str, default='LTL_vocab.txt')
	parset.add_argument('--trace_vocab', type=str, default='trace_vocab.txt')
	parser.add_argument('--model_path', type=str, default=None)

	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

	parser.add_argument('--d_model', type=int, default=128)
	parser.add_argument('--nlayers', type=int, default=8)
	parser.add_argument('--nhead', type=int, default=4)
	parser.add_argument('--nhid', type=int, default=512)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--d_block', type=int, default=256)
	parser.add_argument('--P_node_hid', type=int, default=512)
	parser.add_argument('--P_edge_hid', type=int, default=512)
	parser.add_argument('--loss_weight', type=int, nargs='+', default=[3, 3, 2, 1])

	parser.add_argument('--lr', type=float, default=0.00025)

	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=150)
	parser.add_argument('--is_train', type=bool, require=True)
	
	args = parser.parse_args()

	config=vars(args)

	if config["is_train"]:
		run_train(config)
	else:
		run_val(config)