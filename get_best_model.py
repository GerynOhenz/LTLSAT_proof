import os

with open("model.log", "r") as f:
	lines=[x.strip() for x in f]

best_epoch=0
best_acc=-1

for index in range(0, len(lines), 3):
	epoch=int(lines[index].split(':')[1])
	acc=float(lines[index+1].split(':')[1])
	if acc>best_acc:
		best_epoch, best_acc=epoch, acc

print(best_epoch)
print(best_acc)