from matplotlib import pyplot
import numpy as np
import pickle as p
net1 = None
with open('mlp_model.pickle', 'rb') as f:
	net1 = p.load(f)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(2.0, 1.0)
pyplot.yscale("log")
pyplot.show()
