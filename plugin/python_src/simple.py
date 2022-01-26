import torch
import torch.nn as nn

class MyNN(nn.Module):

	def __init__(self, hidden_dim:int):
		super(MyNN, self).__init__()

		self.seq = nn.Sequential(
		nn.Linear(1, hidden_dim),
		nn.ReLU(),
		nn.Linear(hidden_dim, 1)
	)

	def predict(self, x:int)->int:
		t = torch.tensor([x]).float()
		return self.seq(t).item()