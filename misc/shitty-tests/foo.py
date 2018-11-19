import torch
import torchaudio
from feature import *

vctk_data = torchaudio.datasets.VCTK('~/dataset/', download=True)
data_loader = torch.utils.data.DataLoader(vctk_data, batch_size=1, shuffle=True, num_workers=2)

for i, (inp, targets) in enumerate(data_loader):
	print(i, end="\r")
