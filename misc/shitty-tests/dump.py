import torch
import torchaudio
from vctk import VCTK
from feature import *
import matplotlib.pyplot as plot
import numpy as np

vctk_data = VCTK('~/dataset/', download=True)
data_loader = torch.utils.data.DataLoader(vctk_data, batch_size=1, shuffle=True, num_workers=1)

print(len(vctk_data.labels))
#print(len(vctk_data.list_IDs))
print(vctk_data.__len__())
for i, (inp, targets) in enumerate(data_loader):
	if(i > 0):
		break
	print(i, end="\r")
	
	inputs = inp.numpy()
	print(inp.size(), str(targets[0]))
	inputs = inputs.flatten()
	stft = transform_stft(inputs)
	print(stft.shape)
	
	plot.subplot(221)
	plot.plot(inputs)
	plot.xlabel("Freq")
	plot.ylabel("Time")
	#plot.savefig("foo.png")
	#plot.show()
	
	plot.subplot(222)
	powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(inputs,Fs= 400)
	plot.xlabel('Time')
	plot.ylabel('Frequency')	
	#plot.show()

	plot.subplot(223)
	plot.plot(stft)
	plot.xlabel('Sample')
	plot.ylabel('Amplitude')
	#plot.show()
		
	
	plot.subplot(224)
	powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(stft.flatten(),Fs= 400)
	plot.xlabel('Time')
	plot.ylabel('Frequency')
	#plot.savefig("foo.png")
	plot.show() 



