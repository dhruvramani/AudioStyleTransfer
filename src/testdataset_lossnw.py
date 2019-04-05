from torch.utils.data import Dataset
import numpy as np 
import torch
import librosa
import os
import pickle
import matplotlib.pyplot as plt

N_FFT = 1024
class InstrumentsTestDataset(Dataset):
    def __init__(self, src='/home/nevronas/dataset/IRMAS-TestingData-Part1/Part1', load=False):
        super(InstrumentsTestDataset, self).__init__()
        
        self.instx_spec = []
        self.instx_phase = []
        self.insty = []

        insts = {
                'cel' : 0, # Cello
                'cla' : 1, # Clarinet
                'flu' : 2, # Flute
                'gac' : 3, # Acoustic guitar
                'gel' : 4, # Electric guitar
                'org' : 5, # Organ
                'pia' : 6, # Piano
                'sax' : 7, # Saxophone
                'tru' : 8, # Trumpet
                'vio' : 9 # Violin
            }

        if(load):
            
            files = os.listdir(os.path.join(src))
            for f in files:
                if '.txt' in f:
                    continue
                print(f)
                signal, fs = librosa.load(os.path.join(src, f))

                try:
                    fin = open(os.path.join(src, f[:-4] + ".txt"), "r")
                    labels = [insts[str(i[0:3])] for i in fin.readlines()]
                    self.insty.append(labels)
                    print(labels)
                    fin.close()
                except Exception as e:
                    print(str(e))
                    continue

                D = librosa.stft(signal, N_FFT)
                S, phase = librosa.magphase(D)
                S = np.log1p(S)
                S = S[:, 0:250]
                S = np.expand_dims(S, 0)
                print(S.shape)
                self.instx_spec.append(S)
                self.instx_phase.append(phase)
            file_handler = open('../pickle/insts_pickled_test.dat', 'wb+')
            pickle.dump((self.instx_spec, self.instx_phase, self.insty), file_handler)
            file_handler.close()

        else:

            file_handler = open('../pickle/insts_pickled_test.dat', 'rb+')
            self.instx_spec, self.instx_phase, self.insty = pickle.load(file_handler)
            file_handler.close()

    def __len__(self):
        return len(self.instx_spec)

    def __getitem__(self, idx):
        return self.instx_spec[idx], self.insty[idx]
        
if __name__ == '__main__':
    dataset = InstrumentsTestDataset(src='/home/nevronas/dataset/IRMAS-TestingData-Part1/Part1', load=True)
