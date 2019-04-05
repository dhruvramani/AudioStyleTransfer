from torch.utils.data import Dataset
import numpy as np 
import torch
import librosa
import os
import pickle
import matplotlib.pyplot as plt

N_FFT = 1024
class InstrumentsDataset(Dataset):
    def __init__(self, src='/home/nevronas/dataset/IRMAS-TrainingData', load=False, test=False):
        super(InstrumentsDataset, self).__init__()
        
        self.inst_spec_train = []
        self.inst_label_train = []
        self.inst_spec_test = []
        self.inst_label_test = []
        self.test = test

        insts = [
                'cel', # Cello
                'cla', # Clarinet
                'flu', # Flute
                'gac', # Acoustic guitar
                'gel', # Electric guitar
                'org', # Organ
                'pia', # Piano
                'sax', # Saxophone
                'tru', # Trumpet
                'vio'  # Violin
            ]

        if(load):
            label = 0
            for inst in insts:
                curr_inst_spec_train = []
                curr_inst_label_train = []
                curr_inst_spec_test = []
                curr_inst_label_test = []

                files = os.listdir(os.path.join(src, inst))
                i = 0
                for f in files:
                    print(f)
                    signal, fs = librosa.load(os.path.join(os.path.join(src, inst), f))
                    S = librosa.feature.mfcc(signal)
                    S = np.expand_dims(S, 0)
                    print(S.shape)
                    if(i < int(0.8*len(files))):
                        curr_inst_spec_train.append(S)
                        curr_inst_label_train.append(label)
                    else:
                        curr_inst_spec_test.append(S)
                        curr_inst_label_test.append(label)
                    i = i+1
                file_handler = open('../pickle/insts_train_pickled_{}.dat'.format(inst), 'wb+')
                pickle.dump((curr_inst_spec_train, curr_inst_label_train), file_handler)
                file_handler.close()
                file_handler = open('../pickle/insts_test_pickled_{}.dat'.format(inst), 'wb+')
                pickle.dump((curr_inst_spec_test, curr_inst_label_test), file_handler)
                file_handler.close()
                label = label + 1

        else:

            for inst in insts:
                if(not self.test):
                    file_handler = open('../pickle/insts_train_pickled_{}.dat'.format(inst), 'rb+')
                    curr_inst_spec, curr_inst_label = pickle.load(file_handler)
                    file_handler.close()
                    self.inst_spec_train.extend(curr_inst_spec)
                    self.inst_label_train.extend(curr_inst_label)
                else:
                    file_handler = open('../pickle/insts_test_pickled_{}.dat'.format(inst), 'rb+')
                    curr_inst_spec, curr_inst_label = pickle.load(file_handler)
                    file_handler.close()
                    self.inst_spec_test.extend(curr_inst_spec)
                    self.inst_label_test.extend(curr_inst_label)

    def __len__(self):
        if(not self.test):
            return len(self.inst_spec_train)
        else:
            return len(self.inst_spec_test)

    def __getitem__(self, idx):
        if(not self.test):
            return self.inst_spec_train[idx], self.inst_label_train[idx]
        else:
            return self.inst_spec_test[idx], self.inst_label_test[idx]
        
if __name__ == '__main__':
    dataset = InstrumentsDataset(src='/home/nevronas/dataset/IRMAS-TrainingData', load=True)
