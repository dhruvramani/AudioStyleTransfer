from torch.utils.data import Dataset
import numpy as np 
import torch
import librosa
import os
import pickle
import matplotlib.pyplot as plt

N_FFT = 1024
class VoicesDataset(Dataset):
    def __init__(self, src='/home/nevronas/dataset/accent-new', load=False):
        super(VoicesDataset, self).__init__()
        
        self.voice_spec = []
        self.voice_phase = []

        voices = [
                'english-new', 
                'french-new', 
                'german-new'
            ]

        if(load):
            for voice in voices:
                curr_voice_spec = []
                curr_voice_phase = []
                files = os.listdir(os.path.join(src, voice))
                for f in files:
                    print(f)
                    signal, fs = librosa.load(os.path.join(os.path.join(src, voice), f))
                    D = librosa.stft(signal, N_FFT)
                    S, phase = librosa.magphase(D)
                    S = np.log1p(S)
                    S = S[:, 0:250]
                    S = np.expand_dims(S, 0)

                    if S.shape[2] < 250:
                        print("Chuck this sample!")
                        continue

                    print(S.shape)
                    curr_voice_spec.append(S)
                    curr_voice_phase.append(phase)
                file_handler = open('../pickle/voice_pickled_{}.dat'.format(voice), 'wb+')
                pickle.dump((curr_voice_spec, curr_voice_phase), file_handler)
                file_handler.close()

        else:

            for voice in voices:
                file_handler = open('../pickle/voice_pickled_{}.dat'.format(voice), 'rb+')
                curr_voice_spec, curr_voice_phase = pickle.load(file_handler)
                file_handler.close()
                self.voice_spec.extend(curr_voice_spec)
                self.voice_phase.extend(curr_voice_phase)

    def __len__(self):
        return len(self.instx_spec)

    def __getitem__(self, idx):
        return self.voice_spec[idx], self.voice_phase[idx]

class VCCDataset(Dataset):
    def __init__(self, load=False, transform=None):
        super(VCCDataset, self).__init__()
        
        self.voice_wav = []
        self.voice_fs = []
        self.transform = transform

        if(load):

            folders = os.listdir('/home/nevronas/dataset/vcc2018_training')

            for folder in folders:
                if 'VCC' in folder:
                    files = os.listdir(os.path.join('/home/nevronas/dataset/vcc2018_training', folder))
                    curr_voice_wav = []
                    curr_voice_fs = []
                    for f in files:
                        print(f)
                        signal, fs = librosa.load(os.path.join(os.path.join('/home/nevronas/dataset/vcc2018_training', folder), f))
                        curr_voice_wav.append(signal)
                        curr_voice_fs.append(fs)
                    file_handler = open('../pickle/voice_pickled_{}.dat'.format(folder), 'wb+')
                    pickle.dump((curr_voice_wav, curr_voice_fs), file_handler)
                    file_handler.close()

        else:

            folders = os.listdir('/home/nevronas/dataset/vcc2018_training')

            for folder in folders:
                if 'VCC' in folder:
                    file_handler = open('../pickle/voice_pickled_{}.dat'.format(folder), 'rb+')
                    curr_voice_wav, curr_voice_fs = pickle.load(file_handler)
                    file_handler.close()
                    self.voice_wav.extend(curr_voice_wav)
                    self.voice_fs.extend(curr_voice_fs)
            print(len(self.voice_wav))

    def __len__(self):
        return len(self.voice_wav)

    def __getitem__(self, idx):
        audio = self.voice_wav[idx]
        if self.transform is not None:
                audio = self.transform(audio)
        return audio
        
if __name__ == '__main__':
    # dataset = VoicesDataset(src='/home/nevronas/dataset/accent-new', load=True)
    dataset = VCCDataset(load=False)
