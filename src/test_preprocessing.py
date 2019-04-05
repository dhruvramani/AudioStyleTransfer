import torch
import librosa
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from vctk import VCTK
from torch.utils.data import DataLoader
#import torchfile

from models import *

N_FFT = 512*4
n_iter=2000



def transform_stft(signal, pad=True):
    D = librosa.stft(signal, n_fft=N_FFT)
    S, phase = librosa.magphase(D)
    S = np.log1p(S)
    if(pad):
        S = librosa.util.pad_center(S, 1700)
    return S, phase

def reconstruction(S, phase):
    exp = np.expm1(S)
    comple = exp * np.exp(phase)
    istft = librosa.istft(comple)
    return istft 

def load_audio(audio_path):
    signal, fs = librosa.load(audio_path)
    return signal, fs

def save_file(audio, phase, fs, filename, path = '../save/plots/preprocess/', save_audio = False):
    matplotlib.pyplot.imsave(path+filename+'.png', audio[:, 5000:10000])
    print("==> Saved Spectogram")

    if(save_audio):
        audio_res = reconstruction(audio, phase)
        print(audio_res.shape)
        librosa.output.write_wav(path+"audio/"+filename+".wav", audio_res, fs)
        print("==> Saved Audio")

def inp_transform(inp):
    inp = inp.astype(np.float32)
    inp = inp.flatten()
    inp, phase = transform_stft(inp, pad=False)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp, phase

def test_preprocessing(audio_file, dir = "/home/nevronas/dataset/vctk/raw/"):
    signal, fs = load_audio(dir+audio_file+".wav")
    #signal=librosa.core.resample(signal,fs,44100)
    print("Signal Size : ", signal.shape)

    signal, phase = inp_transform(signal)
    print("Processed Size : ", signal.shape)

    signal = signal[0].numpy()
    signal = signal[0]

    save_file(signal, phase, fs, audio_file, save_audio = True)

def phase_restore(mag, random_phases, N=50):
    p = np.exp(1j * (random_phases))

    for i in range(N):
        _, p = librosa.magphase(librosa.stft(
            librosa.istft(mag * p), n_fft=args.n_fft))
        update_progress(float(i) / N)
    return p
    random_phase = S.copy()




if __name__ == '__main__':

    test_preprocessing("vocals", "/home/nevronas/dataset/dualaudio/DSD100/Sources/Dev/076 - Little Chicago's Finest - My Own/")

    test_preprocessing("other", "/home/nevronas/dataset/dualaudio/DSD100/Sources/Dev/076 - Little Chicago's Finest - My Own/")
    
    test_preprocessing("drums", "/home/nevronas/dataset/dualaudio/DSD100/Sources/Dev/076 - Little Chicago's Finest - My Own/")
    
    # test_preprocessing("p351_423")


spectr = torchfile.load("/home/nevronas/dataset/dualaudio/DSD100/Sources/Dev/076 - Little Chicago's Finest - My Own/")
S = np.zeros([N_FFT / 2 + 1, spectr.shape[1]])
np.random.shuffle(random_phase)
p = phase_restore((np.exp(S) - 1), random_phase, N=n_iter)

# ISTFT
y = librosa.istft((np.exp(S) - 1) * p)

librosa.output.write_wav('../save/plots/preprocess/kuch_bhi.wav', y, args.sr, norm=False) 
