import librosa
import os
import numpy as np
import glob

N_FFT = 256

def transform_stft(signal):
    D = librosa.stft(signal, n_fft=N_FFT, win_length=150, hop_length=50)
    S, phase = librosa.magphase(D)
    S = np.log1p(S)
    return S

def read_audio_spectrum(filename):
    signal, fs = librosa.load(filename)
    S = librosa.stft(signal, N_FFT)
    final = np.log1p(np.abs(S[:,:430]))  
    return final, fs

def power_spectral(signal, pre_emphasis = 0.1):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    S = librosa.stft(emphasized_signal, N_FFT)
    power_spectral=np.square(S)/N_FFT
    final = np.log1p(np.abs(power_spectral[:,:430]))  
    return final

'''
vctk_set = torchaudio.datasets.VCTK('../data/', download=True)
filelist = glob.glob(os.path.join(args.in_folder, "*.txt"))
for f in filelist:
    name_out = f.split("/")[-1]
    power_spectral(f, args.out_folder + "/" + name_out)
'''
