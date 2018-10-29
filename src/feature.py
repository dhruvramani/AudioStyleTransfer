import librosa
import os
import numpy as np
import glob
'''
parser = argparse.ArgumentParser()

parser.add_argument('--out_folder', help='path to spectrogram folder.')
parser.add_argument('--in_folder', help='Path to raw audio folder (wav, mp3, etc.).')
parser.add_argument('--offset', type=float, default=0,
                    help='Start from (sec).')
parser.add_argument('--duration', type=float, default=10,
                    help='Desired duration (sec).')
parser.add_argument('--sr', type=int, default=44100,
                    help='Sample rate to use.')
parser.add_argument('--n_fft', type=int, default=2048,
                    help='FFT Window length.')

args = parser.parse_args()
'''

N_FFT = 2048

def transform_stft(signal):
    D = librosa.stft(signal, n_fft=N_FFT)
    S, phase = librosa.magphase(D)
    S = np.log1p(S)
    return S

def read_audio_spectrum(filename):
    signal, fs = librosa.load(filename)
    S = librosa.stft(signal, N_FFT)
    final = np.log1p(np.abs(S[:,:430]))  
    return final, fs

def power_spectral(signal):
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
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