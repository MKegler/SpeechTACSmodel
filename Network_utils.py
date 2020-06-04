'''
Module contaning functions used to pre-process and obtain input to the model of cortical encoding of speech.
Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

from scipy import signal
import numpy as np
import Network
import time
from PyNSL import pycochlea as cochlea


def load_TIMIT(path, path_noise=None, mode='clean', SNR_dB=np.inf, fs=8000.):
    '''
    Function for loading TIMIT sentences, and associatiated phonetic
    transcription and preprocessing them for simulating in the network.

    Inputs:
        path: path to the TIMIT setence (without .wav)
        path_noise: path to the file containing noise (without .wav, default: None -> no noise)
        fs: sampling rate used in the cochlear processing step (default: 8kHz)
        SNR_dB: SNR of the input SPiN, in dB (default: np.inf -> clean speech only)
        mode: Mixing mode: clean | babble | deaf (default: clean -> no noise)
    Outputs:
        channels: auditory channels representing inputs to the model in pA
        syl_bounds: syllable onset times (last is the end of the sentece)
        syl_lables: syllable lables (one element shorter than bounds)
        full_sent: full sentence transcription
    '''

    # Load TIMIT sentence and process it through the auditory periphery model
    if (mode == 'clean') or (path_noise is None):
        channels = cochlea.process(path + '.WAV', None, mode, SNR_dB, fs)
    else:
        channels = cochlea.process(path + '.WAV', path_noise + '.wav', mode, SNR_dB, fs)

    # Load syllable boundaries and labels
    syl_file = np.load(path + '_SYL.npy', allow_pickle=True, encoding='latin1').item()
    syl_bounds = syl_file['syl_bounds'].astype(np.float)/(16000./fs)
    syl_labels = syl_file['syl_labels']
    full_sent = ' '.join(open(path + '.TXT').readlines()[0].strip().split()[2:]) # Load full sentence

    return channels, syl_bounds, syl_labels, full_sent


def smooth(x,w=5):
    '''
    Simple smoothing function
    Inputs:
        x: NumPy 1-D array
        w: smoothing window size, in samples (default: 5 samples)
    Outputs:
        out: smoothed vector
    '''
    if w%2 == 0:
        w = w - 1
    valid = np.convolve(x,np.ones(w,dtype=int),'valid')/w
    r = np.arange(1,w-1,2)
    start = np.cumsum(x[:w-1])[::2]/r
    finish = (np.cumsum(x[:-w:-1])[::2]/r)[::-1]
    out = np.concatenate((start, valid, finish))
    return out


def get_network_input(channels, S, T, fs=8000, fs_out=100000, fds=100, Tsil=[0.5,0.2]):
    '''
    Obtain the input to the all excictatory neurons in the network given
    spectral (S) and temporal (T) part of the kernel. Implementation based on (Hyafil, et al., 2015)

    Inputs:
        channels: auditory channels from the model of subcortical processing (128 x N)
        S: spectral component of the STRF filter (1 x 32)
        T: temporal component of the STRF filter (1 x 6)
        fs: sampling rate of the auditory channels input X (default: 8kHz).
        fs_out: output sampling rate in Hz (i.e. simulation sampling freq, default: 100kHz)
        fds: sampling rate of the STRF kernel in Hz (default: 100 Hz)
        Tsil: added silence [before onset, after the end] in s
    Outputs:
        Iext: auditory input to the model (84 x N matrix)
        dt: simulation timestep, in ms
    '''
    # Silence from seconds -> samples
    Tsil = (np.array(Tsil)*fs).astype(np.int)
    nchan = S.shape[1]

    # Auditory channels to be used. Here, every 4th to obtain 32 channels projected to Ge neurons.
    ch_idx = (np.arange(1,nchan+1)*np.ceil(channels.shape[1]/nchan)-1).astype(np.int)

    # Obtain input to Ge neurons
    GE_input = channels[:,ch_idx].T

    # Obtain input to Te neurons
    TE_input = channels[:,ch_idx].T
    TE_input = np.dot(S, TE_input) # Process input through spectral weights
    # Downsample the resulting signal to match the temporal resolution of STRF (here 100 Hz)
    TE_input = signal.resample_poly(smooth(TE_input[0,:], np.int(fs/fds)), fds, fs)
    # Filter the resulting signal using temporal portion of the STRF kernel
    TE_input = signal.lfilter(T[0,:], np.array([1,0,0,0,0,0]), TE_input)
    # Upsample to the original sampling rate (padding)
    TE_input = np.concatenate([np.ones(np.int(fs/fds))*i for i in TE_input])
    TE_input = np.tile(TE_input, (10,1)) # Tile the resulting signal x10, input to Te cells

    # Control step
    # If one input is longer than the other (due to resampling rounding)
    # crop both to the length of the shorter one.
    TE_input = TE_input[:,:min([TE_input.shape[1], GE_input.shape[1]])]
    GE_input = GE_input[:,:min([TE_input.shape[1], GE_input.shape[1]])]

    # Pre-allocate matrix of network inputs in the simulations
    Iext = np.zeros((84, sum(Tsil)+TE_input.shape[1]))
    # Popualte with the obtained inputs to Te and Ge neurons (incl. silence before and after)
    Iext[:10, Tsil[0]:(-1*Tsil[1])] = TE_input[:,:]
    Iext[20:52, Tsil[0]:(-1*Tsil[1])] = GE_input[:,:]

    # Upsampling to the desired fs_out
    if fs_out > fs:
        Iext = signal.resample_poly(Iext, fs_out, fs, axis=1)
        dt = 1000./fs_out # in ms
    else:
        dt = 1000./fs # in ms

    return Iext, dt
