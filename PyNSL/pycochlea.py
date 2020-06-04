'''
Module including different flavours of stimulus mixing and peripheral auditory processing.
Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

import numpy as np
from . import PyNSL as NSL
from scipy import signal
from scipy import stats
import soundfile as sf


def set_noise_lvl(x1, x2, SNR):
    '''
    Mix two signal at the desired SNR (computed through RMS).

    Inputs:
        x1, x2: Two signals (1D vectors). x1 - signal, x2 - noise
        SNR: desired SNR, in dB
    Outputs:
        x1, x2: Signals scaled to obtain the desired SNR
    '''
    target_rms = np.sqrt(np.mean(x1**2))
    competing_rms = np.sqrt(np.mean(x2**2))

    desired_babble_rms = (10.**(-SNR/20.)) * target_rms
    scaling_factor = desired_babble_rms/competing_rms
    x2 = x2 * scaling_factor
    return x1[:], x2[:]


def get_spin(target, noise, SNR, mode='babble'):
    '''
    Load tracks of speech-in-noise and mix them at the desired SNR.

    Inputs:
        target/noise: path to .wav files of target and noise
        SNR: desired SNR, in dB
        mode: mixing mode, choose from:
            'clean' - no background noise
            'babble' - babble noise, regular mixing (default)
            'deaf' - do not include target speaker in the mix (i.e. noTarget condition)
    Output:
        x_mixed: output mixture
    '''
    # Load audio tracks
    (x_target, fs) = sf.read(target)  # Target
    (x_noise, fs_noi) = sf.read(noise)  # background noise

    # Babble noise obtained from S.Kadir
    if fs_noi == 44100:
        # Downsample to 16k
        x_noise = signal.resample_poly(x_noise, fs, fs_noi)
        # Remove 1s ramping from start and finish
        x_noise = x_noise[(fs-1):-fs]

    # If noise track is shorter than the sentence, double the noise
    if x_target.shape[0] > x_noise.shape[0]:
        x_noise = np.concatenate((x_noise, x_noise))

    # Adjust levels to appropriate SNR
    x_target, x_noise = set_noise_lvl(x_target, x_noise, SNR)

    x1 = x_target[:]
    x2 = x_noise[0:x_target.shape[0]] # If noise track is longer, crop it

    # Mix
    if mode == 'deaf':
        # When simulating only background noise without the target
        x_mixed = x2[:]
    else:
        x_mixed = x1 + x2

    return x_mixed


def process(path_signal, path_noise=None, mode='clean', SNR=np.inf, fs_out=8000., paras=[8, 8, -2, -1], COCHBA=None):
    '''
    Main function that loads audio, mixes it and processes through the model of auditory periphery (PyNSL).

    Inputs:
        path_signal: path to .wav file containing target speech
        path_noise: path to .wav file containing background noise (if None, mode='clean')
        mode: mixing mode, choose from:
            'clean' - no background noise
            'babble' - babble noise, regular mixing (default)
            'deaf' - do not include target speaker in the mix (i.e. noTarget condition)
        SNR: desired SNR of the mix, in dB
        fs_out: sampling rate of the obtained auditory channels
        paras: parameters for the NSL early auditory procesing model (obtained from Hyafil et al., 2015)
        COCHBA: pre-loaded cochlear filterbank (if None, will be loaded)
    Outputs:
        channels_up: upsampled output of the model of early auditory processing
    '''
    # Do the mixing
    if (mode == 'clean') or (SNR == np.inf):
        (x, _) = sf.read(path_signal)
    elif mode in ('babble', 'deaf'):
        x = get_spin(path_signal, path_noise, SNR, mode)
    else:
        print('Choose valid mode: clean | babble | deaf')
        return None

    # Downsample to 8k (frequencies up to 4 kHz)
    # and standardize audio track (as in Hyafil, et al., 2015)
    # Model of auditory periphery doesn't have any limiting mechanisms,
    # so the latter needs to be done manually.
    x = signal.resample_poly(x, 1, 2)
    x = stats.zscore(x, ddof=1)

    # Rescale the normalized input to obtain the input at ~76 dB SPL
    x = x/8
    p = np.sqrt(np.mean((x)**2))
    p0 = 2*(10**(-5))
    print ('{} dB SPL'.format(20*np.log10(p/p0)))

    # Process
    channels = NSL.wav2aud(x, paras, COCHBA)

    # Upsample the obtained frames to the original sampling rate (default 8kHz sampling rate)
    l = channels.shape[0]
    dt_idx = 1000/fs_out/paras[0]
    idx_up = np.ceil(np.arange(dt_idx, l+0.1*dt_idx, dt_idx)).astype(np.int) - 1
    channels_up = channels[idx_up, :]

    return channels_up
