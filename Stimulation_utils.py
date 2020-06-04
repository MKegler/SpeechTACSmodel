'''
Module contains functions used to obtain the envelope-shaped stimulation waveforms
Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import soundfile as sf
import scipy.stats as stats


def butter_lowpass(data, cutOff, fs, order=2):
    '''
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    '''
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(data, lowcut, highcut, fs, order=2):
    '''
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y


def scale_max(x):
    '''
    Remove mean and peak-normalize (for non-saturated waveforms)
    Input:
        x - envelope-shaped waveform
    Output:
        x_scaled - demeaned and peak-normalized waveform
    '''
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    x_scaled = x/peak
    return x_scaled


def get_envs(path_in, filt=[20], fs_out=100000, phases=[0.], saturate=False, forder=2):
    '''
    Obtain phase-shifted envelopes from TIMIT sentence.
    Inputs:
        path_in: path to TIMIT sentence (.wav)
        filt: corner frequencies of the applied filter (one -> lowpass, default: [20], two -> bandpass)
        fs_out: sampling rate of the output waveform (i.e. upsample to match simulation timestep, default: 100000)
        phases: phase shifts applied to the envelope (in degrees). Default: [0.]
        saturate: fix magnitude at 1? (waveform 'saturation' effect, default: False).
        forder: order of the applied Butterworth filter (default: 2).
    Output:
        envs_dict: dictionary containing phase-shifted envelopes. Keys -> phases, '(xxx)deg', values -> waveforms (1D vectors)
    '''

    # Obtain hilbert envelope
    (sound, fs) = sf.read(path_in)
    env = np.abs(signal.hilbert(sound))

    # Filter it
    if len(filt) == 1:
        env_filt = butter_lowpass(env, filt[0], fs, forder)
    elif len(filt) == 2:
        env_filt = butter_bandpass(env, filt[0], filt[1], fs, forder)
    else:
        env_filt = butter_bandpass(env, 1, 20, fs, forder)

    # Obtain analytic signal of the filtered envelope
    analytic_env = signal.hilbert(env_filt);

    # Saturate? (i.e. fix magnitude)
    if saturate == True:
        mag = 1.
    else:
        mag = np.abs(analytic_env)

    # Compute inst. phases
    phase = np.angle(analytic_env)

    # Shift the filtered envelops in phase
    envs = [mag*np.exp(1j*(phase - np.pi*(ph)/180.)).real for ph in phases]

    # If the signal was not saturated, remove mean and peak-normalize
    # Important for only lowpass filtered signals, so that tACS in not only positive
    if saturate == False:
        envs = [scale_max(e) for e in envs]

    # Upsample if fs_out greater than original fs
    if fs_out > fs:
        envs = [signal.resample_poly(e, fs_out, fs) for e in envs]

    # Build the ouput dictionary
    ph_list = ['{}deg'.format(int(ph)) for ph in phases]
    envs_dict = dict(zip(ph_list, envs))

    return envs_dict
