'''
Methods for analysing and extracting features from the simulations of speech encoding in the modelself.
Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

import numpy as np
import scipy.signal as signal


def burst_detector(spikes, dt, twin=20, std=3):
    '''
    Simple sliding-window detector of Theta burst, which predict syllable onsets.

    Inputs:
        spikes: array of spikes [T x N]
        dt: timestep in ms
        twin: time window to detect bursts in ms
    Ouput:
        detected: vector of detected syllable onsets
    '''
    # Sum of spikes across all neurons
    cumspk = np.sum(spikes, axis=1)

    # Compute sum of TH-i spikes in 20 ms windows. Discard windows with less then 2 spikes.
    square = np.ones(np.int(twin/(dt)))
    square_conv = np.convolve(cumspk, square, mode="same")
    square_conv[square_conv<2] = 0

    # Convolve the sum of spikes with a gaussian to find peak firing rates indicating syllable bounds.
    gx = np.arange(-twin/2, twin/2, dt)
    gaussian = np.exp(-(gx/std)**2/2)
    gauss_conv = np.convolve(square_conv, gaussian, mode="same")/np.sum(gaussian)

    detected = signal.find_peaks(gauss_conv, distance=twin/dt)[0]

    return detected


def get_chunks(detected, syl, syl_labels, dt, offset=20.):
    '''
    Given detected and actual syllables, the function returns boundaries and
    labels of syllables encoded by the network.

    Input:
        detected: detected theta spike bursts
        syl: actual syllable boundaries (last indicated the end of the sentence, not syllable onset)
        syl_labels: syllable labels
        dt: time step (in ms)
        offset: what is the offset before (-) and after (+) the theta burst to consider (in ms)
        Since the model was trained to predict the syllable onsets 20 ms later than the actual,
        taking a 20 ms wider window compensates for it.
    Output:
        th_chunks: array of onset and offset of chunks (including offset padding)
        th_chunks_labels: label of each chunk assigned according to when the syllable onsets was detected
    '''
    # Use only syllables predicted during the sentence presentation, not before of after
    within_sentence = detected[(detected >= syl[0])*(detected < syl[-1])]
    within_sentence = np.append(within_sentence, syl[-1])
    # Pre-allocated matrices
    th_chunks = np.zeros((len(within_sentence)-1, 2))
    th_chunks_labels = []
    # Determine the beginning and the end of each chunk of neural data and assing its label.
    # Labels are assined from the actual syllables during which the onset was detected.
    for i in range(th_chunks.shape[0]):
        th_chunks[i,0] = within_sentence[i] - offset/dt
        th_chunks[i,1] = within_sentence[i+1] + offset/dt
        diff = within_sentence[i] - syl
        th_chunks_labels.append(syl_labels[np.argmin(diff[diff>=0])])
    return th_chunks, th_chunks_labels


def spkd(s1, s2, cost):
    '''
    Fast implementation of victor-purpura spike distance (faster than neo & elephant python packages)
    Direct Python port of http://www-users.med.cornell.edu/~jdvicto/pubalgor.html
    The below code was tested against the original implementation and yielded exact results.
    All credits go to the authors of the original code.

    Input:
        s1,2: pair of vectors of spike times
        cost: cost parameter for computing Victor-Purpura spike distance.
        (Note: the above need to have the same units!)
    Output:
        d: VP spike distance.
    '''
    nspi=len(s1);
    nspj=len(s2);

    scr=np.zeros((nspi+1, nspj+1));

    scr[:,0]=np.arange(nspi+1)
    scr[0,:]=np.arange(nspj+1)

    for i in np.arange(1,nspi+1):
        for j in np.arange(1,nspj+1):
            scr[i,j]=min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*np.abs(s1[i-1]-s2[j-1])]);

    d=scr[nspi,nspj];

    return d


def get_selfdist(st_inp, n_chan=32, cost=60., dt = 0.01):
    '''
    Method for computing pair-wise spike distances from a range of spike trains.

    Inputs:
        st_inp: [2 x N] array with spike times and indices of neurons.
        N - number of spikes generated, 1st row - index of neuron generating given spikes, 2nd row - spike time.
        n_chan - number of neurons (default: 32)
        cost - cost parameter for VP spike distance, in ms (default: 60 ms)
        dt - simulation timestep, in ms (default: 0.01 ms -> 100 kHz)
    Output:
        pc - [n_chan x n_chan] matrix containing pairwise VP spikes distances for each pair of neurons.
    '''
    sts = [st_inp[0,st_inp[1,:]==i] for i in range(n_chan)]
    pc = np.zeros((n_chan, n_chan))
    for i in range(n_chan):
        for j in range(n_chan):
            pc[i,j] = spkd(sts[i], sts[j], dt/(cost))
    return pc
