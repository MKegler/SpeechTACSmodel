'''
Direct Python port of a few functions from NSL toolbox implementing
early stage auditory processing. All credits go to the original authors.
The below implementaion has been tested against the original 
Matlab code (http://nsl.isr.umd.edu/downloads.html) and yielded identical results.
Implmentaion: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

import numpy as np
import scipy.io as sio
from scipy import signal
import os
import pkg_resources

def sigmoid(y, fac):
    '''
    Original documentation below.
    '''
    #     SIGMOID nonlinear funcion for cochlear model
    #     y = sigmoid(y, fac);
    #     fac: non-linear factor
    #     -- fac > 0, transister-like function
    #     -- fac = 0, hard-limiter
    #     -- fac = -1, half-wave rectifier
    #     -- else, no operation, i.e., linear
    #
    # SIGMOID is a monotonic increasing function which simulates
    # hair cell nonlinearity.
    # See also: WAV2AUD, AUD2WAV
    #
    # % Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
    # % v1.00: 01-Jun-97

    if fac > 0:
        y = np.exp(-y/fac)
        y = 1./(1+y)
    elif fac == 0:
        y = (y > 0) #hard-limiter
    elif fac == -1:
        y = np.max(y, 0) # half-wave rectifier
    elif fac == -3:
        y = halfregu(y)

    return y

def halfregu(y):
    # Placeholder
    return y


def wav2aud(x, paras=[8,8,-2,-1], COCHBA=None):
    '''
    Original documentation below.
    '''
    # % WAV2AUD fast auditory spectrogramm (for band 180 - 7246 Hz)
    # x : the acoustic input.
    # %	v5	: the auditory spectrogram, N-by-(M-1)
    # %
    # %	COCHBA  = (global) [cochead; cochfil]; (IIR filter)
    # %       cochead : 1-by-M filter length (<= L) vector.
    # %               f  = real(cochead); filter order
    # %               CF = imag(cochead); characteristic frequency
    # %	cochfil : (Pmax+2)-by-M (L-by-M) [M]-channel filterbank matrix.
    # %		B = real(cochfil); MA (Moving Average) coefficients.
    # %		A = imag(cochfil); AR (AutoRegressive) coefficients.
    # %	M	: highest (frequency) channel
    # %
    # %	COCHBA  = [cochfil]; (IIR filter)
    # %	cochfil : (L-by-M) [M]-channel filterbank impulse responses.
    # %
    # %	PARAS	= [frmlen, tc, fac, shft];
    # %	frmlen	: frame length, typically, 8, 16 or 2^[natural #] ms.
    # %	tc	: time const., typically, 4, 16, or 64 ms, etc.
    # %		  if tc == 0, the leaky integration turns to short-term avg.
    # %	fac	: nonlinear factor (critical level ratio), typically, .1 for
    # %		  a unit sequence, e.g., X -- N(0, 1);
    # %		  The less the value, the more the compression.
    # %		  fac = 0,  y = (x > 0),   full compression, booleaner.
    # %		  fac = -1, y = max(x, 0), half-wave rectifier
    # %		  fac = -2, y = x,         linear function
    # %	shft	: shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
    # %		  etc. SF = 16K * 2^[shft].%
    # %
    # %	filt	: filter type, 'p'--> Powen's IIR filter (default)
    # %			       'p_o' --> Powen's old IIR filter (steeper group delay)
    # %
    # %	IIR filter : (24 channels/oct)
    # %	for the output of 	downsamp/shift	tc (64 ms)/ frame (16 ms)
    # %	==================================================================
    # %	180 - 7246		1	/0	1024	/ 256
    # %	90  - 3623		2	/-1	512	/ 128	*
    # %
    # %	Characteristic Frequency: CF = 440 * 2 .^ ((-31:97)/24);
    # %	Roughly, CF(60) = 1 (.5) kHz for 16 (8) kHz.
    # %
    # %	VERB	: verbose mode
    # %
    # %	WAV2AUD computes the auditory spectrogram for an acoustic waveform.
    # %	This function takes the advantage of IIR filter's fast performance
    # %	which not only reduces the computaion but also saves remarkable
    # %	memory space.
    # %	See also: AUD2WAV, UNITSEQ

    # % Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
    # % v1.00: 01-Jun-97

    # % Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
    # % v1.10: 04-Sep-98, add Kuansan's filter (as FIR filter)

    # % Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
    # % v2.00: 24-Jul-01, add hair cell membrane (lowpass) function

    # % Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
    # % v2.10: 04-Apr-04, remove FIR filtering option (see wav2aud_fir.m)

    # % get filter bank,
    # %	L: filter coefficient length;
    # %	M: no. of channels

    # Load cochlear filterbank if not pre-loaded
    if COCHBA is None:
        data_path = pkg_resources.resource_filename('PyNSL', 'aud24.mat')
        f = sio.loadmat(data_path)
        COCHBA = f['COCHBA']
        del f

    (L, M) = COCHBA.shape # p_max = L - 2
    L_x = len(x) # length of input

    # octave shift, nonlinear factor, frame length, leaky integration
    shft = paras[3] # octave shift (default -1, so 16kHz input == 8 kHz)
    fac = paras[2] # nonlinear factor (-2 == linear)
    L_frm = np.round(paras[0] * 2**(4+shft)).astype(int) # frame length (points), paras[0] 8 -> miliseconds

    if paras[1]:
        alph = np.exp(-1/(paras[1]*2**(4+shft))) # decaying factor
    else:
        alph = 0 # short-term avg.

    # hair cell time constant in ms
    haircell_tc = 0.5
    beta = np.exp(-1/(haircell_tc*2**(4+shft)))

    # get data, allocate memory for ouput
    N = np.ceil(L_x / L_frm).astype(int) # No. of frames
    x_tmp = np.zeros(N * L_frm)
    x_tmp[0:len(x)] = x[:]
    x = x_tmp[:]
    del x_tmp
    v5 = np.zeros((N, M-1))
    # CF = 440 * 2 .^ ((-31:97)/24) # Center frequencies

    # last channel (highest frequency)
    p = COCHBA[0, M-1].real
    idx = np.arange(0,p+1, dtype=int) + 1
    B = COCHBA[idx, M-1].real
    A = COCHBA[idx, M-1].imag
    y1 = signal.lfilter(B, A, x)
    y2 = sigmoid(y1, fac)

    # hair cell membrane (low-pass <= 4 kHz)
    # ignored for LINEAR ionic channels (fac == -2)
    if (fac != -2):
        y2 = signal.lfilter([1.], [1 -beta], y2)

    y2_h = y2[:]
    y3_h = 0

    for ch in (np.arange(M-1, 0, -1) - 1):
        p = COCHBA[0, ch].real
        idx = np.arange(0,p+1, dtype=int) + 1
        B = COCHBA[idx, ch].real
        A = COCHBA[idx, ch].imag
        y1 = signal.lfilter(B, A, x)

        # TRANSDUCTION: hair cells
        # Fluid cillia coupling (preemphasis) (ignored)
        # ionic channels (sigmoid function)
        y2 = sigmoid(y1, fac)[:]
        # hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
        if (fac != -2):
            y2 = signal.lfilter([1.], [1 -beta], y2)

        # lateral inhibitory network
        # masked by higher (frequency) spatial response
        y3 = y2[:] - y2_h[:]
        y2_h = y2[:]

        # half-wave rectifier ---> y4
        y4 = np.maximum(y3, np.zeros(len(y3)))

        # temporal integration window ---> y5
        if alph: # leaky integration
            y5 = signal.lfilter([1.], [1, -alph], y4)
            v5[:, ch] = y5[(L_frm*np.arange(1,N+1)) - 1]
        else: # % short-term average
            if (L_frm == 1):
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(y4.reshape(L_frm,N,order='F').copy())

    return v5
