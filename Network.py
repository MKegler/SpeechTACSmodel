'''
Module contains hard-coded network of speech encoding through coupled oscillations.
Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

import numpy as np

def run(T, dt, Iext=None, params=None, connectivity=None, seed=None):
    '''
    Run simulation of the model.

    Inputs:
        T - simulation time, in miliseconds
        dt - timestep, in miliseconds
        Iext - external input to the network (84 x T/dt)
        connectivity - custom connectivity matrix, 84 x 84 array (optional)
        seed - fix seed of the RNG
    Outputs:
        Vt - membrane potentials of neurons in the network over the simulation time course.
        spikes - binary coded spiking in the network over the simulation time course.
        LFP - local field potential over the simulation time course
    '''

    if seed != None:
        np.random.seed(seed)

    def to_matrix(n, vals):
        '''
        Helper functions that turn parameter values into matrix form.
        Use for pre-processing network parameters.
        '''
        ar = np.array([n, vals]).T
        matrix = np.vstack([np.ones((int(s[0]),1))*s[1] for s in ar]) # concantate into row matrix
        return matrix

    TT = np.int(np.ceil(T/dt))

    if params == None:
        # Neuron counts and indexing
        nTE = 10
        nTI = 10
        nGE = 32
        nGI = 32
        N = nTE + nTI + nGE + nGI # total number of neurons
        n_count = [nTE, nTI, nGE, nGI]
        all_TE = np.arange(0, n_count[0])
        all_TI = np.arange(n_count[0], sum(n_count[:2]))
        all_GE = np.arange(sum(n_count[:2]), sum(n_count[:3]))
        all_GI = np.arange(sum(n_count[:3]), sum(n_count[:]))
        all_exc = np.concatenate((all_TE, all_GE))
        all_inh = np.concatenate((all_TI, all_GI))

        # General neuron parameters
        C = 1. # Conductance (pF)
        VL = -67. # Leak potential (mV)
        Vthr = -40. # spiking threshold potential (mV)
        Vres= -87. # reset potential (mV)

        if connectivity is None:
            # connectivity setup (all in nS)
            # Connectivity matrix preallallocation
            g = np.zeros((N,N))
            # Theta connectivity
            gTee = 0./nTE   # E-E conductance [nS]
            gTii = 4.32/nTI # I-I conductance [nS]
            gTie = 2.07/nTE # I-E conductance [nS]
            gTei = 3.33/nTI # E-I conductance [nS]
            # Gamma connectivity
            gGee = 0./nGE   # E-E conductance [nS]
            gGei = 10./nGI  # E-I conductance [nS]
            gGie = 5./nGE   # I-E conductance [nS]
            gGii = 0./nGI   # I-I conductance [nS]
            # Coupling
            gGeTe = 1./nTE # Te-Ge conductance [nS]
            # Filling in connectivity matrix using to_matrix populating it on a column by column
            g[:,all_TE] = np.tile(to_matrix(n_count, [gTee, gTie , 0, 0]), (1,nTE))
            g[:,all_TI] = np.tile(to_matrix(n_count, [gTei, gTii, 0, 0]), (1,nTI))
            g[:,all_GE] = np.tile(to_matrix(n_count, [gGeTe, 0, gGee, gGie]), (1,nGE))
            g[:,all_GI] = np.tile(to_matrix(n_count, [0, 0, gGei, gGii]), (1,nGI))
            g[all_TI, all_TI] = 0 # Remove self-connection in Ti population
        else:
            print ('Using custom connectivity...')
            g = connectivity[:]

        # Leak conductance (all in nS)
        gL_TE = 0.0264 # leak conductance for Te neurons 
        gL_TI = 0.1 # leak conductance for Ti neurons
        gL_GE = 0.1 # leak conductance for Ge neurons
        gL_GI = 0.1 # leak conductance for Gi neurons
        gL = to_matrix(n_count, [gL_TE, gL_TI, gL_GE, gL_GI]).T

        # DC currents (all in pA)
        IdcTE = 1.25
        IdcTI = 0.08512
        IdcGE = 3.
        IdcGI = 1.
        Idc = to_matrix(n_count, [IdcTE, IdcTI, IdcGE, IdcGI]).T

        # TauR -> rise time constant (all in ms)
        tauR_TE = 4. # rising time constant of Te neurons (ms)
        tauR_TI = 5. # rising time constant of Ti neurons (ms)
        tauR_GE = 0.2 # rising time constant of Ge neurons (ms)
        tauR_GI = 0.5 # rising time constant of Gi neurons (ms)
        tauR = to_matrix(n_count, [tauR_TE, tauR_TI, tauR_GE, tauR_GI])

        # TauD -> decay time constant (all in ms)
        tauD_TE = 24.3150 # decay time constant of Te neurons (ms)
        tauD_TI = 30.3575 # decay time constant of Ti neurons (ms)
        tauD_GE = 2. # decay time constant of Ge neurons (ms)
        tauD_GI = 20. # decay time constant of Gi neurons (ms)
        tauD = to_matrix(n_count, [tauD_TE, tauD_TI, tauD_GE, tauD_GI])

        # noise (all in pA x sqrt(ms))
        sigma_TE = 0.2817 # noise parameter for Te neurons
        sigma_TI = 2.0284 # noise parameter for Ti neurons
        sigma_GE = 2.0284 # noise parameter for Ge neurons
        sigma_GI = 2.0284 # noise parameter for Ge neurons
        sigma = to_matrix(n_count, [sigma_TE, sigma_TI, sigma_GE, sigma_GI]).T

        # Reversal potentials (all in mV)
        VI = -90. # Equilibirum potential for inhibitory neurons
        VE = 0. # Equilibrium potentinal for excitatory neurons
        Vsyn = to_matrix(n_count, [VE, VI, VE, VI])
        Vsyn = np.tile(Vsyn,(1,N))

    else:
        # Handle some other parameters, otherwise use the default (above)
        pass

    # No external currents
    if Iext is None:
        Iext = np.zeros((TT,N))

    # Pre-allocate memory
    ss = np.zeros((N,1))
    sr = np.zeros((N,1))
    Vt = np.zeros((TT,N))
    LFPt = np.zeros(TT)
    spikes = np.zeros((TT,N))
    LFPgamma = np.zeros(TT)
    LFPtheta = np.zeros(TT)

    # Initial membrane values -> random
    V = VL + (np.random.rand(1,N) - .5)*(Vthr - VL);
    V[0,all_exc] = -80+40*np.random.rand(1, all_exc.shape[0])

    # Running simulation (euler method)
    for t in range(TT):
        # Leak current
        IL = gL*(VL-V)
        # Noise current
        Inoise = sigma*np.random.randn(1,N)/np.sqrt(dt)
        # Synaptic current
        sr = sr*(1-(dt/tauR))
        ss = ss+(dt*(sr-ss)/tauD)
        Iss = g*np.tile(ss,(1,N))*(Vsyn-np.tile(V,(N,1)));
        Isyn = np.sum(Iss, axis=0)
        # Update membrane potential
        V += (dt*(IL + Idc + Iext[t,:] + Isyn + Inoise)/C)
        # Update LFP (-1 -> convention)
        LFP = -1*np.sum(abs(Iss[:,all_exc]))
        # Check for spiking
        above_thr = np.where(V>=Vthr)[1]
        # Reset spiking neurons
        V[0,above_thr] = Vres
        # Inrease SR values upon spike (delta event)
        sr[above_thr,0] = sr[above_thr,0] + 1
        # Fill in matrices
        spikes[t, above_thr] = 1
        Vt[t,:] = V[:]
        LFPt[t] = LFP

    return Vt, spikes, LFPt
