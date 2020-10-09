'''
Module contains hard-coded network of speech encoding through coupled oscillations.
Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)
'''

import numpy as np

def run(T, dt, Iext=None, connectivity=None, seed=None):
    '''
    Run simulation of the model.

    Inputs:
        T - simulation time (ms)
        dt - timestep (ms)
        Iext - external input to the network (84 x T/dt)
        connectivity - custom connectivity matrix, 84 x 84 array (optional)
        seed - fix seed of the RNG (optional)
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

    # Neuron counts and indexing
    nTe = 10
    nTi = 10
    nGe = 32
    nGi = 32
    N = nTe + nTi + nGe + nGi # total number of neurons
    n_count = [nTe, nTi, nGe, nGi]
    all_Te = np.arange(0, n_count[0])
    all_Ti = np.arange(n_count[0], sum(n_count[:2]))
    all_Ge = np.arange(sum(n_count[:2]), sum(n_count[:3]))
    all_Gi = np.arange(sum(n_count[:3]), sum(n_count[:]))
    all_exc = np.concatenate((all_Te, all_Ge))
    all_inh = np.concatenate((all_Ti, all_Gi))

    # General neuron parameters
    C = 1. # Membrance capacitance (pF)
    VL = -67. # Leak potential (mV)
    Vthr = -40. # spiking threshold potential (mV)
    Vres= -87. # reset potential (mV)

    # Connectivity matrix rows -> from, columns -> to; (i.e. g[i,j], i-to-j synapse)
    g = np.zeros((N,N))
    
    # Connectivity (To <- From, ex. GeGi - from: Gi, to: Ge) (nS)
    if connectivity is None:    
        # Theta connectivity
        gTeTe = 0./nTe
        gTeTi = 2.07/nTi
        gTiTe = 6.66/nTe
        gTiTi = 4.32/nTi
        # Gamma connectivity
        gGeGe = 0./nGe
        gGeGi = 5./nGi
        gGiGe = 10./nGe
        gGiGi = 0./nGi
        # X-freq. Coupling
        gGeTe = 1./nTe #

        # Filling in connectivity to_matrix column by column
        g[:,all_Te] = np.tile(to_matrix(n_count, [gTeTe, gTeTi , 0, 0]), (1,nTe))
        g[:,all_Ti] = np.tile(to_matrix(n_count, [gTiTe, gTiTi, 0, 0]), (1,nTi))
        g[:,all_Ge] = np.tile(to_matrix(n_count, [gGeTe, 0, gGeGe, gGeGi]), (1,nGe))
        g[:,all_Gi] = np.tile(to_matrix(n_count, [0, 0, gGiGe, gGiGi]), (1,nGi))
        
    else:
        print ('Using custom connectivity...')
        g = connectivity[:]

    # Leak conductance (nS)
    gL_Te = 0.0264 # leak conductance for Te neurons 
    gL_Ti = 0.1 # leak conductance for Ti neurons
    gL_Ge = 0.1 # leak conductance for Ge neurons
    gL_Gi = 0.1 # leak conductance for Gi neurons
    gL = to_matrix(n_count, [gL_Te, gL_Ti, gL_Ge, gL_Gi]).T

    # DC currents (pA)
    IdcTe = 1.25
    IdcTi = 0.08512
    IdcGe = 3.
    IdcGi = 1.
    Idc = to_matrix(n_count, [IdcTe, IdcTi, IdcGe, IdcGi]).T

    # TauR -> rise time constant (ms)
    tauR_Te = 4. 
    tauR_Ti = 5. 
    tauR_Ge = 0.2 
    tauR_Gi = 0.5
    tauR = to_matrix(n_count, [tauR_Te, tauR_Ti, tauR_Ge, tauR_Gi])

    # TauD -> decay time constant (ms)
    tauD_Te = 24.3150
    tauD_Ti = 30.3575
    tauD_Ge = 2.
    tauD_Gi = 20.
    tauD = to_matrix(n_count, [tauD_Te, tauD_Ti, tauD_Ge, tauD_Gi])

    # noise (pA x sqrt(ms))
    sigma_Te = 0.2817
    sigma_Ti = 2.0284
    sigma_Ge = 2.0284
    sigma_Gi = 2.0284
    sigma = to_matrix(n_count, [sigma_Te, sigma_Ti, sigma_Ge, sigma_Gi]).T

    # Reversal potentials (mV)
    VI = -80. # Equilibrium potential for inhibitory neurons
    VE = 0. # Equilibrium potential for excitatory neurons
    Vsyn = to_matrix(n_count, [VE, VI, VE, VI])
    Vsyn = np.tile(Vsyn,(1,N))
        
    # No external currents
    if Iext is None:
        Iext = np.zeros((TT,N))

    # Pre-allocate matrices
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
