## Modelling the effects of transcranial alternating current stimulation on the neural encoding of speech in noise

### Implementation of the model introduced in Kegler M. & Reichenbach T. (under review) *Modelling the effects of transcranial alternating current stimulation on the neural encoding of speech in noise*

#### Demo.ipynb is an annotated jupyter notebook including sample simulations of the model

The core of the neural network encoding natural speech is mainly based on [Hyafil et al., 2015](https://elifesciences.org/articles/06213.pdf). This notebook is a demo of the network processing a speech utterance and encoding its content through spiking activity. In particular, the slower theta activity parses the utterance into chunks and faster gamma activity captures acoustic content of each chunk. Custom written code is divided into 5 modules:
- **PyNSL** - Direct Python port of parts of NSL toolbox [Chi et al., 2005](https://asa.scitation.org/doi/full/10.1121/1.1945807). The original Matlab implementation is available [here](http://nsl.isr.umd.edu/downloads.html).
- **Network** - Core implementation of the model used in simulations. Includes all the equations and parameters with hard-coded parameter values.
- **Network_utils** - Module containing functions used for preprocessing auditory inputs to the model.
- **Stimulation_utils** - Module containing functions used to extract and preprocess envelope-shaped stimulation waveforms derived from the speech stimulus.
- **Analysis_utils** - Module containing functions used for analysis of model simulations to the extraction features representing syllable encoding in the model simulation.

Each module contains Python functions called in this demo. Large-scale simulations described in the paper were performed using the Imperial College high-performance computing cluster. This demo is illustrating a single simulation of the model and extraction of features later used in the analysis employing different conditions. For the sake of this demo, sample audio tracks of a randomly selected TIMIT sentence & pre-mixed babble noise are available in the SampleAudio directory. Full TIMIT speech corpus used in the paper is available [here](https://catalog.ldc.upenn.edu/LDC93S1).

#### Demo outline
- Loading & preprocessing auditory inputs to the model
- Envelope-shaped stimulation waveform extraction
- Simulation & visualization
- Extraction of features encoded in the simulation

#### Required 3rd party packages
- [SciPy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)

For convenience *tACSmodel_env.yml* can be used to set up environment as described [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). The environment and the code were tested on a Linux (Ubuntu 16.04 LTS) and MAC (macOS Catalina 10.15.5) machines.

Author: Mikolaj Kegler (mikolaj.kegler16@imperial.ac.uk)

In case of any issues, questions or suggestions, please do open an issue in the repository and/or email me directly.

Last updated: 31th July 2020
