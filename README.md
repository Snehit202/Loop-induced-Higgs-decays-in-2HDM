# Loop-induced-Higgs-decays-in-2HDM
A repository of python codes that plot the Branching ratio 125 GeV higgs decay to 2 photons, and a Z boson and a photon, along with the excluded and best-fit regions of the 2HDM parameter space calculated from HiggsBounds and HiggsSignals. 

The code file THDMconstraint.py plots B(h-> $\gamma$ $\gamma$) and B(h->Z $\gamma$) as a function of 2HDM parameter, alpha. The plot is interactive whereby other parameters like beta and charged higgs mass can be changed with a slider. Alongside, the plot also has regions of excluded and best-fit parameter space.

The best-fit and excluded regions of the 2HDM parameter space is calculated in the code file 2HDM_create_data.py. The code uses HiggsTools repos that contains the latest versions of HiggsBounds and HiggsSignals and can be run with a built-in python interface. The details of it can be seen in the HiggsTools repository.

Required packages-
1. HiggsTools with HiggsBounds and HiggsSignals database. Instructions for using the python interface is in the documentation of HiggsTools repository.
2. rundec: a python wrapper built on CRunDec for calculating running quark mass and strong coupling constant.
