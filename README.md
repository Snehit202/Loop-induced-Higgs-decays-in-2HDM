# Loop-induced-Higgs-decays-in-2HDM
A repository of python codes that plot the Branching ratio 125 GeV higgs decay to 2 photons, and a Z boson and a photon, along with the excluded and best-fit regions of the 2HDM parameter space calculated from HiggsBounds and HiggsSignals. 

The code file THDMconstraint.py plots B(h $\rightarrow$ $\gamma$ $\gamma$) and B(h $\rightarrow$ Z $\gamma$) as a function of 2HDM parameter, $\alpha$. The plot is interactive whereby other parameters, like $\tan \beta$ and charged higgs mass($m_{H^{\pm}}$), can be changed with a slider. Alongside, the plot also has regions of excluded and best-fit parameter space.

The best-fit and excluded regions of the 2HDM parameter space is calculated in the code file 2HDM_create_data.py. The code uses HiggsTools repos that contains the latest versions of HiggsBounds and HiggsSignals and can be run with a built-in python interface. The details of it can be seen in the HiggsTools repository.

Required packages-
1. HiggsTools with HiggsBounds and HiggsSignals database. Instructions for using the python interface is in the documentation of HiggsTools repository.
2. rundec: a python wrapper built on CRunDec for calculating running quark mass and strong coupling constant.
3. Standard python packages: numpy, matplotlib, pandas

Instructions on running the code:
1. Example files are already available in the datafiles folder. One can simply run the file THDMconstraint.py(requires additional rundec package: type "pip install rundec" to install) to get the final interactive plots.
2. To generate your own datafiles with different mass values of H and A, first, clone the latest version of HiggsBounds and HiggsSignals from HiggsTools and install the python interface. The instructions are given at-
3. Once the interface is installed, download the HiggsBound and HiggsSignal database from the same git. One has to edit the directory paths in 2HDM_create_data.py's code to access the database and save the generated files at desired destination.
4. One can then run 2HDM_create_data.py to generate exclusion limits and $\chi^2$ values of the parameter space for different mass values of H and A. After that run THDMconstraint.py to plot the desired results.
