import pandas as pd
import pmecg
import h5py


#Frequency used
fs = 400

#Record is the first ecg in dataset being plotted. 
f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
record = f["tracings"][0]

#Check if needed
ecgprep_leads = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
rename = {'DI':'I','DII':'II','DIII':'III','AVR':'aVR','AVL':'aVL','AVF':'aVF'}
df = pd.DataFrame(record, columns=[rename.get(l, l) for l in ecgprep_leads])

#Plot 3 of the leads
# 2 x 6 
# 4 x 3 + 1
# With and without lead names



plotter = pmecg.ECGPlotter()
configuration = pmecg.template_factory("1x3", df, leads_map=None)

#Make a figure, standard config
fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs)
fig.savefig("ecg_preprocessed_sample.png", dpi=300, bbox_inches="tight")