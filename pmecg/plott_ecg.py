import pandas as pd
import pmecg
import h5py



#frequency used
fs = 400

f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
record = f["tracings"][0]
ecgprep_leads = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
rename = {'DI':'I','DII':'II','DIII':'III','AVR':'aVR','AVL':'aVL','AVF':'aVF'}
df = pd.DataFrame(record, columns=[rename.get(l, l) for l in ecgprep_leads])


plotter = pmecg.ECGPlotter()
configuration = pmecg.template_factory("1x3", df, leads_map=None)
fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs)
fig.savefig("ecg_preprocessed_sample.png", dpi=300, bbox_inches="tight")

