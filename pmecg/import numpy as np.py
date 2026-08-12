import numpy as np
import pandas as pd
import pmecg

fs = 500
t = np.linspace(0, 10, int(fs * 10))
df = pd.DataFrame({name: np.random.randn(len(t)) * 0.1 for name in pmecg.SUPPORTED_LEADS})

plotter = pmecg.ECGPlotter()
configuration = pmecg.template_factory("4x3", df, leads_map=None)
fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs)
fig.savefig("ecg.png", dpi=300, bbox_inches="tight")

