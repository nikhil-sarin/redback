import warnings
import matplotlib.pyplot as plt
from redback.model_library import all_models_dict
from redback.constants import day_to_s
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
from scipy.optimize import curve_fit



model = 'tophat_redback'
function = all_models_dict[model]

z = 1.0
thv = 0.1
loge0 = 52
thc = 0.1
logn0 = 0
p = 2.5
logepse = -1.0
logepsb = -2.0
g0 = 100
xiN = 1.0

time = np.logspace(2, 8, 100)/day_to_s
fr_1ghz = 1e9 * np.ones(len(time))
kwargs = {}
kwargs["output_format"] = "flux_density"

lc1_1ghz = function(time, z, thv=thv, loge0=loge0, thc=thc, logn0=logn0, p=p,
                    logepse=logepse, logepsb=logepsb, g0=g0, xiN=xiN, frequency=fr_1ghz, **kwargs)

# Identify the peak
peak_index = np.argmax(lc1_1ghz)
post_peak_time = time[peak_index:]
post_peak_lc = lc1_1ghz[peak_index:]

# Define a function to fit the post-peak data
def trend_func(t, a, b):
    return a * t**b

# Fit the trend line to the post-peak data
params, _ = curve_fit(trend_func, post_peak_time, post_peak_lc)
print(params)


plt.loglog(time, lc1_1ghz, label='1 GHz')
plt.xlabel('Time [days]')
plt.ylabel(r'$L_{\rm bol}$')

plt.loglog(time, lc1_1ghz)
plt.show()
