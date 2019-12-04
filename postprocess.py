from astroML.datasets import sdss_corrected_spectra
import numpy as np
import matplotlib.pyplot as plt

# load spectra as produced by astroML
data = np.load('/epyc/users/sportill/specAE/spec64k.npz')
rawspec = sdss_corrected_spectra.reconstruct_spectra(data)
lam = sdss_corrected_spectra.compute_wavelengths(data)

lowspecerr = (data['spec_err'] < 0.25) # some errors ~ 0 or are negative?
spec_mask = data['mask']

# normalized spectra (L1 norm)
spec = rawspec / data['norms'][:,None]
spec_err = data['spec_err'] / data['norms'][:,None]
spec_weig = 1./(spec_err*spec_err + 1./(2e6)) # cap weights
# masked pixels and pixels with too small errors are given zero weight
spec_weig[spec_mask] = 0
spec_weig[lowspecerr] = 0
spec_err[spec_mask] = float('inf')
spec_err[lowspecerr] = float('inf')

specclass = data['lineindex_cln']
# map lineindex_cln 2,3 to 0 and 4,5,6 to 1,2,3
specclass -= 3
specclass[specclass == -1] = 0 # set line_index 2 -> 0
specclass[specclass < -1] = -1
specclass[specclass > 3] = -1

ds = np.load('64k_20190612/datasplit.npz')
valididx = ds['valididx']
trainidx = ds['trainidx']

np.savez('spec64k_normed.npz', spec=spec, spec_weig=spec_weig, spec_err=spec_err, spec_class=specclass, trainidx=trainidx, valididx=valididx, lam=lam)
