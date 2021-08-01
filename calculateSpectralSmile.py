#// Calculate Spectral Smile present in the hyperspectral data

#// Import required libraries
import numpy as np
import scipy.io as sp
import pysptools.util as util

#// Read the hyperspectral data
[cube, header] = util.load_ENVI_file("EO1H0440342002212110PY_cropped.hdr")
cube = np.array(cube, dtype = np.float32)
[rw, cl, bands] = np.shape(cube)

#// Extract wavelength and FWHM values
wavelength = []
wavelength.append(header['wavelength'])
wavelength = np.array(wavelength, dtype = np.float32)

fwhm = []
fwhm.append(header['fwhm'])
fwhm = np.array(fwhm, dtype = np.float32)

#// Find O2 absorption band near 772nm in the hyperspectral data
arr = np.abs(wavelength - 772)
oxyIdx = np.where(arr == np.amin(arr))
oxyIdx = np.array(oxyIdx, dtype = np.int64)

#// Find CO2 absorption band near 2012nm in the hyperspectral data
arr = np.abs(wavelength - 2012)
carbonIdx = np.where(arr == np.amin(arr))
carbonIdx = np.array(carbonIdx, dtype = np.int64)

#// Calculate the first derivative of the right-hand side shoulders of O2 and
#// CO2 absorption bands
oxy = (cube[0:rw, 0:cl, oxyIdx[1,0]+1] - cube[0:rw, 0:cl, \
       oxyIdx[1,0]]) / ((fwhm[0, oxyIdx[1,0]+1] + fwhm[0, oxyIdx[1,0]])/2)

carbon = (cube[0:rw, 0:cl, carbonIdx[1,0]+1] - cube[0:rw, 0:cl, \
          carbonIdx[1,0]]) / ((fwhm[0, carbonIdx[1,0]+1] + \
fwhm[0, carbonIdx[1,0]])/2)

#// Calculate column mean of the derivative values
oxyDeriv = np.mean(oxy, axis = 0)
carbonDeriv = np.mean(carbon, axis = 0)

#// Calculate standard deviation of the normalized column mean of the
#// derivative values
oxyDeriv_rescaled = np.divide(oxyDeriv-np.min(oxyDeriv), \
                              np.max(oxyDeriv)-np.min(oxyDeriv))
carbonDeriv_rescaled = np.divide(carbonDeriv-np.min(carbonDeriv), \
                                 np.max(carbonDeriv)-np.min(carbonDeriv))
stdOxyDeriv = np.std(oxyDeriv_rescaled, ddof = 1)
stdCarbonDeriv = np.std(carbonDeriv_rescaled, ddof = 1)

#// Export results in .MAT file
sp.savemat('results.mat', {'stdOxyDeriv':stdOxyDeriv, \
                           'stdCarbonDeriv':stdCarbonDeriv, \
                           'oxyDeriv':oxyDeriv, 'carbonDeriv':carbonDeriv})


