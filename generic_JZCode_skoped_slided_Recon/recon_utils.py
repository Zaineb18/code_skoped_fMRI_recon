#from mri.operators.utils import normalize_frequency_locations
import numpy as np
from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD, convert_NCNSxD_to_NCxNSxD
import mapvbvd
import warnings

def normalize_frequency_locations(samples, Kmax=None):
    """
    This function normalizes the sample locations between [-0.5; 0.5[ for
    the non-Cartesian case.
    Parameters
    ----------
    samples: np.ndarray
        Unnormalized samples
    Kmax: int, float, array-like or None
        Maximum Frequency of the samples locations is supposed to be equal to
        base Resolution / (2* Field of View)
    Returns
    -------
    normalized_samples: np.ndarray
        Same shape as the parameters but with values between [-0.5; 0.5[
    """
    samples_locations = np.copy(samples.astype('float'))
    if Kmax is None:
        Kmax = np.abs(samples_locations).max(axis=0)
    elif isinstance(Kmax, (float, int)):
        Kmax = [Kmax] * samples_locations.shape[-1]
    Kmax = np.array(Kmax)
    samples_locations /= (2 * Kmax)
    if np.abs(samples_locations).max() >= 0.5:
        warnings.warn("Frequencies outside the 0.5 limit will be wrapped.")
        samples_locations = (samples_locations + 0.5) % 1 - 0.5
    return samples_locations

def get_samples(filename,dwell_t, num_adc_samples, kmax):
    sample_locations = convert_NCxNSxD_to_NCNSxD(get_kspace_loc_from_gradfile(filename, dwell_t, num_adc_samples)[0])
    sample_locations = normalize_frequency_locations(sample_locations, Kmax= kmax)
    return sample_locations

def add_phase_kspace(kspace_data, kspace_loc, shifts={}):
    if shifts == {}:
        shifts = (0,) * kspace_loc.shape[1]
    if len(shifts) != kspace_loc.shape[1]:
        raise ValueError("Dimension mismatch between shift and kspace locations! "
                         "Ensure that shifts are right")
    phi = np.zeros_like(kspace_data)
    for i in range(kspace_loc.shape[1]):
        phi += kspace_loc[:, i] * shifts[i]
    phase = np.exp(-2 * np.pi * 1j * phi)
    return kspace_data * phase

def get_shifts(twixObj):
    shifts = tuple([
        0 if
        twixObj.search_header_for_val('Phoenix', ('sWiPMemBlock', 'adFree', str(s))) == []
        else
        twixObj.search_header_for_val('Phoenix', ('sWiPMemBlock', 'adFree', str(s)))[0]
        for s in [7, 6, 8]
    ])
    return(shifts)

def read_kspace_data_rep(twixObj, i):
    kspace_data = twixObj.image[:,:,:,i]
    kspace_data = np.moveaxis(kspace_data, (0,1,2),(2,0,1))
    kspace_data = np.reshape(kspace_data, (32,kspace_data.shape[1]*kspace_data.shape[2]))
    return(kspace_data)

def read_kspace_data_sliding_rep(twixObj, i,div,n_shots):
    if i%div==0:
        kspace_data = twixObj.image[:,:,:,i//div]
        kspace_data = np.moveaxis(kspace_data, (0,1,2),(2,0,1))
        kspace_data = np.reshape(kspace_data, (32,kspace_data.shape[1]*kspace_data.shape[2]))
    else:
        kspace_start = twixObj.image[:,:,:,i//div]
        kspace_end = twixObj.image[:,:,:,(i//div)+1]
        kspace_data = np.concatenate( (kspace_start[:,:,(n_shots//div)*(i%4):], kspace_end[:,:,:(n_shots//div)*(i%4)]), axis=-1)
        kspace_data = np.moveaxis(kspace_data, (0,1,2),(2,0,1))
        kspace_data = np.reshape(kspace_data, (32,kspace_data.shape[1]*kspace_data.shape[2]))
    return(kspace_data)

def get_traj_params(traj_file):
    dico_params = get_kspace_loc_from_gradfile(traj_file)[1]
    FOV = dico_params['FOV']
    M = dico_params['img_size']
    Ns = dico_params['num_samples_per_shot']
    OS = dico_params['min_osf']
    n_shots = dico_params['num_shots']  
    return(FOV, M, Ns, OS, n_shots)


