from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD, convert_NCNSxD_to_NCxNSxD
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import warnings
import mat73

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

def get_samples(filename, dwell_t, num_adc_samples, kmax):
    sample_locations = convert_NCxNSxD_to_NCNSxD(get_kspace_loc_from_gradfile(filename, dwell_t, num_adc_samples)[0])
    sample_locations = normalize_frequency_locations(sample_locations, Kmax=kmax)
    return sample_locations
def get_traj_params(traj_file):
    dico_params = get_kspace_loc_from_gradfile(traj_file)[1]
    FOV = dico_params['FOV']
    M = dico_params['img_size']
    Ns = dico_params['num_samples_per_shot']
    OS = dico_params['min_osf']
    return(FOV, M, Ns, OS)

def read_meas_matrix(meas_file, ndummy):
    #Get kxyz: mat73 is used for big matrices
    try:
        meas_traj =scipy.io.loadmat(meas_file)['ksphaFilt'][::2,1:,:]/(2*np.pi)
    except:
        meas_traj =mat73.loadmat(meas_file)['ksphaFilt'][::2,1:,:]/(2*np.pi)
    meas_traj = np.moveaxis(meas_traj, (0,1,2), (1,2,0) ) 
    meas_traj = meas_traj[ndummy:,:,:]

    #Get k0
    try: 
        meas_k0 = scipy.io.loadmat(meas_file)['ksphaFilt'][::2,:1,:]
    except:
        meas_k0 = mat73.loadmat(meas_file)['ksphaFilt'][::2,:1,:]
    meas_k0 = np.moveaxis(meas_k0, (0,1,2), (1,2,0) ) 
    meas_k0 = meas_k0[ndummy:,:,:]
    return(meas_traj, meas_k0)

def find_skope_delay(s_range,e_range, meas_traj, kspace_loc,nsamp, nshots):
    corr = []
    delay = np.arange(s_range,e_range, 1)
    for i in delay: 
        c = np.corrcoef(-meas_traj[:nshots,int(i):int(i)+nsamp,0].flatten(),kspace_loc[:,:,1].flatten())
        corr.append( c [1,0] )
        ind = np.argmax(corr)
    return(corr, ind, delay, corr[ind], delay[ind])

def extract_synch_trajs_and_k0(meas_traj, meas_k0,dim,nsamp, delay_ind, Kmax):
    new_meas_traj = np.zeros((meas_traj.shape[0], nsamp, dim), np.float64)
    new_meas_traj[:,:,0] = -meas_traj[:,delay_ind:delay_ind+nsamp,1]
    new_meas_traj[:,:,1] = -meas_traj[:,delay_ind:delay_ind+nsamp,0]
    new_meas_traj[:,:,2] = -meas_traj[:,delay_ind:delay_ind+nsamp,2]
    new_meas_traj = convert_NCxNSxD_to_NCNSxD(new_meas_traj)
    new_meas_traj = normalize_frequency_locations(new_meas_traj, Kmax=Kmax)
    meas_k0 = meas_k0[:,delay_ind:delay_ind+nsamp]
    meas_k0 = convert_NCxNSxD_to_NCNSxD(meas_k0)
    return(new_meas_traj, meas_k0)

def save_trajs_and_k0(new_meas_traj, meas_k0, path_traj, path_k0):
    np.save(path_traj, new_meas_traj)
    np.save(path_k0, meas_k0)
    

def ECCPhase_prep(ecc_file, nshots, nsamp):
    eccP = scipy.io.loadmat(ecc_file)['ECCPhase']
    eccP = np.moveaxis(eccP, (0,1), (1,0))
    eccP = np.reshape(eccP, (nshots,nsamp,1))
    eccP = convert_NCxNSxD_to_NCNSxD(eccP)
    ecc = np.exp(-1j*eccP)
    return(ecc)

def get_traj_params(traj_file):
    dico_params = get_kspace_loc_from_gradfile(traj_file)[1]
    FOV = dico_params['FOV']
    M = dico_params['img_size']
    Ns = dico_params['num_samples_per_shot']
    OS = dico_params['min_osf']
    n_shots = dico_params['num_shots']  
    return(FOV, M, Ns, OS,n_shots)