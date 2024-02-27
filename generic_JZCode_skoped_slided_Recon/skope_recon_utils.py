from recon_utils import *
import numpy as np
from mri.operators import NonCartesianFFT, WaveletN, ORCFFTWrapper
from mri.reconstructors import SelfCalibrationReconstructor
from mri.operators.fourier.utils import estimate_density_compensation
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold


def ECC_undone(kspace_data, ecc):
    kspace_data_undoECC = np.zeros_like(kspace_data)
    for i in range(32):
        kspace_data_undoECC[i,:] = np.multiply(kspace_data[i,:], ecc)
    return(kspace_data_undoECC)

def demodulate_k0(k0, kspace_data_undoECC):
    demod_k0 = np.complex64(np.exp(-1j*k0[:,0]))
    demod_obs = np.zeros_like(kspace_data_undoECC)
    print(demod_k0.shape, demod_obs.shape)
    for i in range(32):
        demod_obs[i,:] = np.multiply(kspace_data_undoECC[i,:], demod_k0)
    return(demod_obs)


def recon_volume(twixObj, kspace_loc, M, smaps_file, shifts_slice, i, k0, ecc,output_folder, b0Map, mask, use_k0=True, use_B0=True):
    # read kspace data and 0'th order dynamic correction
    kspace_data = read_kspace_data_rep(twixObj, i)
    kspace_data = add_phase_kspace(kspace_data, kspace_loc, shifts=shifts_slice)
    if use_k0==True:
        L = kspace_loc.shape[0]
        print(L, k0[i*L:(i+1)*L,:].shape)
        kspace_data = ECC_undone(kspace_data, ecc[i*L:(i+1)*L,0])
        kspace_data = demodulate_k0(k0[i*L:(i+1)*L,:], kspace_data)
    # read smaps
    smaps = np.load(smaps_file, allow_pickle=True, fix_imports=True)
    print("data read and demodulated")
    # define the reconstructor's operators
    density_comp = estimate_density_compensation(kspace_loc, M)
    regularizer_op = SparseThreshold(Identity(), 1e-8, thresh_type="soft")
    linear_op = WaveletN(wavelet_name='sym8',
                         nb_scale=3,
                         dim=3,
                         padding='periodization')
    if use_B0==False:
        fourier_op = NonCartesianFFT(samples=kspace_loc,
                                 shape=M,
                                 implementation='gpuNUFFT',
                                 density_comp=density_comp,
                                 n_coils=32, )
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=fourier_op,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation='synthesis',
            Smaps=smaps,
            n_jobs=5,
            verbose=21
        )

    if use_BO==True:
        dwell_time = 10e-6 / 5
        nb_adc_samples = 13440
        Te = 20e-3
        time_vec = dwell_time * np.arange(nb_adc_samples)
        echo_time = Te - dwell_time * nb_adc_samples / 2
        time_vec = (time_vec + echo_time).astype(np.float32)
        ifourier_op = ORCFFTWrapper(fourier_op, b0Map, time_vec, mask, n_bins=1000, num_interpolators=30)
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=ifourier_op,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation='synthesis',
            Smaps = smaps,
            num_check_lips = 0,
            lipschitz_cst = 0.06,
            # lips_calc_max_iter = 5,
            n_jobs = 5,
            verbose = 5
        )

    print("reconstructor defined")
    reconst_data, cost, _ = reconstructor.reconstruct(kspace_data=kspace_data,
                                                      optimization_alg='pogm', num_iterations=15,
                                                      recompute_smaps=False,
                                                      )
    filename = str(i)
    np.save(output_folder + filename, abs(reconst_data),
            allow_pickle=True,
            fix_imports=True)
    return (i, reconst_data)



