from recon_utils import *
from skope_recon_utils import *
import numpy as np
from mri.operators import NonCartesianFFT, WaveletN, ORCFFTWrapper
from mri.reconstructors import SelfCalibrationReconstructor
from mri.operators.fourier.utils import estimate_density_compensation
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import mapvbvd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--div",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--obs",
        type=str,
    )
    parser.add_argument(
        "--traj",
        type=str,
    )
    parser.add_argument(
        "--ecc",
        type=str,
    )
    parser.add_argument(
        "--k0",
        type=str,
    )
    parser.add_argument(
        "--out",
        type=str,
    )
    parser.add_argument(
        "--mask",
        type=str,
    )
    parser.add_argument(
        "--b0",
        type=str,
    )
    parser.add_argument(
        "--smaps",
        type=str,
    )
    args = parser.parse_args()
    i = args.i
    div = int(args.div)
    obs_file = args.obs
    output_folder= args.out
    mask_file = args.mask
    b0_file = args.b0
    smaps_file = args.smaps
    k0_file = args.k0
    ecc_file = args.ecc
    meas_traj_file = args.traj
    print("IND", div)
    #config
    use_meas_traj = True
    use_k0 = True
    use_B0 = True
    
    #load and rearrange kspace locations
    traj_file = "/gpfsstore/rech/hih/uwa98fg/InputData/at_retest/dim3_i_RadialIO_P0.75_N192x192x128_FOV0.192x0.192x0.128_Nc8_Ns2688_c25_d2__D9M9Y2021T1017_reproject.bin"
    FOV, M, Ns, OS, n_shots = get_traj_params(traj_file)
    if use_meas_traj==False:
        kspace_loc_orig = get_samples(traj_file, 0.01/OS, Ns*OS,kmax=(M[0]/(2*FOV[0]), M[1]/(2*FOV[1]), M[2]/(2*FOV[2])))
        density_comp_orig = estimate_density_compensation(kspace_loc_orig, [M])
        if i%div==0:
            kspace_loc = kspace_loc_orig
            density_comp = density_comp_orig
        else:
            kspace_loc = np.concatenate(
                    (kspace_loc_orig[(n_shots//div)*(i%div)*Ns*OS:,:],
                        kspace_loc_orig[:(n_shots//div)*(i%div)*Ns*OS,:])
             )
            density_comp = np.concatenate(
            (density_comp_orig[(n_shots//div)*(i%div)*Ns*OS:],
            density_comp_orig[:(n_shots//div)*(i%div)*Ns*OS])
             )
    else:
        kspace_loc = np.load(meas_traj_file)
        kspace_loc = kspace_loc[i*(n_shots//div)*Ns*OS:(n_shots+i*(n_shots//div))*Ns*OS,:]
        density_comp = estimate_density_compensation(kspace_loc, M)
        #kspace_loc = kspace_loc[i*12*13440:(48+i*12)*13440, :]
        #kspace_loc = kspace_loc[i*24*13440:(48+i*24)*13440, :]
    
    #load and rearrange kspace data
    twixObj = mapvbvd.mapVBVD(obs_file)
    twixObj.image.flagRemoveOS = False
    twixObj.image.squeeze = True
    shifts = get_shifts(twixObj)
    shifts = ((shifts[0] * M[0])/FOV[0],(shifts[1] * M[1])/FOV[1],(shifts[2] * M[2])/FOV[2])
    shifts = (shifts[0]*0.001,shifts[1]*0.001, shifts[2]*0.001)
    print("\n Dimensions: ", kspace_loc.shape)
    print("\n Shifts", shifts)
    if div!=1: #div ==1 means the reconstruction is a simple sequential one
        kspace_data = read_kspace_data_sliding_rep(twixObj, i, div, n_shots)
    else:     
        kspace_data = read_kspace_data_rep(twixObj, i)

    kspace_data = add_phase_kspace(kspace_data, kspace_loc, shifts=shifts)
    
    b0Map = np.load(b0_file)
    mask = np.load(mask_file)
    
    #prep static B0 correction
    dwell_time = 10e-6 / 5
    nb_adc_samples = Ns*OS
    Te = 20e-3
    time_vec = dwell_time * np.arange(nb_adc_samples)
    echo_time = Te - dwell_time * nb_adc_samples / 2
    time_vec = (time_vec + echo_time).astype(np.float32)

    #load data for 0 dynamic correction
    k0 = np.load(k0_file)
    ecc = np.complex64(np.load(ecc_file))
    #kspace_data = read_kspace_data_sliding_rep(twixObj, i, div, n_shots)
    #kspace_data = read_kspace_data_rep(twixObj, i)
    #kspace_data = add_phase_kspace(kspace_data, kspace_loc, shifts=shifts)
    del(twixObj)
   
    #demodulate kspace data
    if use_k0 == True:
        #L = 645120
        print("\n ECC", ecc[i*(n_shots//div)*Ns*OS:(n_shots+i*(n_shots//div))*Ns*OS, 0].shape)
        #print("ECC", ecc[i*24*13440:(48+i*24)*13440, 0].shape)

        print("\n kspace_data", kspace_data.shape)
        kspace_data = ECC_undone(kspace_data, ecc[i*(n_shots//div)*Ns*OS:(n_shots+i*(n_shots//div))*Ns*OS, 0])
        #kspace_data = ECC_undone(kspace_data, ecc[i*24*13440:(48+i*24)*13440, 0])
        print("\n kspace_data", kspace_data.shape)
        print("\n k0", k0[i*(n_shots//div)*Ns*OS:(n_shots+i*(n_shots//div))*Ns*OS, :].shape)
        #print("k0", k0[i*24*13440:(48+i*24)*13440, :].shape)

        kspace_data = demodulate_k0(k0[i*(n_shots//div)*Ns*OS:(n_shots+i*(n_shots//div))*Ns*OS, :], kspace_data)
        #kspace_data = demodulate_k0(k0[i*24*13440:(48+i*24)*13440, :], kspace_data)
   
    #read smaps
    smaps = np.load(smaps_file, allow_pickle=True, fix_imports=True)
    #smaps = smaps[:,:,:,2:-2]
    print("\n data read and demodulated")

    #define the reconstructor's operators
    #density_comp = estimate_density_compensation(kspace_loc, M)
    print("\n DESNITY COMP DIMS", density_comp.shape)
    regularizer_op = SparseThreshold(Identity(), 1e-8, thresh_type="soft")
    linear_op = WaveletN(wavelet_name='sym8',
                     nb_scale=3,
                     dim=3,
                     padding='periodization')
    fourier_op = NonCartesianFFT(samples=kspace_loc,
                             shape=M,
                             implementation='gpuNUFFT',
                             density_comp=density_comp,
                             n_coils=32, )
    if use_B0==True:
        ifourier_op = ORCFFTWrapper(fourier_op, b0Map, time_vec, mask, n_bins=1000, num_interpolators=30)
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=ifourier_op,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation='synthesis',
            Smaps=smaps,
            num_check_lips=0,
            lipschitz_cst=0.06,
            # lips_calc_max_iter = 5,
            n_jobs=5,
            verbose=5
    )
        print("\n reconstructor defined with b0")
    else:
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=fourier_op,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation='synthesis',
            Smaps=smaps,
            n_jobs=5,
            verbose=5
             )
        print("\n reconstructor defined without b0")

    reconst_data, cost, _ = reconstructor.reconstruct(kspace_data=kspace_data,
                                                  optimization_alg='pogm', num_iterations=15,
                                                  recompute_smaps=False,
                                                  )
    filename = str(i)
    np.save(output_folder + filename, abs(reconst_data),
        allow_pickle=True,
        fix_imports=True)
