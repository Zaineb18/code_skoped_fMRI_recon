from numpy.fft import ifftn, ifftshift, fftn, fftshift
import mapvbvd
from skimage.filters import window
from skimage.filters import threshold_li
import scipy.io
import numpy as np
import scipy.ndimage

def read_sense_maps_kspace(obs_file):
    twixObj = mapvbvd.mapVBVD(obs_file)
    twixObj.image.flagRemoveOS = False
    kspace_data = twixObj.image[:, :, :, :, 0]
    kspace_data = np.reshape(kspace_data, (128, 32, 64, 44))
    obs_data = np.moveaxis(kspace_data, (0, 1, 2, 3), (3, 0, 1, 2))
    return(obs_data)

def recon_uncombined_data(obs_data, target_res):
    #prep hamming window and recon matrix
    windHamming = window('hamming', (64, 44, 64 * 2))
    windHamming_f = fftshift(fftn(windHamming))
    n_coils = obs_data.shape[0]
    #recons_per_channel = np.zeros((32, 192, 192, 132), np.complex64)
    #recons_per_channel = np.zeros((32, 110, 110, 76), np.complex64)
    dim_x = int(192/target_res[0])
    dim_y = int(192/target_res[1])
    dim_z = int(132/target_res[2])
    recons_per_channel = np.zeros((32, dim_x, dim_y,dim_z ), np.complex64)

    # recon per channel
    for i in range(n_coils):
        # get data per channel, reshape it, pad it.
        obs_channel = obs_data[i, :, :, :]
        obs_reshaped = windHamming * obs_channel
        #obs_data_pad = np.zeros((110, 76, 110 * 2), np.complex64)
        #obs_data_pad[64:128, 44:88, 128:256] = obs_reshaped
        #obs_data_pad[23:87,16:60,46:174] = obs_reshaped
        obs_data_pad = np.zeros((dim_x, dim_z, dim_y * 2), np.complex64)
        obs_data_pad[(dim_x//2)-(64//2):(dim_x//2)+(64//2), (dim_z//2)-(44//2):(dim_z//2)+(44//2), (dim_y)-64:(dim_y)+64] = obs_reshaped
        #obs_data_pad[32:96, 22:66, 64:192] = obs_reshaped
        obs_data_pad = fftshift(obs_data_pad)
        recon_cart = ifftn(obs_data_pad)
        recon_shifted = ifftshift(recon_cart)
        #recon_cropped = recon_shifted[:, :, 96:192 + 96]
        recon_cropped = recon_shifted[:, :,  (dim_y)-(dim_y//2):(dim_y)+(dim_y//2)]
        recon_moved = np.moveaxis(recon_cropped, (0, 1, 2), (1, 2, 0))
        recons_per_channel[i, :, :, :] = recon_moved
    return(recons_per_channel)

def get_sos_and_mask(recons_per_channel):
    sos = np.sqrt(np.sum(np.abs(recons_per_channel) ** 2, axis=0))
    thresh_li = threshold_li(sos)
    mask_li = np.copy(sos)
    mask_li[mask_li < thresh_li] = 0
    mask_li[mask_li != 0] = 1
    return(sos, mask_li)

def get_smaps_and_masked_smaps(recons_per_channel, sos, mask_li, target_res):
    dim_x = int(192/target_res[0])
    dim_y = int(192/target_res[1])
    dim_z = int(132/target_res[2])
    smaps = np.zeros((32, dim_x, dim_y, dim_z), np.complex64)
    n_coils = recons_per_channel.shape[0]
    for i in range(n_coils):
        smaps[i, :, :, :] = recons_per_channel[i, :, :, :] / sos
    masked_smaps = np.multiply(smaps, mask_li)
    return(smaps, masked_smaps)

def save_smaps(smaps, masked_smaps, path_smaps, path_masked_smaps): 
    masked_smaps = np.moveaxis(masked_smaps, (0,1,2,3), (0,2,1,3))
    smaps = np.moveaxis(smaps, (0,1,2,3), (0,2,1,3))
    np.save(path_smaps, smaps) 
    np.save(path_masked_smaps, masked_smaps)
    
def BO_map_and_Mask_gen(b0_file, mask_file, target_res):
    dim_x = int(192/target_res[0])
    dim_y = int(192/target_res[1])
    dim_z = int(132/target_res[2])
    
    mask = scipy.io.loadmat(mask_file)["Mask"]
    b0 = scipy.io.loadmat(b0_file)["B0map"]
    mask = scipy.ndimage.zoom(mask, (dim_x/64,dim_y/64,dim_z/44))
    b0 = scipy.ndimage.zoom(b0, (dim_x/64,dim_y/64,dim_z/44))
    mask, b0 = mask[::-1,:,::-1], b0[::-1,:,::-1]
    return(b0, mask)

def save_B0map(B0, Mask, path_B0, path_Mask): 
    np.save(path_B0, B0) 
    np.save(path_Mask, Mask)