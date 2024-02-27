import os
import numpy as np
import nibabel as nib

def get_4D_fMRI_data(folder):
    files = os.listdir(folder)
    files = sorted(files, key=lambda x: int((x.split('.')[0])))
    M = np.load(os.path.join(folder, files[0])).shape
    volumes = np.zeros((M[0],M[1],M[2],len(files)), np.float64)
    for i in range(len(files)):
        print(i)
        volumes[:,:,:,i] = np.load(os.path.join(folder, files[i]))
    return(volumes, volumes.shape)

def from_npy_to_nifti(npy_file, ref_nifti, scale):
    npy_volumes = np.load(npy_file)*scale
    npy_volumes = np.moveaxis(npy_volumes, (0,1,2,3), (1,0,2,3))
    npy_volumes = npy_volumes[:,:,::-1,:]
    aff = nib.load(ref_nifti).affine
    nifti_volumes = nib.Nifti1Image(npy_volumes, affine=aff)
    return(nifti_volumes)


def make_movie(filename, array, share_norm=True, fps=2,):
    """Make a movie from a n_frames x N x N array."""
    import imageio
    
    min_val = np.min(array, axis = None if share_norm else (1,2))
    max_val = np.max(array, axis = None if share_norm else (1,2))
    array_val = 255*(array - min_val)/(max_val-min_val)
    array_val = np.uint8(array_val)
    
    imageio.mimsave(filename, array_val, fps=fps,)

