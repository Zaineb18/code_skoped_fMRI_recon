#!/bin/bash

module purge

module load python/3.8.8 cuda/10.2 cudnn/8.0.4.30-cuda-10.2 intel-compilers/19.0.4 openmpi/4.0.2-cuda nccl/2.6.4-1-cuda cmake/3.21.3 gcc/7.3.0
module load cpuarch/amd

module load cuda/10.2 
conda activate csfmri_env 
