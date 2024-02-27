#!/bin/bash
##SBATCH --job-name=ReconFMRI
##SBATCH --partition=gpu_p2l
##SBATCH --account=hih@v100
##SBATCH --ntasks=1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=3
##SBATCH --hint=nomultithread
##SBATCH --time=20:00:00
##SBATCH --qos=qos_gpu-t3
##SBATCH --output=fmriRecon_arraygpu_%A_%a.out # nom du fichier de sortie
##SBATCH --error=fmriRecon_arraygpu_%A_%a.out
##SBATCH --array=1-2

#SBATCH --job-name=ReconFMRI
##SBATCH --partition=gpu_p4
##SBATCH --account=hih@v100
#SBATCH -C a100
#SBATCH --account=hih@a100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=fmriRecon_arraygpu_%A_%a.out # nom du fichier de sortie
#SBATCH --error=fmriRecon_arraygpu_%A_%a.out
#SBATCH --array=1-500


set -x
#NUM_GPUS=8
GROUP=1
source $HOME/mod_loading.sh
ctr=0
for ((j=0; j<500; j++))
do
	#export CUDA_VISIBLE_DEVICES=$(($(($((ctr))%NUM_GPUS))))
        ctr=$((ctr+1))
	for ((i=0; i<$GROUP; i++))
	do
		if [ $ctr -eq $SLURM_ARRAY_TASK_ID ]
		then    
			index=$(($GROUP*$j+$i))     	
			python recon_script.py --i=$index --div=4 --obs="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/R2/meas_MID246_3D_SPARKLING_1mmiso_2_4s_R2_FID12651.dat"  --traj="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/R2/kxyz_R2.npy"  --ecc="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/ecc.npy" --k0="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/R2/k0_R2.npy"  --out="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Output/R2/"  --mask="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/R2/Mask_R2.npy" --b0="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/R2/B0_R2.npy" --smaps="/gpfsstore/rech/hih/uwa98fg/Data_Zaineb/cr170417_13022024_LewisLikeParadigm/Input/R2/R2_MaskedSmaps.npy"   &
			echo "Running $index"
		fi 	
	done

done  
wait
