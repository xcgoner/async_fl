#!/bin/bash
#PBS -l select=1:ncpus=112 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=56


### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

basedir=/home/user/fedasync
logdir=/homs/user/fedasync/results

# training data
inputdir=$basedir/dataset

# validation data
valdir=$inputdir/dataset_split_100

# prepare datasets
# python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output $inputdir

watchfile=$logdir/experiment_script_1.log

model="default"
lr=0.1
rho=0.01
alpha=0.8
maxdelay=16
localepochs=1
seed=337

logfile=$logdir/fedasync.txt

python train_cifar10_mxnet_fedasync_singlethread_impl.py --classes 10 --model ${model} --nsplit 100 --batchsize 50 --lr ${lr} --rho ${rho} --alpha ${alpha} --alpha-decay 0.5 --alpha-decay-epoch 800 --epochs 2000 --max-delay ${maxdelay} --iterations ${localepochs} --seed ${seed} --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile

