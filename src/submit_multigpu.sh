#!/bin/bash

#SBATCH --job-name=multigpu
##SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # number of MP tasks
#SBATCH --gres=gpu:2                # number of GPUs per node
##SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --cpus-per-task=4        # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --comment=pytorch

######################
### Set enviroment ###
######################
#source activateEnviroment.sh
#source ~/module.sh
module load gcc/10.2.0 cuda/12.1 cudampi/openmpi-4.1.1
source ~/.bashrc
conda activate transformer
export HF_HOME=/scratch/qualis
export GPUS_PER_NODE=2
######################

export SCRIPT=/scratch/qualis/llm/transformer/complete_nlp_example.py
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir /scratch/qualis/llm/transformer/outputt \
    --with_tracking \
    "

#accelerate launch --num_processes $GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS
accelerate launch --num_processes $GPUS_PER_NODE /scratch/qualis/llm/transformer/nlp_example.py

