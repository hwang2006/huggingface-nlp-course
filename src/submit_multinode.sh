#!/bin/bash

#SBATCH --job-name=multinode
##SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=2         # number of MP tasks
#SBATCH --gres=gpu:2                # number of GPUs per node
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

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "
export SCRIPT="/scratch/qualis/llm/transformer/complete_nlp_example.py"
export SCRIPT1="/scratch/qualis/llm/transformer/nlp_example.py"
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir /scratch/qualis/llm/transformer/outputt \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
export CMD1="$LAUNCHER $SCRIPT1"  
echo $LAUNCHER
#srun $CMD
echo $CMD1
srun $CMD1
