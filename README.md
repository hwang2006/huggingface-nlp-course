# huggingface-nlp-course

This repository contains hands-on exercises designed to run the [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) on a supercomputer, allowing users to leverage SLURM-managed supercomputer for advanced natural language processing tasks and learning experiences.  

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Installing Conda](#installing-conda)
* [Creating a Conda Virtual Environment](#creating-a-conda-virtual-environment)
* [Running Jupyter](#running-jupyter)


## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
no change     /scratch/qualis/miniconda3/etc/profile.d/conda.csh
modified      /home01/qualis/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 23.9.0
```

## Creating a Conda Virtual Environment
You want to create a virtual envrionment with a python version 3.10 for Generative AI Practices.
```
[glogin01]$ conda create -n hf-nlp-course python=3.10
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/hf-nlp-course

  added / updated specs:
    - python=3.10
.
.
.
Proceed ([y]/n)? y    <========== type yes 


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate hf-nlp-course
#
# To deactivate an active environment, use
#
#     $ conda deactivate

[glogin01]$ conda activate hf-nlp-course
(hf-nlp-course) [glogin01]$ pip pip install scipy scikit-learn huggingface_hub bitsandbytes accelerate evaluate seqeval transformers datasets deepspeed
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting scipy
  Downloading scipy-1.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
Collecting scikit-learn
  Downloading scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (13 kB)
Requirement already satisfied: huggingface_hub in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (0.25.0)
Collecting bitsandbytes
  Downloading bitsandbytes-0.42.0-py3-none-any.whl.metadata (9.9 kB)
Collecting accelerate
  Downloading accelerate-0.34.2-py3-none-any.whl.metadata (19 kB)
Collecting evaluate
  Downloading evaluate-0.4.3-py3-none-any.whl.metadata (9.2 kB)
Collecting seqeval
  Downloading seqeval-1.2.2.tar.gz (43 kB)
  Preparing metadata (setup.py) ... done
Requirement already satisfied: transformers in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (4.44.2)
Requirement already satisfied: datasets in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (3.0.0)
Requirement already satisfied: numpy<2.3,>=1.23.5 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from scipy) (2.0.1)
Collecting joblib>=1.2.0 (from scikit-learn)
  Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn)
  Downloading threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: filelock in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (3.13.1)
Requirement already satisfied: fsspec>=2023.5.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (2024.6.1)
Requirement already satisfied: packaging>=20.9 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (24.1)
Requirement already satisfied: pyyaml>=5.1 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (6.0.1)
Requirement already satisfied: requests in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (2.32.3)
Requirement already satisfied: tqdm>=4.42.1 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (4.66.5)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from huggingface_hub) (4.11.0)
Requirement already satisfied: psutil in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from accelerate) (5.9.0)
Requirement already satisfied: torch>=1.10.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from accelerate) (2.4.0)
Requirement already satisfied: safetensors>=0.4.3 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from accelerate) (0.4.5)
Requirement already satisfied: dill in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from evaluate) (0.3.8)
Requirement already satisfied: pandas in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from evaluate) (2.2.3)
Requirement already satisfied: xxhash in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from evaluate) (3.5.0)
Requirement already satisfied: multiprocess in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from evaluate) (0.70.16)
Requirement already satisfied: regex!=2019.12.17 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from transformers) (2024.9.11)
Requirement already satisfied: tokenizers<0.20,>=0.19 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from transformers) (0.19.1)
Requirement already satisfied: pyarrow>=15.0.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from datasets) (17.0.0)
Requirement already satisfied: aiohttp in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from datasets) (3.10.5)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (2.4.0)
Requirement already satisfied: aiosignal>=1.1.2 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (23.1.0)
Requirement already satisfied: frozenlist>=1.1.1 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (1.4.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (6.1.0)
Requirement already satisfied: yarl<2.0,>=1.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (1.11.1)
Requirement already satisfied: async-timeout<5.0,>=4.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from aiohttp->datasets) (4.0.3)
Requirement already satisfied: charset-normalizer<4,>=2 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from requests->huggingface_hub) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from requests->huggingface_hub) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from requests->huggingface_hub) (2.2.2)
Requirement already satisfied: certifi>=2017.4.17 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from requests->huggingface_hub) (2024.8.30)
Requirement already satisfied: sympy in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (1.13.2)
Requirement already satisfied: networkx in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (3.2.1)
Requirement already satisfied: jinja2 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (3.1.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from pandas->evaluate) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from pandas->evaluate) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from pandas->evaluate) (2024.1)
Requirement already satisfied: six>=1.5 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas->evaluate) (1.16.0)
Requirement already satisfied: MarkupSafe>=2.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /scratch/qualis/miniconda3/envs/hf-nlp-course/lib/python3.10/site-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)
Downloading scipy-1.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (41.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.2/41.2 MB 289.6 MB/s eta 0:00:00
Downloading scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.3/13.3 MB 349.5 MB/s eta 0:00:00
Downloading bitsandbytes-0.42.0-py3-none-any.whl (105.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.0/105.0 MB 50.3 MB/s eta 0:00:00
Downloading accelerate-0.34.2-py3-none-any.whl (324 kB)
Downloading evaluate-0.4.3-py3-none-any.whl (84 kB)
Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
Building wheels for collected packages: seqeval
  Building wheel for seqeval (setup.py) ... done
  Created wheel for seqeval: filename=seqeval-1.2.2-py3-none-any.whl size=16161 sha256=20ef7d1cc2849b2fd6e6f914ef34dd6df2221b2b01a14b7e2857a5160b45e0dd
  Stored in directory: /tmp/pip-ephem-wheel-cache-8khnxgsg/wheels/1a/67/4a/ad4082dd7dfc30f2abfe4d80a2ed5926a506eb8a972b4767fa
Successfully built seqeval
Installing collected packages: threadpoolctl, scipy, joblib, scikit-learn, bitsandbytes, seqeval, accelerate, evaluate
Successfully installed accelerate-0.34.2 bitsandbytes-0.42.0 evaluate-0.4.3 joblib-1.4.2 scikit-learn-1.5.2 scipy-1.14.1 seqeval-1.2.2 threadpoolctl-3.5.0
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from the browser on your PC or labtop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the `hf-nlp-course` virtual envrionment that you have created as a python kernel.
1. activate the `hf-nlp-course` virtual environment, if it's not activated:
```
[glogin01]$ conda activate hf-nlp-course
```
2. install Jupyter on the virtual environment:
```
(hf-nlp-course) [glogin01]$ conda install jupyter chardet cchardet 
  
```
3. add the virtual environment as a jupyter kernel:
```
(hf-nlp-course) [glogin01]$ python -m ipykernel install --user --name hf-nlp-course 
```
4. check the list of kernels currently installed:
```
(hf-nlp-course) [glogin01]$ jupyter kernelspec list
Available kernels:
python3               /home01/$USER/.local/share/jupyter/kernels/python3
hf-nlp-course         /home01/$USER/.local/share/jupyter/kernels/hf-nlp-course
```
5. launch a jupyter notebook server on a worker node 
- to deactivate the virtual environment
```
(hf-nlp-course) [glogin01]$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
[glogin01]$ cat jupyter_run.sh
#!/bin/bash
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
##SBATCH --partition=amd_a100nv_8
#SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=8     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
#module load gcc/10.2.0 cuda/11.6
module load gcc/10.2.0 cuda/12.1

echo "execute jupyter"
source ~/.bashrc
conda activate hf-nlp-course
cd /scratch/$USER  # the root/work directory of Jupyter lab/notebook
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --no-browser --NotebookApp.token=${USER} #jupyter token: your account ID
echo "end of the job"

```
- to launch a jupyter notebook server 
```
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```
- to check if the jupyter notebook server is up and running
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a new SSH client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc) on your PC or laptop and log in to the Neuron system just by copying and pasting the port_forwarding_command:

![20240123_102609](https://github.com/hwang2006/Generative-AI-with-LLMs/assets/84169368/1f5dd57f-9872-491b-8dd4-0aa99b867789)

7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888
Password or token: $USER    # your account name on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 
