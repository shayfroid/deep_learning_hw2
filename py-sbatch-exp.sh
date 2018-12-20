#!/bin/bash

###
# CS236605: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
QUEUE=236605
JOB_NAME="test_job"
MAIL_USER="jonathan@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236605-hw

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	-p $QUEUE \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python -m hw2.experiments run-exp -n exp1_1_K32_L2 -K 32 -L 2 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_1_K32_L4 -K 32 -L 4 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_1_K32_L6 -K 32 -L 6 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_1_K32_L8 -K 32 -L 8 -P 2 -H 100

python -m hw2.experiments run-exp -n exp1_1_K64_L2 -K 64 -L 2 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_1_K64_L4 -K 64 -L 4 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_1_K64_L6 -K 64 -L 6 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_1_K64_L8 -K 64 -L 8 -P 2 -H 100


python -m hw2.experiments run-exp -n exp1_2_K32_L2 -K 32 -L 2 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_2_K64_L2 -K 64 -L 2 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_2_K128_L2 -K 128 -L 2 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_2_K256_L2 -K 256 -L 2 -P 2 -H 100

python -m hw2.experiments run-exp -n exp1_4_K32_L2 -K 32 -L 4 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_4_K64_L2 -K 64 -L 4 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_4_K128_L2 -K 128 -L 4 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_4_K256_L2 -K 256 -L 4 -P 2 -H 100

python -m hw2.experiments run-exp -n exp1_8_K32_L2 -K 32 -L 8 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_8_K64_L2 -K 64 -L 8 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_8_K128_L2 -K 128 -L 8 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_8_K256_L2 -K 256 -L 8 -P 2 -H 100


python -m hw2.experiments run-exp -n exp1_3_L1_K64-128-256 -K 64 128 256 -L 1 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_3_L2_K64-128-256 -K 64 128 256 -L 2 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_3_L3_K64-128-256 -K 64 128 256 -L 3 -P 2 -H 100
python -m hw2.experiments run-exp -n exp1_3_L4_K64-128-256 -K 64 128 256 -L 4 -P 2 -H 100

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

