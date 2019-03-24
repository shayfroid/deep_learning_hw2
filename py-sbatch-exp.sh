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
python -m hw2.experiments run-exp -n exp1_MNIST_NORM_CONVVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset MNIST --rounding 0
python -m hw2.experiments run-exp -n exp1_MNIST_NORM_CONVVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset MNIST --rounding 1
python -m hw2.experiments run-exp -n exp1_MNIST_NORM_CONVVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset MNIST --rounding 2
python -m hw2.experiments run-exp -n exp1_MNIST_NORM_CONVVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset MNIST --rounding 100
python -m hw2.experiments run-exp -n exp1_MNIST_NORM_CONVVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset MNIST --rounding 1000

python -m hw2.experiments run-exp -n exp2_MNIST_UNIF_CONVVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset MNIST --rounding 0
python -m hw2.experiments run-exp -n exp2_MNIST_UNIF_CONVVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset MNIST --rounding 1
python -m hw2.experiments run-exp -n exp2_MNIST_UNIF_CONVVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset MNIST --rounding 2
python -m hw2.experiments run-exp -n exp2_MNIST_UNIF_CONVVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset MNIST --rounding 100
python -m hw2.experiments run-exp -n exp2_MNIST_UNIF_CONVVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset MNIST --rounding 1000

python -m hw2.experiments run-exp -n exp3_MNIST_NORM_LINEARVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset MNIST --rounding 0
python -m hw2.experiments run-exp -n exp3_MNIST_NORM_LINEARVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset MNIST --rounding 1
python -m hw2.experiments run-exp -n exp3_MNIST_NORM_LINEARVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset MNIST --rounding 2
python -m hw2.experiments run-exp -n exp3_MNIST_NORM_LINEARVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset MNIST --rounding 100
python -m hw2.experiments run-exp -n exp3_MNIST_NORM_LINEARVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset MNIST --rounding 1000

python -m hw2.experiments run-exp -n exp4_MNIST_UNIF_LINEARVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset MNIST --rounding 0
python -m hw2.experiments run-exp -n exp4_MNIST_UNIF_LINEARVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset MNIST --rounding 1
python -m hw2.experiments run-exp -n exp4_MNIST_UNIF_LINEARVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset MNIST --rounding 2
python -m hw2.experiments run-exp -n exp4_MNIST_UNIF_LINEARVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset MNIST --rounding 100
python -m hw2.experiments run-exp -n exp4_MNIST_UNIF_LINEARVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset MNIST --rounding 1000

python -m hw2.experiments run-exp -n exp5_CIFAR10_NORM_CONVVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset CIFAR10 --rounding 0
python -m hw2.experiments run-exp -n exp5_CIFAR10_NORM_CONVVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset CIFAR10 --rounding 1
python -m hw2.experiments run-exp -n exp5_CIFAR10_NORM_CONVVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset CIFAR10 --rounding 2
python -m hw2.experiments run-exp -n exp5_CIFAR10_NORM_CONVVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset CIFAR10 --rounding 100
python -m hw2.experiments run-exp -n exp5_CIFAR10_NORM_CONVVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVariance --dataset CIFAR10 --rounding 1000

python -m hw2.experiments run-exp -n exp6_CIFAR10_UNIF_CONVVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset CIFAR10 --rounding 0
python -m hw2.experiments run-exp -n exp6_CIFAR10_UNIF_CONVVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset CIFAR10 --rounding 1
python -m hw2.experiments run-exp -n exp6_CIFAR10_UNIF_CONVVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset CIFAR10 --rounding 2
python -m hw2.experiments run-exp -n exp6_CIFAR10_UNIF_CONVVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset CIFAR10 --rounding 100
python -m hw2.experiments run-exp -n exp6_CIFAR10_UNIF_CONVVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5ConvVarianceUnif --dataset CIFAR10 --rounding 1000

python -m hw2.experiments run-exp -n exp7_CIFAR10_NORM_LINEARVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset CIFAR10 --rounding 0
python -m hw2.experiments run-exp -n exp7_CIFAR10_NORM_LINEARVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset CIFAR10 --rounding 1
python -m hw2.experiments run-exp -n exp7_CIFAR10_NORM_LINEARVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset CIFAR10 --rounding 2
python -m hw2.experiments run-exp -n exp7_CIFAR10_NORM_LINEARVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset CIFAR10 --rounding 100
python -m hw2.experiments run-exp -n exp7_CIFAR10_NORM_LINEARVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVariance --dataset CIFAR10 --rounding 1000

python -m hw2.experiments run-exp -n exp8_CIFAR10_UNIF_LINEARVAR_R0 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset CIFAR10 --rounding 0
python -m hw2.experiments run-exp -n exp8_CIFAR10_UNIF_LINEARVAR_R1  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset CIFAR10 --rounding 1
python -m hw2.experiments run-exp -n exp8_CIFAR10_UNIF_LINEARVAR_R2  --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset CIFAR10 --rounding 2
python -m hw2.experiments run-exp -n exp8_CIFAR10_UNIF_LINEARVAR_R100 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset CIFAR10 --rounding 100
python -m hw2.experiments run-exp -n exp8_CIFAR10_UNIF_LINEARVAR_R1000 --early-stopping 5 --reg 0.0002 --lr 0.0015 --ycn LeNet5FCVarianceUnif --dataset CIFAR10 --rounding 1000

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

