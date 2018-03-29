"""
Gregory Way 2018
scripts/z_sweep_jobs_submit.py

This script will submit several jobs to the PMACS cluster at the University of
Pennsylvania. Each job will train a model for a prespecified number of
components, which corresponds to the latent space dimensionality.


Usage: Run in command line: python scripts/latent_space_sweep_submit.py

     with required command arguments:

       --pmacs_config       filepath pointing to PMACS configuration file
       --param_config       location of tsv file (param by z dimension) for the
                            specific parameter combination for each z dimension
       --out_dir            filepath of where to save the results

     and optional arguments:

       --python_path        absolute path of PMACS python in select environment
                              default: '~/.conda/envs/tybalt-gpu/bin/python'
       --num_seeds          how many models to build (random seeds to set)
                              default: 5
       --components         a comma separated string of z dimensions

Output:
Submit jobs given certain number of latent space dimensionality
"""

import os
import argparse
import pandas as pd
from bsub_helper import bsub_help

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--pmacs_config',
                    help='location of the configuration file for PMACS')
parser.add_argument('-y', '--param_config',
                    help='locaiton of the parameter configuration')
parser.add_argument('-d', '--out_dir', help='folder to store results')
parser.add_argument('-p', '--python_path', help='absolute path of python',
                    default='~/.conda/envs/tybalt-gpu/bin/python')
parser.add_argument('-s', '--num_seeds', default=5,
                    help='number of different seeds to run on current data')
parser.add_argument('-n', '--components', help='dimensionality to sweep over',
                    default='2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,' +
                            '45,50,60,70,80,90,100')
args = parser.parse_args()

pmacs_config_file = args.pmacs_config
param_config_file = args.param_config
out_dir = args.out_dir
python_path = args.python_path
num_seeds = args.num_seeds
components = args.components.split(',')

if not os.path.exists(param_folder):
    os.makedirs(param_folder)

# Load data
config_df = pd.read_table(pmacs_config_file, index_col=0)

# Retrieve PMACS configuration
queue = config_df.loc['queue']['assign']
num_gpus = config_df.loc['num_gpus']['assign']
num_gpus_shared = config_df.loc['num_gpus_shared']['assign']
walltime = config_df.loc['walltime']['assign']

# Set default parameter combination
conda = ['conda', 'activate', 'tybalt-gpu']
default_params = ['--param_config', param_config_file,
                  '--out_dir', out_dir,
                  '--num_seeds', num_seeds]

# Build lists of job commands depending on input algorithm
all_commands = []
for z in components:
    z_command = conda + [python_path, 'train_models_given_z.py',
                         '--num_components', z] + default_params
    all_commands.append(z_command)

# Submit the jobs to PMACS
for command in all_commands:
    b = bsub_help(command=command,
                  queue=queue,
                  num_gpus=num_gpus,
                  num_gpus_shared=num_gpus_shared,
                  walltime=walltime)
    b.submit_command()
