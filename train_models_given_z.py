"""
2018 Gregory Way
train_models_given_z.py

This script will loop through different combinations of latent space
dimensionality, train a distinct model, and save associated decoder weights
and z matrices. The script will need to pull appropriate hyperparameters from
respective files after several initial sweeps testing various hyperparameter
combinations with different z dimensionality.

The script will fit several different compression algorithms and output results

Usage:

    python run_model_with_input_dimensions.py

    With required command line arguments:

        --num_components    The z dimensionality we're testing
        --param_config      A tsv file (param by z dimension) indicating the
                            specific parameter combination for the z dimension
        --out_dir           The directory to store the output files

    And optional command line arguments

        --num_seeds         The number of specific models to generate
"""

import argparse
import numpy as np
import pandas as pd
from tybalt.data_models import DataModel

def get_lowest_loss(matrix_list, reconstruction_df,
                    algorithms=['pca', 'ica', 'nmf', 'dae', 'vae']):
    """
    Determine the specific model with the lowest loss using reconstruciton cost
    """
    final_matrix_list = []
    for alg in algorithms:
        # Get the lost reconstruction cost for given algorithm
        min_recon_subset = reconstruciton_df.loc[alg, :].idxmin

        # subset the matrix to the minimum loss
        best_matrix = matrix_list[min_recon_subset]

        # Extract the algorithm specific columns from the concatenated matrix
        use_cols = best_matrix.columns.str.starts_with(alg)
        best_matrix_subset = best_matrix.loc[:, use_cols]

        # Build the final matrix that will eventually be output
        final_matrix_list.append(best_matrix_subset)

    return pd.concat(final_matrix_list)


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_components', help='dimensionality of z')
parser.add_argument('-s', '--num_seeds', default=5,
                    help='number of different seeds to run on current data')
parser.add_argument('-p', '--param_config',
                    help='text file optimal hyperparameter assignment for z')
parser.add_argument('-o', '--out_dir', help='where to save the output files')
args = parser.parse_args()

# Load command arguments
num_components = args.num_components
seeds = args.seeds
param_config = args.param_config

# Extract parameters from parameter configuation file
loss = param_config.loc[num_components, 'loss'][0]
vae_epochs = param_config.loc[num_components, 'vae_epochs'][0]
dae_epochs = param_config.loc[num_components, 'dae_epochs'][0]
vae_lr = param_config.loc[num_components, 'vae_lr'][0]
dae_lr = param_config.loc[num_components, 'dae_lr'][0]
vae_batch_size = param_config.loc[num_components, 'vae_batch_size'][0]
dae_batch_size = param_config.loc[num_components, 'dae_batch_size'][0]
dae_noise = param_config.loc[num_components, 'dae_noise'][0]
dae_sparsity = param_config.loc[num_components, 'dae_sparsity'][0]
vae_kappa = param_config.loc[num_components, 'vae_kappa'][0]

# Output file names
file_prefix = '{}_components_'.format(num_components)
w_file = os.path.join(out_dir, '{}weight_matrix.tsv'.format(file_prefix))
z_file = os.path.join(out_dir, '{}z_matrix.tsv'.format(file_prefix))
recon_file = os.path.join(out_dir, '{}reconstruction.tsv'.format(file_prefix))

# Load Data
data_df = pd.read_table('data', 'pancan_rnaseq_scaled_zeroone.tsv')

# Build models
random_seeds = np.random.randint(0, high=1000000, size=num_seeds)
algorithms = ['pca', 'ica', 'nmf', 'adage', 'tybalt']

z_matrices = []
weight_matrices = []
reconstruction_results = []
for seed in random_seeds:
    data_model.pca(n_components=num_components)
    data_model.ica(n_components=num_components)
    data_model.nmf(n_components=num_components)

    data_model.nn(n_components=num_components, model='tybalt', loss=loss,
                  epochs=vae_epochs, batch_size=vae_batch_size,
                  learning_rate=vae_learning_rate, verbose=False)
    data_model.nn(n_components=n_components, model='adage', loss=loss,
                  epochs=dae_epochs, batch_size=dae_batch_size,
                  learning_rate=dae_learning_rate, noise=dae_noise,
                  sparsity=dae_sparsity, verbose=False)

    # Obtain z matrix (sample scores per latent space feature) for all models
    z_matrix = data_models.compile_z_matrix(algorithms)

    # Obtain weight matrices (gene by latent space feature) for all models
    weight_matrix = data_models.compile_weight_matrix(algorithms)

    # Store reconstruction costs at training end
    reconstruction = data_models.compile_reconstruction(algorithms)

    # Concatenate Results
    z_matrices.append(z_matrix)
    weight_matrices.append(weight_matrix)
    reconstruction_results.append(reconstruction)

# Identify models with the lowest loss across seeds and chose these to save
reconstruction_df = pd.concat(reconstruction_results)

# Process the final matrices that are to be stored
final_weight_matrix = get_lowest_loss(weight_matrices, reconstruction_df)
final_z_matrix = get_lowest_loss(z_matrices, reconstruction_df)

# Output files
final_weight_matrix.to_csv(w_file, sep='\t')
final_z_matrix.to_csv(z_file, sep='\t')
reconstruction_df.to_csv(recon_file, sep='\t')
