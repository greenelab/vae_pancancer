# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/util/simulate_expression.R
#
# Functions to facilitate gene expression data simulation
#
# Usage: import only 
# source("scripts/util/simulate_expression.R")

sampleGroupMatrix <- function(num_samples, mean_matrix, sd_matrix) {
  # Sample different "groups" based on mean and standard deviation matrices
  #
  # num_samples - the number of samples to simulate
  # mean_matrix - a matrix of different group means
  # sd_matrix - a matrix of different group standard deviations
  #
  # The rows of each matrix indicate group specific data (group mean or group
  # standard deviation) and the columns represent different group features.
  # (nrow = num groups) (ncol = num features that describe each group)
  #
  # Return:
  # list of length 2 - 1st element is the group specific feature matrix
  #                  - 2nd element is a vector of labels ('a', 'b', 'c', etc.)

  group_params <- c()
  group_name <- letters[1:nrow(mean_matrix)]
  for (param_idx in 1:ncol(mean_matrix)) {
    mean_vector <- mean_matrix[, param_idx]
    sd_vector <- sd_matrix[, param_idx]

    group_vector <- rnorm(n, mean = mean_vector, sd = sd_vector)
    group_params <- cbind(group_params, group_vector)
  }

  return_list <- list()
  return_list[[1]] <- group_params
  return_list[[2]] <- rep(group_name, num_samples / length(group_name))
  return(return_list)
}


sampleCellMatrix <- function(num_samples, cell_mean_matrix, cell_sd_matrix) {
  # Sample "cell-types" and then add together with different proportions
  #
  # num_samples - the number of samples to simulate
  # cell_mean_matrix - a matrix of cell-type means
  # cell_sd_matrix - a matrix of cell-type standard deviations
  #
  # Each matrix represents features (columns) describing cell-types (rows)
  # (Currently supports only two cell-types)
  #
  # Return:
  # list of length 2 - 1st element is the cell-type mixing data
  #                  - 2nd element is the ground truth cell-type proportion

  # Loop through specific input artificial "cell-types"
  cell_type_params <- list()
  for (cell_type_idx in 1:nrow(cell_mean_matrix)) {

    # loop through specific input cell-type features
    cell_type_feature <- c()
    for (cell_feature_idx in 1:ncol(cell_mean_matrix)) {

      # Obtain and sample from input parameters specific to cell-type feature
      mean_cell <- cell_mean_matrix[cell_type_idx, cell_feature_idx]
      sd_cell <- cell_sd_matrix[cell_type_idx, cell_feature_idx]

      cell_type_vector <- rnorm(num_samples, mean = mean_cell, sd = sd_cell)
      cell_type_feature <- cbind(cell_type_feature, cell_type_vector)
    }

    # Save each feature in internal list
    cell_type_params[[cell_type_idx]] <- cell_type_feature
  }

  # Uniform sampling between 0 and 1 represents random mixing proportions
  rand_cell_type_1 <- runif(num_samples, min = 0, max = 1)
  rand_cell_type_2 <- 1 - rand_cell_type_1
  cell_type_prop_list <- list(rand_cell_type_1, rand_cell_type_2)

  # Loop over sampled cell-type parameters (columns, or features, of input)
  for (cell_idx in 1:length(cell_type_params)) {

    # Perform element-wise multiplication 
    cell_type_params[[cell_idx]] <- cell_type_prop_list[[cell_idx]] * 
      cell_type_params[[cell_idx]]
  }

  # Add mixing proportions of cell-type together
  cell_type_data <- cell_type_params[[1]] + cell_type_params[[2]]
  return_list <- list(cell_type_data, rand_cell_type_1)
  return(return_list)
}


getSimulatedExpression <- function(n, mean_df, sd_df, r, func_list, b,
                                   cell_type_mean_df, cell_type_sd_df,
                                   zero_one_normalize = TRUE) {
  # Obtain a matrix with simulated parameters of input size
  #
  # n - integer indicating the total number of samples
  # mean_df - matrix of means describing groups
  # sd_df - matrix of standard deviations describing groups
  #         for mean and sd, ncol = number of features, nrow = number of groups
  # r - the number of random noise parameters
  # func_list - each element in the list stores a function to apply to a
  #             random noise sampling (each element indicates a single param)
  # b - the number of presence/absence features (independent from group)
  #     (value is either 0, or is sampled from a standard normal)
  # cell_type_mean_df - matrix of means describing artificial cell-types
  # cell_type_sd_df - matrix of standard deviations describing cell-types
  #       Each row represents different cell types (only 2 currently supported)
  #       Each column represents features describing the cell types
  # zero_one_normalize - boolean to zero one normalize simulated features
  #
  # Return:
  # List of length 2: The first element is the simulated data matrix
  #                   The second element is important metadata including group
  #                       membership, cell type proportion, and the domain of
  #                       the input functions.

  if (sum(mean_df + sd_df) == 0) {
    if (all(dim(mean_df) == dim(sd_df))) {
      stop('provide the same number of mean and standard deviation parameters')
    }
  }

  # Extract Group Features
  group_params <- sampleGroupMatrix(n, mean_df, sd_df)
  group_data <- group_params[[1]]
  group_info <- group_params[[2]]

  # Get Random Noise Features
  rand_params <- c()
  for (rand_idx in 1:r) {
    rand_vector <- runif(n, min = 0, max = 1)
    rand_params <- cbind(rand_params, rand_vector)
  }

  # Get Continuous Function Features
  cont_params <- c()
  cont_other_params <- c()
  for (cont_idx in 1:length(func_list)) {
    continuous_rand_x <- runif(n, min = -1, max = 1)
    continuous_rand_y <- func_list[[cont_idx]](continuous_rand_x)
    
    cont_params <- cbind(cont_params, continuous_rand_y)
    cont_other_params <- cbind(cont_other_params, continuous_rand_x)
  }
  
  # Get Presence/Absence of a Features
  pres_params <- c()
  for (pres_idx in 1:b) {
    rand_presence <- rnorm(n, mean = 1, sd = 0.5)
    rand_zerone <- sample(c(0, 1), n, replace = TRUE)

    rand_presence <- rand_presence * rand_zerone
    pres_params <- cbind(pres_params, rand_presence)
  }

  # Get cell-type Features
  if (sum(cell_type_mean_df + cell_type_sd_df) == 0) {
    if (all(dim(cell_type_mean_df) == dim(cell_type_sd_df))) {
      stop('provide the same cell type parameter dimensions')
    } else {
      cell_type_data <- c()
      cell_type_other <- c()
    }
  } else {
    # This will generate cell-types and then automatically simulate
    # differential cell-type proportion
    cell_type_info <- sampleCellMatrix(n, cell_type_mean_df, cell_type_sd_df)
    cell_type_data <- cell_type_info[[1]]
    cell_type_other <- cell_type_info[[2]]
  }

  # Merge Features
  all_features <- cbind(group_data, rand_params, cont_params, pres_params,
                        cell_type_data)
  other_features <- cbind(group_info, cont_other_params, cell_type_other)
  
  # Normalize data by zero-one-normalization
  if (zero_one_normalize) {
    zeroonenorm <- function(x){(x - min(x)) / (max(x) - min(x))}
    all_features <- apply(all_features, MARGIN = 2, FUN = zeroonenorm)
  }

  return_list <- list(features = all_features, other = other_features)
  return(return_list)
}
