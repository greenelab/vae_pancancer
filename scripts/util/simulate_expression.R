# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/util/simulate_expression.R
#
# Functions to facilitate gene expression data simulation
#
# Usage: run once to simulate 25 datasets with different noise and sample size
#   Rscript scripts/util/simulate_expression.R

library(WGCNA)
library(ggcorrplot)
library(gplots)

allowWGCNAThreads()

set.seed(1234)

simulateExpression <- function(n, num_sets, num_genes, num_modules,
                               background_noise, min_cor, max_cor,
                               sample_set_A, sample_set_B, mod_prop,
                               leave_out_A, leave_out_B) {
  # Output a simulated gene expression matrix using WGCNA
  #
  # Arguments:
  # n - The number of samples to simulate
  # num_sets - The number of simulated "groups" of samples
  # num_genes - The number of genes to simulate
  # num_modules - The number of distinct simulated "groups" of genes
  # background_noise - N(0, sd = background_noise) added to all values
  # min_cor - The minimum correlation of genes in modules to core module
  # max_cor - The maximum correlation of genes in modules to core module
  # sample_set_A - character vector of sample group A centroids
  # sample_set_B - character vector of sample group B centroids
  # mod_prop - character vector (must sum to 1) of gene module proportions
  # leave_out_A - character vector of booleans indicating gene module presence
  # leave_out_B - adds option to leave out modules for different group
  #
  # Output:
  # An n x num_genes simulated gene expression matrix with predefined groups
  # of samples and gene modules

  # Sample eigen gene matrix
  eigen_gene_samples <- rbind(
    matrix(rnorm(num_modules * n / 2, mean = sample_set_A, sd = 1),
           n / 2, num_modules, byrow = TRUE),
    matrix(rnorm(num_modules * n / 2, mean = sample_set_B, sd = 1),
           n / 2, num_modules, byrow = TRUE)
  )

  # Simulate Expression
  # the two matrices differ based on the leave out argument. This is to enable
  # an assessment of latent space arithmetic to test if an algorithm can isolate
  # the "essence" of the left out gene module through subtraction.
  x_1 <- WGCNA::simulateDatExpr(eigengenes = eigen_gene_samples,
                                nGenes = num_genes,
                                modProportions = mod_prop,
                                minCor = min_cor,
                                maxCor = max_cor,
                                leaveOut = leave_out_A,
                                propNegativeCor = 0.1,
                                backgroundNoise = background_noise)

  x_2 <- WGCNA::simulateDatExpr(eigengenes = eigen_gene_samples,
                                nGenes = num_genes,
                                modProportions = mod_prop,
                                minCor = min_cor,
                                maxCor = max_cor,
                                leaveOut = leave_out_B,
                                propNegativeCor = 0.1,
                                backgroundNoise = background_noise)

  # Group labels
  sample_labels <- sort(rep(rep(LETTERS[1:num_sets], n / num_sets), 2))
  num_remainder <- (n * 2) - length(sample_labels)

  if (num_remainder > 0) {
    sample_labels <- sort(c(sample_labels, LETTERS[1:num_remainder]))
  }

  gene_labels <- x_1$allLabels

  # Combine Matrix
  x_matrix <- tibble::as_data_frame(rbind(x_1$datExpr,
                                          x_2$datExpr))
  colnames(x_matrix) <- gsub('[.]', '_', colnames(x_matrix))
  x_matrix$groups <- sample_labels
  x_matrix <- rbind(gene_labels, x_matrix)
  return(x_matrix)
}

# Vary sample size and amount of background noise
ns <- c(250, 500, 1000, 2000, 4000)
background_noises <- c(0, 0.1, 0.5, 1, 3)
genes <- c(500, 1000)

# Other constants
num_sets <- 4
num_modules <- 5

min_cor <- 0.4
max_cor <- 0.9

sample_set_A <- c(1, -1, -3)
sample_set_B <- c(-2, 4, 2)

mod_prop <- c(0.25, 0.2, 0.15, 0.1, 0.05, 0.25)

leave_out_A <- rep(FALSE, num_modules)
leave_out_B <- c(rep(FALSE, num_modules / 2), TRUE,
                 rep(FALSE, (num_modules / 2)))

for (n in ns) {
  for (noise in background_noises) {
    for (g in genes) {
      out_file <- paste0("sim_data_samplesize_", n * 2, "_noise_", noise,
                         "_genes_", g, ".tsv")
      out_file <- file.path("data", "simulation", out_file)
      x <- simulateExpression(n = n,
                              num_genes = g,
                              background_noise = noise,
                              num_sets = num_sets,
                              num_modules = num_modules,
                              min_cor = min_cor,
                              max_cor = max_cor,
                              sample_set_A = sample_set_A,
                              sample_set_B = sample_set_B,
                              mod_prop = mod_prop,
                              leave_out_A = leave_out_A,
                              leave_out_B = leave_out_B)

      readr::write_tsv(x, out_file)
    }
  }
}
