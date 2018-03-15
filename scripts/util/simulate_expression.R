# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/util/simulate_expression.R
#
# Functions to facilitate gene expression data simulation
#
# Usage: run once to simulate 25 datasets with different noise and sample size
#   Rscript scripts/util/simulate_expression

library(WGCNA)
library(ggcorrplot)
library(gplots)

allowWGCNAThreads()

set.seed(1234)

simulateExpression <- function(n, num_sets, num_genes, num_modules,
                               background_noise, min_cor, max_cor,
                               sample_set_A, sample_set_B, mod_prop,
                               leave_out, leave_out_other) {
  # Output a simulated gene expression matrix using WGCNA
  # Sample eigen gene matrix
  eigen_gene_samples <- rbind(
    matrix(rnorm(num_modules * n / 2, mean = sample_set_A, sd = 1),
           n / 2, num_modules, byrow = TRUE),
    matrix(rnorm(num_modules * n / 2, mean = sample_set_B, sd = 1),
           n / 2, num_modules, byrow = TRUE)
  )
  
  # Simulate Expression
  x_1 <- simulateDatExpr(eigengenes = eigen_gene_samples,
                         nGenes = num_genes,
                         modProportions = mod_prop,
                         minCor = min_cor,
                         maxCor = max_cor,
                         propNegativeCor = 0.1,
                         backgroundNoise = background_noise)
  
  x_2 <- simulateDatExpr(eigengenes = eigen_gene_samples,
                         nGenes = num_genes,
                         modProportions = mod_prop,
                         minCor = min_cor,
                         maxCor = max_cor,
                         leaveOut = leave_out_other,
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

# Other constants
num_sets <- 4
num_genes <- 1000
num_modules <- 5

min_cor <- 0.4
max_cor <- 0.9

sample_set_A <- c(1, -1, -3)
sample_set_B <- c(-2, 4, 2)

mod_prop <- c(0.25, 0.2, 0.15, 0.1, 0.05, 0.25)

leave_out <- rep(FALSE, num_modules)
leave_out_other <- c(rep(FALSE, num_modules / 2), TRUE,
                     rep(FALSE, (num_modules / 2)))

for (n in ns) {
  for (noise in background_noises) {
    out_file <- paste0("sim_data_samplesize_", n * 2, "_noise_", noise, ".tsv")
    out_file <- file.path("data", "simulation", out_file)
    x <- simulateExpression(n = n,
                            num_sets = num_sets,
                            num_genes = num_genes,
                            num_modules = num_modules,
                            background_noise = noise,
                            min_cor = min_cor,
                            max_cor = max_cor,
                            sample_set_A = sample_set_A,
                            sample_set_B = sample_set_B,
                            mod_prop = mod_prop,
                            leave_out = leave_out,
                            leave_out_other = leave_out_other)
    
    readr::write_tsv(x, out_file)
  }
}
