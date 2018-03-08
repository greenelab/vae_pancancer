# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/util/simulate_expression.R
#
# Functions to facilitate gene expression data simulation
#
# Usage: import only 
# source("scripts/util/simulate_expression.R")

library(WGCNA)
library(ggcorrplot)
library(gplots)

allowWGCNAThreads()

n <- 2000  # There will be two sets (total n = 4000)
num_sets <- 4
num_genes <- 1000
num_modules <- 5
background_noise <- 0.1
min_cor <- 0.4
max_cor <- 0.9

sample_set_A <- c(1, -1, -3)
sample_set_B <- c(-2, 4, 2)

mod_prop <- c(0.25, 0.2, 0.15, 0.1, 0.05, 0.25)

leave_out <- rep(FALSE, num_modules)
leave_out_b <- c(rep(FALSE, num_modules / 2), TRUE, rep(FALSE, (num_modules / 2)))

# Sample eigen gene matrix
eigen_gene_samples <- rbind(
  matrix(rnorm(num_modules * n / 2, mean = sample_set_A, sd = 1),
         n / 2, num_modules, byrow = TRUE),
  matrix(rnorm(num_modules * n / 2, mean = sample_set_B, sd = 1),
         n / 2, num_modules, byrow = TRUE)
)
ggcorrplot(cor(t(eigen_gene_samples)), hc.order = TRUE, outline.color = 'white')

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
                       leaveOut = leave_out_b,
                       propNegativeCor = 0.1,
                       backgroundNoise = background_noise)


# Group labels
sample_labels <- rep(sort(rep(LETTERS[1:num_sets], n / 4)), 2)
gene_labels <- x_1$allLabels

# Combine Matrix
x_matrix <- tibble::as_data_frame(rbind(x_1$datExpr,
                                        x_2$datExpr))
colnames(x_matrix) <- gsub('[.]', '_', colnames(x_matrix))
x_matrix$groups <- sample_labels
x_matrix <- rbind(gene_labels, x_matrix)

gene_corr <- round(cor(x_matrix[2:nrow(x_matrix), 1:num_genes]), 4)
ggcorrplot(gene_corr, hc.order = TRUE, outline.col = "white")

heatmap.2(as.matrix(x_matrix[2:nrow(x_matrix), 1:num_genes]), trace = "none",
          RowSideColors = c(rep('blue', n / 2),
                            rep('red', n / 2),
                            rep('green', n / 2),
                            rep('orange', n/ 2)),
          hclustfun = function(x) hclust(x, method = 'centroid'),
          distfun = function(x) dist(x, method = 'euclidean'),
          dendrogram = "row", Rowv = TRUE, Colv = TRUE)

readr::write_tsv(x_matrix, 'data/simulation/wgcna_test_v7.tsv')
