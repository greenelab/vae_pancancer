# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/simulation_experiment.R
#
# Generate reproducible random data for simulation experiment
#
# Usage: Rscript scripts/simulation_viz_tests.R
#
# Output:
# Several dataframes that will be used in the compression algorithm evaluations

source("scripts/util/simulate_expression.R")

n <- 50
r <- 0
b <- 0

group_mean <- as.matrix(cbind(c(1, 2, 3), c(4, 1, 0)))
group_sd <- as.matrix(cbind(c(1, 1, 1), c(1, 1, 1)))

x_squared <- function(x) { x ** 2 + 3}
c_list <- list(x_squared)

cell_mean <- as.matrix(cbind(c(1, 4), c(5, 1)))
cell_sd <- as.matrix(cbind(c(1, 1), c(1, 1)))

test_output <- getSimulatedExpression(n = n, mean_df = group_mean,
                                      sd_df = group_sd, r = r,
                                      func_list = c_list, b = b,
                                      seed = 123,
                                      cell_type_mean_df = cell_mean,
                                      cell_type_sd_df = cell_sd)

test_df <- cbind(test_output[[1]], test_output[[2]])
colnames(test_df) <- c('group_1', 'group_2', 'rand_1', 'rand_2', 'continuous_1',
                       'rand_presence_1', 'rand_presence_2', 'cell_type_1',
                       'cell_type_2', 'group_info', 'continous_x_val',
                       'cell_type_prop')


write.table(test_df, 'data/test_simulated_gene_expression.tsv', sep='\t')
test_df <- as.data.frame(test_df)
test_df$group_1 <- as.numeric(paste(test_df$group_1))
test_df$group_2 <- as.numeric(paste(test_df$group_2))
test_df$cell_type_1 <- as.numeric(paste(test_df$cell_type_1))
test_df$cell_type_2 <- as.numeric(paste(test_df$cell_type_2))
test_df$cell_type_prop <- as.numeric(paste(test_df$cell_type_prop))


library(ggplot2)
ggplot(test_df, aes(x = cell_type_2, y = cell_type_prop)) + geom_point(aes(color = group_info))


# library(ggplot2)
# set.seed(123)
# 
# n_samples <- 1500
# 
# a <- rnorm(n_samples / 3, mean = 4, sd = 1)
# b <- rnorm(n_samples / 3, mean = 2, sd = 1)
# 
# group_1 <- data.frame(cbind(a, b))
# group_1$group = 'A'
# 
# c <- rnorm(n_samples / 3, mean = 0, sd = 1)
# d <- rnorm(n_samples / 3, mean = -1, sd = 1)
# 
# group_2 <- data.frame(cbind(b, c))
# group_2$group = 'B'
# colnames(group_2)[1:2] <- c('a', 'b')
# e <- rnorm(n_samples / 3, mean = -1, sd = 1)
# f <- rnorm(n_samples / 3, mean = 2, sd = 1)
# 
# group_3 <- data.frame(cbind(e, f))
# group_3$group = 'C'
# 
# colnames(group_3)[1:2] <- c('a', 'b')
# 
# test <- rbind(group_1, group_2, group_3)
# 
# ggplot(test, aes(x=a, y=b)) + geom_point(aes(color = group))
# 
# 
# rand_unif <- runif(n_samples, min=0, max=1)
# rand_unif <- data.frame(cbind(rand_unif, runif(n_samples, min=0, max=1)))
# colnames(rand_unif) <- c('rand_a', 'rand_b')
# 
# sim_matrix <- cbind(test[, c(1,2)], rand_unif)
# 
# # Presence or absence of a feature
# rand_presence <- rnorm(n_samples, mean=3, sd=0.5)
# rand_zerone <- sample(c(0,1), n_samples, replace=TRUE)
# rand_presence <- rand_presence * rand_zerone
# 
# sim_matrix <- cbind(test[, c(1,2)], rand_unif, rand_presence)
# 
# 
# # Continuous Function
# continuous_rand_x <- runif(n_samples, min=-1, max=1)
# continuous_rand_y <- continuous_rand_x ** 2 + 3
# 
# #cont_rand <- cbind(continuous_rand_x, continuous_rand_y)
# # sim_matrix <- cbind(sim_matrix, cont_rand)
# 
# sim_matrix <- cbind(sim_matrix, continuous_rand_y)
# 
# range01 <- function(x){(x-min(x))/(max(x)-min(x))}
# sim_matrix <- range01(sim_matrix)
# 
# sim_matrix_out <- cbind(sim_matrix, c(group_1$group, group_2$group,
#                                       group_3$group))
# colnames(sim_matrix_out)[ncol(sim_matrix_out)] <- 'group'
# write.table(sim_matrix_out, 'data/simulated_expression_1500.tsv',
#             sep='\t', row.names = FALSE)
# 
# ggplot(sim_matrix_out, aes(x=continuous_rand_x, y=continuous_rand_y)) +
#   geom_point(aes(color = group, alpha=0.5))
# 
# ggplot(sim_matrix, aes(x=a, y=rand_presence)) +
#   geom_point(aes(color = group, alpha=0.5))
# 
# 
# 
vae_sim <- readr::read_tsv('data/test_simulation_vae_compress_large.tsv')
colnames(vae_sim) <- paste('vae', colnames(vae_sim), sep='_')
# 
# pca_sim <- readr::read_tsv('data/test_simulation_pca_compress.tsv')
# 
all_data <- as.data.frame(cbind(test_df, vae_sim))
all_data$group_1 <- as.numeric(paste(all_data$group_1))
all_data$group_2 <- as.numeric(paste(all_data$group_2))
all_data$cell_type_1 <- as.numeric(paste(all_data$cell_type_1))
all_data$cell_type_2 <- as.numeric(paste(all_data$cell_type_2))
all_data$rand_presence_1 <- as.numeric(paste(all_data$rand_presence_1))
all_data$rand_presence_2 <- as.numeric(paste(all_data$rand_presence_2))
all_data$cell_type_prop <- as.numeric(paste(all_data$cell_type_prop))


ggplot(all_data, aes(x = vae_4, y = group_1)) +
  geom_point(aes(color = vae_group_info), alpha=0.5)
# 
# ggplot(all_data, aes(x = pca_4, y = pca_3)) +
#   geom_point(aes(color = group), alpha=0.5)
# 
# ggplot(all_data, aes(x = pca_2, y = b)) +
#   geom_point(aes(color = group), alpha=0.5)
# 
#  ggplot(all_data, aes(x = continuous_rand_x, y = continuous_rand_y)) +
#   geom_point(aes(color = vae_group, size=vae_3), alpha=0.5)
# 
# ggplot(all_data, aes(x = vae_3, y = continuous_rand_y)) +
#   geom_point(aes(color = vae_group), alpha=0.5)
# 
# 
# # This one may be erroneous? Requires more interpretaion
# ggplot(all_data, aes(x = vae_3, y = vae_4)) +
#   geom_point(aes(color = vae_group), alpha=0.5)
# 
# # 
# ggplot(all_data, aes(x=vae_5, y=continuous_rand_y)) +
#   geom_point(aes(color = group, alpha=0.5))
# 
# 
# sim_weights <- readr::read_tsv('data/test_simulation_weights_pca.tsv')
sim_weights <- readr::read_tsv('data/test_simulation_weights_large.tsv')
# 
sim_weight_df <- t(data.frame(sim_weights))
colnames(sim_weight_df) <- colnames(vae_sim)[1:4]
rownames(sim_weight_df) <- c('group_a', 'group_b', 'random_a', 'random_b',
                             'transform_c', 'random_binary_a', 'random_binary_b',
                             'cell_type_a', 'cell_type_b')
weight_melted <- reshape2::melt(sim_weight_df)
colnames(weight_melted) <- c('variable', 'vae_feature', 'weight')
# 
ggplot(weight_melted, aes(x=vae_feature)) + geom_bar(aes(y=weight), stat='identity') +
  facet_wrap(~variable)
# 
# test_mean = as.matrix(cbind(c(1, 2, 3), c(3, 2, 1)))
# test_sd = as.matrix(cbind(c(1, 1, 1), c(1, 1, 1)))

