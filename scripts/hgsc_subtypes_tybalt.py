
# coding: utf-8

# # Interpolating high grade serous ovarian cancer subtypes in VAE space
# 
# Recent applications of generative models (GANs and VAEs) in image processing has demonstrated the remarkable ability of the latent dimensions to capture a meaningful manifold representation of the input space. Here, we assess if the VAE learns a latent space that can be mathematically manipulated to reveal insight into the gene expression activation patterns of high grade serous ovarian cancer (HGSC) subtypes.
# 
# Several previous studies have reported the presence of four gene expression based HGSC subtypes. However, we recently [published a paper](https://doi.org/10.1534/g3.116.033514) that revealed the inconsistency of subtype assignments across populations. We observed repeatable structure in the data transitioning between setting clustering algorithms to find different solutions. For instance, when setting algorithms to find 2 subtypes, the mesenchymal and immunoreactive and the proliferative and differentiated subtype consistently collapsed together. These observations may suggest that the subtypes exist on a gene expression continuum of differential activation patterns, and may only be artificially associated with "subtypes". Here, we test if the VAE can help to identify some differential patterns of expression.

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use("seaborn-notebook")


# In[3]:

sns.set(style="white", color_codes=True)
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":15, "axes.labelsize":20,
                             'xtick.labelsize':14, 'ytick.labelsize':14})   


# In[4]:

# Load Models
decoder_model_file = os.path.join('models', 'decoder_onehidden_vae.hdf5')
decoder = keras.models.load_model(decoder_model_file)


# In[5]:

rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
rnaseq_df.shape


# In[6]:

ov_file = os.path.join('data', 'ov_subtype_info.tsv')
ov_df = pd.read_table(ov_file, index_col=0)
ov_df.head(2)


# In[7]:

encoded_file = os.path.join('data', "encoded_rnaseq_onehidden_warmup_batchnorm.tsv")
encoded_df = pd.read_table(encoded_file, index_col=0)
encoded_df.shape


# In[8]:

# Subset and merge the HGSC subtype info with the latent space feature activations
ov_samples = list(set(encoded_df.index) & (set(ov_df.index)))

ov_encoded = encoded_df.loc[ov_samples, ]
ov_encoded_subtype = pd.merge(ov_df.loc[:, ['SUBTYPE', 'SILHOUETTE WIDTH']], ov_encoded,
                              how='right', left_index=True, right_index=True)
ov_encoded_subtype = ov_encoded_subtype.assign(subtype_color = ov_encoded_subtype['SUBTYPE'])

ov_subtype_color_dict = {'Differentiated': 'purple',
                         'Immunoreactive': 'green',
                         'Mesenchymal': 'blue',
                         'Proliferative': 'red'}
ov_encoded_subtype = ov_encoded_subtype.replace({'subtype_color': ov_subtype_color_dict})

print(ov_encoded_subtype.shape)
ov_encoded_subtype.head(2)


# In[9]:

# Get the HGSC mean feature activation
ov_mean_subtypes = ov_encoded_subtype.groupby('SUBTYPE').mean()
ov_mean_subtypes


# ## HGSC Subtype Math
# 
# Because of the relationship observed in the consistent clustering solutions, perform the following subtractions
# 
# 1. Immunoreactive - Mesenchymal
# 2. Differentiated - Proliferative
# 
# The goal is to observe the features with the largest difference between the aformentioned comparisons. The differences should be in absolute directions

# In[10]:

mes_mean_vector = ov_mean_subtypes.loc['Mesenchymal', [str(x) for x in range(1, 101)]]
imm_mean_vector = ov_mean_subtypes.loc['Immunoreactive', [str(x) for x in range(1, 101)]]
pro_mean_vector = ov_mean_subtypes.loc['Proliferative', [str(x) for x in range(1, 101)]]
dif_mean_vector = ov_mean_subtypes.loc['Differentiated', [str(x) for x in range(1, 101)]]


# In[11]:

high_immuno = (imm_mean_vector - mes_mean_vector).sort_values(ascending=False).head(2)
high_mesenc = (imm_mean_vector - mes_mean_vector).sort_values(ascending=False).tail(2)

print("Features with large differences: Immuno high, Mesenchymal low")
print(high_immuno)
print("Features with large differences: Mesenchymal high, Immuno low")
print(high_mesenc)


# In[12]:

# Select to visualize encoding 56 because it has high immuno and low everything else
ov_mean_subtypes.loc[:, ['87', '77', '56']]


# In[13]:

# Obtain the decoder weights
weights = []
for l in decoder.layers:
    weights.append(l.get_weights())
    
weight_layer = pd.DataFrame(weights[1][0], columns=rnaseq_df.columns)
weight_layer.head(2)


# In[14]:

# Get the high weight genes describing the immunoreactive subtype
i_87_genes = weight_layer.loc[87, :].sort_values(ascending = False)
i_77_genes = weight_layer.loc[77, :].sort_values(ascending = False)
i_56_genes = weight_layer.loc[56, :].sort_values(ascending = False)


# In[15]:

# File names for output genes
node87pos_file = os.path.join('results', 'hgsc_node87genes_pos.tsv')
node87neg_file = os.path.join('results', 'hgsc_node87genes_neg.tsv')
node77pos_file = os.path.join('results', 'hgsc_node77genes_pos.tsv')
node77neg_file = os.path.join('results', 'hgsc_node77genes_neg.tsv')
node56pos_file = os.path.join('results', 'hgsc_node56genes_pos.tsv')
node56neg_file = os.path.join('results', 'hgsc_node56genes_neg.tsv')

high_std = 2

# Get high weight genes
node87pos_df = (i_87_genes[i_87_genes > i_87_genes.std() * high_std])
node87neg_df = (i_87_genes[i_87_genes < -1 * (i_87_genes.std() * high_std)])

node77pos_df = (i_77_genes[i_77_genes > i_77_genes.std() * high_std])
node77neg_df = (i_77_genes[i_77_genes < -1 * (i_77_genes.std() * high_std)])

node56pos_df = (i_56_genes[i_56_genes > i_56_genes.std() * high_std])
node56neg_df = (i_56_genes[i_56_genes < -1 * (i_56_genes.std() * high_std)])


# In[16]:

# Process and write out tsv files
col_names = ['genes', 'weight']

node87pos_df = pd.DataFrame(node87pos_df).reset_index()
node87pos_df.columns = col_names
node87neg_df = pd.DataFrame(node87neg_df).reset_index()
node87neg_df.columns = col_names

node77pos_df = pd.DataFrame(node77pos_df).reset_index()
node77pos_df.columns = col_names
node77neg_df = pd.DataFrame(node77neg_df).reset_index()
node77neg_df.columns = col_names

node56pos_df = pd.DataFrame(node56pos_df).reset_index()
node56pos_df.columns = col_names
node56neg_df = pd.DataFrame(node56neg_df).reset_index()
node56neg_df.columns = col_names

# Write to file
node87pos_df.to_csv(node87pos_file, index=False, sep='\t')
node87neg_df.to_csv(node87neg_file, index=False, sep='\t')

node77pos_df.to_csv(node77pos_file, index=False, sep='\t')
node77neg_df.to_csv(node77neg_file, index=False, sep='\t')

node56pos_df.to_csv(node56pos_file, index=False, sep='\t')
node56neg_df.to_csv(node56neg_file, index=False, sep='\t')


# In[17]:

high_differ = (dif_mean_vector - pro_mean_vector).sort_values(ascending=False).head(2)
high_prolif = (dif_mean_vector - pro_mean_vector).sort_values(ascending=False).tail(2)

print("Features with large differences: Differentiated high, Proliferative low")
print(high_differ)
print("Features with large differences: Proliferative high, Differentiated low")
print(high_prolif)


# In[18]:

# Get the high weight genes describing the differentiated subtype
d_79_genes = weight_layer.loc[79, :].sort_values(ascending = False)
d_38_genes = weight_layer.loc[38, :].sort_values(ascending = False)


# In[19]:

# File names for output genes
node79pos_file = os.path.join('results', 'hgsc_node79genes_diffpro_pos.tsv')
node79neg_file = os.path.join('results', 'hgsc_node79genes_diffpro_neg.tsv')
node38pos_file = os.path.join('results', 'hgsc_node38genes_diffpro_pos.tsv')
node38neg_file = os.path.join('results', 'hgsc_node38genes_diffpro_neg.tsv')

# Get high weight genes
node79pos_df = (d_79_genes[d_79_genes > d_79_genes.std() * high_std])
node79neg_df = (d_79_genes[d_79_genes < -1 * (d_79_genes.std() * high_std)])

node38pos_df = (d_38_genes[d_38_genes > d_38_genes.std() * high_std])
node38neg_df = (d_38_genes[d_38_genes < -1 * (d_38_genes.std() * high_std)])


# In[20]:

# Process and write out tsv files
node79pos_df = pd.DataFrame(node79pos_df).reset_index()
node79pos_df.columns = col_names
node79neg_df = pd.DataFrame(node79neg_df).reset_index()
node79neg_df.columns = col_names

node38pos_df = pd.DataFrame(node38pos_df).reset_index()
node38pos_df.columns = col_names
node38neg_df = pd.DataFrame(node38neg_df).reset_index()
node38neg_df.columns = col_names

node79pos_df.to_csv(node79pos_file, index=False, sep='\t')
node79neg_df.to_csv(node79neg_file, index=False, sep='\t')

node38pos_df.to_csv(node38pos_file, index=False, sep='\t')
node38neg_df.to_csv(node38neg_file, index=False, sep='\t')


# In[21]:

# Node 87 has high mesenchymal, low immunoreactive
node87_file = os.path.join('figures', 'node87_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '87', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 87')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node87_file)


# In[22]:

# Node 77 has high immunoreactive, low mesenchymal
node77_file = os.path.join('figures', 'node77_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '77', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 77')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node77_file)


# In[23]:

# Node 56 has high immunoreactive, low mesenchymal (but also low proliferative)
node56_file = os.path.join('figures', 'node56_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '56', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 56')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node56_file)


# In[24]:

# Node 79 has high proliferative, low differentiated
node79_file = os.path.join('figures', 'node79_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '79', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 79')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node79_file)


# In[25]:

# Node 38 has high differentiated, low proliferative
node38_file = os.path.join('figures', 'node38_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '38', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 38')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node38_file)

