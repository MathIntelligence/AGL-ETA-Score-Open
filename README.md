# AGL-ETA-Score-Open
To compute features for the AGL-EAT-Score model, use the provided code. The main source code is located in the ```src``` folder. You can generate features for a given protein-ligand dataset using the ```get_agl_eat_features.py``` script.
# Required Packages
- NumPy
- SciPy
- Pandas
- BioPandas
- RDKit
# Example
To generate features for the PDBbind v2016 general set for the AGL-EAT model with an exponential kernel type and parameters $\kappa = 16.5$ and $\tau = 1.5$, located at index 112 of the ```kernels.csv``` file in the utils folder, follow these steps. Assume the dataset structures are in the ```../PDBbind_v2016_general_Set``` directory and the features should be saved in the ```../Features``` directory.
```shell
 # Generate the AGL extended atom type (SYBYL) features
 python get_agl_eat_features.py -k 112 -c 12 -m Adjacency -f '../csv_data_file/PDBbindv2016_GeneralSet.csv' -dd '../PDBbind_v2016_general_set' -fd '../Features'
```
