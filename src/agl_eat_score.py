#!/usr/bin/env python
# Author: Masud Rana <masud.rana@uky.edu>
# Last modified: Apr 7, 2023

import numpy as np
import pandas as pd
#import scipy.sparse as sp
import scipy.linalg as spla
import os
from os import listdir
#from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
#from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
import sys
import time

class KernelFunction:

    def __init__(self, kernel_type='exponential_kernel',
                 kappa=2.0, tau=1.0):
        self.kernel_type = kernel_type
        self.kappa = kappa
        self.tau = tau

        self.kernel_function = self.build_kernel_function(kernel_type)

    def build_kernel_function(self, kernel_type):
        if kernel_type[0] in ['E', 'e']:
            return self.exponential_kernel
        elif kernel_type[0] in ['L', 'l']:
            return self.lorentz_kernel

    def exponential_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii

        return np.exp(-(d/eta)**self.kappa)

    def lorentz_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii

        return 1/(1+(d/eta)**self.kappa)

class AGL_EAT:

    protein_atom_types_df = pd.read_csv(
        '../utils/protein_atom_types.csv')

    ligand_atom_types_df = pd.read_csv(
        '../utils/ligand_SYBYL_atom_types.csv')

    protein_atom_types = protein_atom_types_df['AtomType'].tolist()
    protein_atom_radii = protein_atom_types_df['Radius'].tolist()

    ligand_atom_types = ligand_atom_types_df['AtomType'].tolist()

    ligand_atom_radii = ligand_atom_types_df['Radius'].tolist()
    

    protein_ligand_atom_types = [
        i[0]+"-"+i[1] for i in product(protein_atom_types, ligand_atom_types)]

    sigma = np.mean([np.std(protein_atom_radii),np.std(ligand_atom_radii)])

    def __init__(self, Kernel, cutoff, matrix_type='Laplace'):

        self.Kernel = Kernel
        self.cutoff = cutoff
        self.matrix_type = matrix_type

        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()

    def get_pairwise_atom_type_radii(self):

        protein_atom_radii_dict = {a: r for (a, r) in zip(
            self.protein_atom_types, self.protein_atom_radii)}

        ligand_atom_radii_dict = {a: r for (a, r) in zip(
            self.ligand_atom_types, self.ligand_atom_radii)}

        pairwise_atom_type_radii = {i[0]+"-"+i[1]: protein_atom_radii_dict[i[0]] +
                                    ligand_atom_radii_dict[i[1]] for i in product(self.protein_atom_types, self.ligand_atom_types)}

        return pairwise_atom_type_radii

    def mol2_to_df(self, mol2_file):
        df_mol2_all = PandasMol2().read_mol2(mol2_file).df
        df_mol2 = df_mol2_all[df_mol2_all['atom_type'].isin(self.ligand_atom_types)]
        df = pd.DataFrame(data={'ATOM_INDEX': df_mol2['atom_id'],
                                'ATOM_ELEMENT': df_mol2['atom_type'],
                                'X': df_mol2['x'],
                                'Y': df_mol2['y'],
                                'Z': df_mol2['z']})

        if len(set(df["ATOM_ELEMENT"]) - set(self.ligand_atom_types)) > 0:
            print(
                "WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
        return(df)

    def pdb_to_df(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)
        ppdb_all_df = ppdb.df['ATOM']
        ppdb_df = ppdb_all_df[ppdb_all_df['atom_name'].isin(
            self.protein_atom_types)]
        atom_index = ppdb_df['atom_number']
        atom_element = ppdb_df['atom_name']
        x, y, z = ppdb_df['x_coord'], ppdb_df['y_coord'], ppdb_df['z_coord']
        df = pd.DataFrame.from_dict({'ATOM_INDEX': atom_index, 'ATOM_ELEMENT': atom_element,
                                     'X': x, 'Y': y, 'Z': z})

        return df
        

    def get_mwcg(self, protein_file, ligand_file):
        protein = self.pdb_to_df(protein_file)
        ligand = self.mol2_to_df(ligand_file)

        # select protein atoms in a cubic with a size of cutoff from ligand
        for i in ["X", "Y", "Z"]:
            protein = protein[protein[i] < float(ligand[i].max())+self.cutoff]
            protein = protein[protein[i] > float(ligand[i].min())-self.cutoff]
        
        
        prot_grouped = protein.groupby("ATOM_ELEMENT")
        prot_group_df = prot_grouped.size().to_frame(name='COUNTS')
        
        lig_grouped = ligand.groupby("ATOM_ELEMENT")
        lig_group_df = lig_grouped.size().to_frame(name='COUNTS')
        
        atom_pairs = list(
            product(protein["ATOM_ELEMENT"], ligand["ATOM_ELEMENT"]))
        atom_pairs = [x[0]+"-"+x[1] for x in atom_pairs]
        pairwise_radii = [self.pairwise_atom_type_radii[x]
                          for x in atom_pairs]
        pairwise_radii = np.asarray(pairwise_radii)

        pairwise_mwcg = pd.DataFrame(atom_pairs, columns=["ATOM_PAIR"])
        distances = cdist(protein[["X", "Y", "Z"]],
                          ligand[["X", "Y", "Z"]], metric="euclidean")
        pairwise_radii = pairwise_radii.reshape(
            distances.shape[0], distances.shape[1])
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        distances = distances.ravel()
        mwcg_distances = mwcg_distances.ravel()
        mwcg_distances = pd.DataFrame(
            data={"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances], axis=1)

        return prot_group_df, lig_group_df, pairwise_mwcg


    def get_adjacency_matrix(self, mwcg_mat, p_size, l_size):
        N = p_size + l_size
        
        adj_mat = np.zeros((N,N))
        adj_mat[:p_size, p_size:N] = -mwcg_mat
        adj_mat[p_size:N, :p_size] = -mwcg_mat.transpose()
        
        return adj_mat
    
    def get_laplacian_matrix(self, mwcg_mat, p_size, l_size):
        adj_mat = self.get_adjacency_matrix(mwcg_mat, p_size, l_size)
        
        lap_mat = adj_mat
        diag_entry = adj_mat.sum(axis=1)
        
        np.fill_diagonal(lap_mat, -diag_entry)
        
        return lap_mat
                
    def graph_features(self, graph_matrix):

        ews = spla.eigh(graph_matrix, eigvals_only = True)

        ews = np.round(ews, 5)
        
        if self.matrix_type[0] in ['L' or 'l']:
            assert (ews >= 0).all(), "Laplacian matrix's eigenvalues !< 0"

        if np.sum(np.abs(ews)) == 0:
            agl_features = np.zeros((9, ))
            return agl_features

        positive_eigen_values = list(filter(lambda v: v > 0, ews))
        agl_features = [
                np.sum(positive_eigen_values),
                np.min(positive_eigen_values),  # Fiedler value for Laplacian matrices
                np.max(positive_eigen_values),
                np.mean(positive_eigen_values),
                np.median(positive_eigen_values),
                np.std(positive_eigen_values),
                np.var(positive_eigen_values),
                len(positive_eigen_values),
                np.sum(np.power(positive_eigen_values, 2))

        ]

        return np.round(agl_features, 5)   
        
    def get_agl_score(self, protein_file, ligand_file):
        p_size_df, l_size_df, pairwise_mwcg = self.get_mwcg(protein_file, ligand_file)
        
        _atom_pair = pairwise_mwcg['ATOM_PAIR'].values
        _atom_pair = np.unique(_atom_pair)
        
        all_features = np.zeros((len(_atom_pair),9))
        all_features_df = pairwise_mwcg.groupby('ATOM_PAIR')
        all_features_df = all_features_df.size().to_frame(name='COUNTS')
        
        for _index, pair in enumerate(_atom_pair):
            pairwise_mat = pairwise_mwcg[pairwise_mwcg['ATOM_PAIR']==pair]
            pair_mat = pairwise_mat['MWCG_DISTANCE'].values
            
            p_atom = pair.split('-')[0]
            l_atom = pair.split('-')[1]

            p_size = p_size_df.loc[p_atom].values[0]
            l_size = l_size_df.loc[l_atom].values[0]
            
            cutoff_index = np.where(pairwise_mat['DISTANCE']>self.cutoff)
            pair_mat[cutoff_index] = 0

            covalent_rstriction = self.pairwise_atom_type_radii[pair] + self.sigma
            covalent_index = np.where(pairwise_mat['DISTANCE']<covalent_rstriction)
            pair_mat[covalent_index] = 0

            pair_mat = pair_mat.reshape(p_size, l_size)
            
            if self.matrix_type[0] in ['A', 'a']:
                graph_mat = self.get_adjacency_matrix(pair_mat, p_size, l_size)
                
            elif self.matrix_type[0] in ['L', 'l']:
                graph_mat = self.get_laplacian_matrix(pair_mat, p_size, l_size)
                
            pair_feature = self.graph_features(graph_mat)
            all_features[_index,:] = pair_feature
            

        features_df = pd.DataFrame(all_features, 
                                       columns=['SUM', 'MIN', 'MAX', 'MEAN', 'MEDIAN','STD','VAR','NUM_EWS','SUM_SQUARED'],
                                      index=_atom_pair)

        all_features_df = all_features_df.join(features_df)

        features_name = ['COUNTS', 'SUM', 'MIN', 'MAX', 'MEAN', 'MEDIAN','STD','VAR','NUM_EWS','SUM_SQUARED']
        mwcg_columns = {'ATOM_PAIR': self.protein_ligand_atom_types}
        for _f in features_name:
            mwcg_columns[_f] = np.zeros(len(self.protein_ligand_atom_types))
        agl_score = pd.DataFrame(data=mwcg_columns)
        agl_score = agl_score.set_index('ATOM_PAIR').add(
            all_features_df, fill_value=0).reindex(self.protein_ligand_atom_types).reset_index()
        
        return agl_score
                

def do_example():
    protein_file = '../example_complex/1a0q_protein.pdb'
    ligand_file = '../example_complex/1a0q_ligand.mol2'

    t0 = time.time()

    Kernel = KernelFunction(kernel_type='exponential_kernel',
                 kappa=6, tau=4)

    agl_eat = AGL_EAT(Kernel, cutoff=12.0, matrix_type='Adjacency')
    agl_score_df = agl_eat.get_agl_score(protein_file, ligand_file)

    print('Shape: ', agl_score_df.shape)

    t1 = time.time()

    print('Elapsed time: ', t1-t0)

if __name__=="__main__":
    do_example()




