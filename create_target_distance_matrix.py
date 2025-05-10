import math
import os
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm
import numpy as np
parser = PDBParser(PERMISSIVE=1)

# 生成距离矩阵并进行存储
# def generate_dis_metirc(pdbid):
#     CA_Metric = []
#     coordinate_list = []
#     structure = parser.get_structure(pdbid, 'train_set/target_pdb/' + pdbid + '.pdb')
#     for chains in structure:
#         for chain in chains:
#             for residue in chain:
#                 for atom in residue:
#                     if atom.get_name() == 'CA':
#                         coordinate_list.append(list(atom.get_vector()))
#     for i in range(len(coordinate_list)):
#         ca_raw_list = []
#         for j in range(len(coordinate_list)):
#             if i == j:
#                 ca_raw_list.append(0)
#             else:
#                 ca_raw_list.append(math.sqrt((coordinate_list[i][0]- coordinate_list[j][0]) ** 2 + (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + (coordinate_list[i][2] - coordinate_list[j][2]) ** 2))
#         CA_Metric.append(ca_raw_list)
#     return CA_Metric

# print(generate_dis_metirc('1a30'))
#               structure = parser.get_structure(pdbid, 'train_set/target_pdb/' + pdbid + '.pdb')



pdb_dir = 'train_set/target_pdb'##

distance_dir = 'train_set/distance_matrix'


def generate_dis_metirc(pdb_dir,distance_dir):
    parser = PDBParser()

    for pdb_file in tqdm(os.listdir(pdb_dir)):
        CA_Metric = []
        coordinate_list = []
        if pdb_file.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            dis_path = os.path.join(distance_dir, pdb_file.replace('.pdb', '.npz'))
            structure = parser.get_structure(pdb_file, pdb_path)
            for chains in structure:
                for chain in chains:
                    for residue in chain:
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                coordinate_list.append(list(atom.get_vector()))
            for i in range(len(coordinate_list)):
                ca_raw_list = []
                for j in range(len(coordinate_list)):
                    if i == j:
                        ca_raw_list.append(0)
                    else:
                        ca_raw_list.append(math.sqrt((coordinate_list[i][0]- coordinate_list[j][0]) ** 2 + (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + (coordinate_list[i][2] - coordinate_list[j][2]) ** 2))
                CA_Metric.append(ca_raw_list)
        
        CA_np=np.array(CA_Metric)
        np.savez(dis_path,map=CA_np)
    # return CA_Metric


generate_dis_metirc(pdb_dir,distance_dir)