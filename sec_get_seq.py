import os
from rdkit import Chem
from rdkit.Chem import MolFromMol2File, MolToSmiles
# from Bio.PDB import PDBParser
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# from Bio import SeqIO
from tqdm import tqdm

# import os
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqUtils import seq1

# 文件夹路径
sdf_dir = 'train_set/drug_sdf'##
mol2_dir = 'train_set/drug_mol2'##
pdb_dir = 'train_set/target_pdb'##
smi_dir = 'train_set/drug_smiles'
fasta_dir = 'train_set/target_fasta'


# 确保目标文件夹存在
os.makedirs(smi_dir, exist_ok=True)
os.makedirs(fasta_dir, exist_ok=True)

# 转换SDF到SMI
def convert_sdf_to_smi(sdf_dir, smi_dir):
    for sdf_file in tqdm(os.listdir(sdf_dir)):
        if sdf_file.endswith('.sdf'):
            sdf_path = os.path.join(sdf_dir, sdf_file)
            smi_path = os.path.join(smi_dir, sdf_file.replace('.sdf', '.smi'))
            suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
            with open(smi_path, 'w') as smi_file:
                for mol in suppl:
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                            mol = Chem.RemoveHs(mol)
                            # 转换分子为SMILES字符串
                            smiles = Chem.MolToSmiles(mol)
                            smi_file.write(smiles + '\n')

                            # smi_file.write(Chem.MolToSmiles(mol) + '\n')
                        except:
                            print("无法转换为 SMILES")
                        # smi_file.write(Chem.MolToSmiles(mol) + '\n')

def convert_mol2_to_smi(mol2_dir, smi_dir):
    # 确保输出目录存在
    os.makedirs(smi_dir, exist_ok=True)

    for mol2_file in tqdm(os.listdir(mol2_dir)):
        if mol2_file.endswith('.mol2'):
            mol2_path = os.path.join(mol2_dir, mol2_file)
            smi_path = os.path.join(smi_dir, mol2_file.replace('.mol2', '.smi'))
            try:
                # 从MOL2文件读取分子
                mol = MolFromMol2File(mol2_path, sanitize=fasta_file)
                if mol:
                    # 转换分子为SMILES字符串
                    smiles = MolToSmiles(mol)
                    with open(smi_path, 'w') as smi_file:
                        smi_file.write(smiles + '\n')

                # 从MOL2文件读取分子
                mol = MolFromMol2File(mol2_path, sanitize=False, removeHs=True)
                if mol:
                    # 移除分子中的所有氢原子
                    Chem.SanitizeMol(mol)
                    noH_mol = Chem.RemoveHs(mol)
                    # 转换分子为SMILES字符串
                    smiles = MolToSmiles(noH_mol)
                    with open(smi_path, 'w') as smi_file:
                        smi_file.write(smiles + '\n')
            except Exception as e:
                print(f"无法转换为 SMILES: {mol2_file}，错误: {str(e)}")


# 转换PDB到FASTA

def convert_pdb_to_fasta(pdb_dir, fasta_dir):
    parser = PDBParser()
    ppb = PPBuilder()  # 创建一个多肽链建造者对象
    for pdb_file in tqdm(os.listdir(pdb_dir)):
        # if pdb_file != '3syr.pdb':
        #     continue
        if pdb_file.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            fasta_path = os.path.join(fasta_dir, pdb_file.replace('.pdb', '.fasta'))
            structure = parser.get_structure(pdb_file, pdb_path)

            full_sequence = ''  # 初始化空字符串以保存所有链的序列
            chain_ids = []  # 保存链的ID以用于描述

            for model in structure:
                for chain in model:
                    for pp in ppb.build_peptides(chain):
                        sequence = pp.get_sequence()
                        full_sequence += str(sequence)  # 连接序列
                        chain_ids.append(chain.id)  # 添加链ID

            # 创建一个序列记录对象
            if full_sequence:  # 确保序列不为空
                seq_record = SeqRecord(Seq(full_sequence),
                                       id='_'.join(chain_ids),  # 使用链ID作为序列ID
                                       description="Combined chains")
                # 写入文件
                with open(fasta_path, 'w') as fasta_file:
                    SeqIO.write([seq_record], fasta_file, 'fasta')

convert_sdf_to_smi(sdf_dir, smi_dir)
convert_pdb_to_fasta(pdb_dir, fasta_dir)
# convert_mol2_to_smi(mol2_dir, smi_dir)

print("转换完成！")
