import os
import openbabel
from tqdm import tqdm
# 设置A文件夹和B文件夹路径
input_folder = "train_set/drug_sdf"
output_folder = "train_set/drug_smiles"


# sdf_dir = 'train_set/drug_sdf'##
# mol2_dir = 'train_set/drug_mol2'##
# pdb_dir = 'train_set/target_pdb'##
# smi_dir = 'train_set/drug_smiles'
# fasta_dir = 'train_set/target_fasta'
# 确保B文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化Open Babel转换器
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("sdf", "smi")

# 遍历A文件夹中的所有文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".sdf"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".sdf", ".smi"))

        # 创建OBMol对象来存储分子数据
        mol = openbabel.OBMol()

        # 读取SDF文件
        obConversion.ReadFile(mol, input_path)

        # 删除氢原子
        mol.DeleteHydrogens()

        # 将分子转换为SMILES格式
        smiles = obConversion.WriteString(mol).strip()
        print(smiles)
        # 保存转换后的SMILES到文件
        with open(output_path, "w") as outfile:
            outfile.write(smiles)

print("转换完成！")
