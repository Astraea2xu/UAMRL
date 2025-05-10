import os
import shutil
from tqdm import tqdm
# 文件夹路径
# folder_a_path = 'train_set/general-set-except-refined'##
folder_a_path = 'train_set/refined-set'##
sdf_output_folder = 'train_set/drug_sdf'##
pdb_output_folder = 'train_set/target_pdb'##
mol2_output_folder = 'train_set/drug_mol2'##

# 确保输出文件夹存在
os.makedirs(sdf_output_folder, exist_ok=True)
os.makedirs(pdb_output_folder, exist_ok=True)
os.makedirs(mol2_output_folder, exist_ok=True)

for folder_name in tqdm(os.listdir(folder_a_path)):
    folder_path = os.path.join(folder_a_path, folder_name)

    # 检查文件夹名称长度是否为4，且是文件夹
    if len(folder_name) == 4 and os.path.isdir(folder_path):
        # 遍历该文件夹中的所有文件
        sdf_file = None
        pdb_file = None
        mol2_file = None
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mol2'):
                mol2_file = file_name
            if file_name.endswith('.sdf'):
                sdf_file = file_name
            elif file_name.endswith('_protein.pdb'):
                pdb_file = file_name

        # 检查是否同时存在sdf和pdb文件
        if mol2_file:
            # 构造源文件路径
            sdf_file_path = os.path.join(folder_path, sdf_file)
            pdb_file_path = os.path.join(folder_path, pdb_file)
            mol2_file_path = os.path.join(folder_path, mol2_file)

            # 构造目标文件路径，并改名为小文件夹的名字
            sdf_output_path = os.path.join(sdf_output_folder, f"{folder_name}.sdf")
            pdb_output_path = os.path.join(pdb_output_folder, f"{folder_name}.pdb")
            mol2_output_path = os.path.join(mol2_output_folder, f"{folder_name}.mol2")

            # 复制文件
            shutil.copy(sdf_file_path, sdf_output_path)
            print(f"Copied {sdf_file_path} to {sdf_output_path}")
            shutil.copy(pdb_file_path, pdb_output_path)
            print(f"Copied {pdb_file_path} to {pdb_output_path}")
            shutil.copy(mol2_file_path, mol2_output_path)
            print(f"Copied {mol2_file_path} to {mol2_output_path}")

print("Completed!")
