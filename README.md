# UMRAL

UAMRL:Multi-Granularity Uncertainty-Aware Multimodal Representation Learning for Drug-Target Affinity Prediction

## Requirements

[numpy](https://numpy.org/)==1.23.5

[pandas](https://pandas.pydata.org/)==1.5.2

[biopython](https://biopython.org/)==1.79

[scipy](https://scipy.org/)==1.9.3

[torch](https://pytorch.org/)==2.0.1

[torch_geometric]([PyG Documentation — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/index.html))==2.3.1

## Example usage

 ```bash
# Convert source files to different file formats
python first_sel_files.py
python sec_get_seq.py


# You can get the drug map representation and the target distance matrix by running the following command.
python create_drug_graph.py
python create_target_distance_matrix.py

# When all the data is ready, you can train your own model by running the following command.
python training.py

 ```
