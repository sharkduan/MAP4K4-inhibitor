pocket: ./data_protain/pocket_h.pdb
ckpt: ./ckpt/PubChem-pretrained-315000.pt
num_gen: 1500
name: pocket_h.pdb
device: cuda:0
atom_temperature: 1.0
bond_temperature: 1.0
max_atom_num: 50
focus_threshold: 0.5
choose_max: True
min_dist_inter_mol: 3.0
bond_length_range: (1.0, 2.0)
max_double_in_6ring: 0
with_print: True
root_path: gen_results
readme: None