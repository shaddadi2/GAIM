
# **GAIM: Attacking Graph Neural Networks via Adversarial Influence Maximization**

This repo provides the implementations for our attack method. 

## Requirements

- python 3.8.16
- dgl 0.4.3
- numpy 1.23.5
- ogb 1.3.6
- torch 2.0.1
- torch-geometric 2.3.1
- torch-sparse 0.6.17

## Implementation

This repo includes four python files for out attacks. The file 'attack_type_i.py' and 'attack_type_ii.py' corresponds to our Type-I attack and Type-II attack, respectively. The file 'attack_untargeted_norm_length.py' is used to run experiments on datasets Cora, Citeseer and Pubmed. The file 'attack_untargeted.py' is for dataset Flickr and Reddit.

Example commands to run the code:
`python attack_untargeted_norm_length.py --dataset cora --seed 0 --threshold 0.1 --node_perc 0.01 --feature_perc 0.02 --epochs 200`
`python attack_type_i.py --dataset cora --seed 0 --threshold 0.1 --node_perc 0.01 --feature_perc 0.02 --epochs 200`
