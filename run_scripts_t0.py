import os
import copy
import numpy as np
import argparse
from datetime import datetime

if __name__ == '__main__':
    """
    commands = []
    epoch = 1200
    tiled = 0
    log_path = 'test_syn_log.txt'
    for dropoutgcn in [0.4]:
        for dropoutgat in [0.3, 0.5]:
            for hidden in [8, 16]:
                for weight_decay in [0.0005, 0.001]:
                    for seed in [1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20,21,22,23,24,25]:
                        command = f"CUDA_VISIBLE_DEVICES=1 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data.py --dataset cora --epochs {epoch} --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {tiled} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy --log_path {log_path} --seed {seed}"
                        commands.append(command)
    for cmd in commands:
        print(cmd)
        os.system(cmd)
    """
    commands = []
    epoch = 1200
    tiled = 0
    log_path = '/home/local/ASUAD/changyu2/GIB/citeseer_log.txt'
    dataset = 'citeseer'
    ps_labels_path = '/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/citeseer/all_ps_labels.npy'
    for dropoutgcn in [0.6]:
        for dropoutgat in [0.5]:
            for hidden in [64, 128, 256]:
                for weight_decay in [0.0005]:
                    for seed in [1,2,3,4,5]:
                        command = f"CUDA_VISIBLE_DEVICES=2 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train.py --dataset {dataset} --epochs {epoch} --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {tiled} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path {ps_labels_path} --log_path {log_path} --seed {seed}"
                        commands.append(command)
    for cmd in commands:
        print(cmd)
        os.system(cmd)