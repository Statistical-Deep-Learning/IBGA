import os
import copy
import numpy as np
import argparse
from datetime import datetime

if __name__ == '__main__':
    commands = []
    dropoutgcn=0.4
    dropoutgat=0.5
    weight_decay = 0.0005
    hidden= 8
    seed = 5
    for t in [0,1,2]:
        for sample in [2]:
            for lam in [1, 0.5]:
                for dropnode_rate in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    for tem in [0.1, 0.2]:
                        for seed in [5, 10]:
                            command = f"CUDA_VISIBLE_DEVICES=1 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data_with_grand.py --dataset cora --epochs 1200 --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {t} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy --log_path test_syn_grand_log.txt.txt --seed {seed} --sample {sample} --tem {tem} --lam {lam} --dropnode_rate {dropnode_rate}"
                            commands.append(command)

    for t in [0,1,2]:
        for sample in [2]:
            for lam in [1, 0.5]:
                for dropnode_rate in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    for tem in [0.1, 0.2]:
                        for seed in [15, 20]:
                            command = f"CUDA_VISIBLE_DEVICES=1 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data_with_grand.py --dataset cora --epochs 1200 --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {t} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy --log_path test_syn_grand_log.txt.txt --seed {seed} --sample {sample} --tem {tem} --lam {lam} --dropnode_rate {dropnode_rate}"
                            commands.append(command)

    for cmd in commands:
        print(cmd)
        os.system(cmd)

    

    

