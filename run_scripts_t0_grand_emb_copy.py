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
    hidden=16 
    seed = 1
    for sample in [6,8]:
        for lam in [1, 0.5]:
            for dropnode_rate in [0.5, 0.7]:
                for tem in [0.1, 0.3]:
                        command = f"CUDA_VISIBLE_DEVICES=2 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_emb_data_with_grand.py --dataset cora --epochs 1200 --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes 0 --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy --log_path test_grand_emb_log_k68.txt --seed {seed} --sample {sample} --tem {tem} --lam {lam} --dropnode_rate {dropnode_rate}"
                        commands.append(command)
    for cmd in commands:
        print(cmd)
        os.system(cmd)