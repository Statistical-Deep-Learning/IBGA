import os
import copy
import numpy as np
import argparse
from datetime import datetime

if __name__ == '__main__':
    commands = []
    dropoutgcn=0.6
    dropoutgat=0.5
    weight_decay = 0.0005
    tem = 0.1
    lam = 1
    sample = 2
    t = 0
    for hidden in [64, 128]: 
        for dropnode_rate in [0.5, 0.7]:
            for seed in [1,2,3,4]:
                            command = f"CUDA_VISIBLE_DEVICES=2 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data_with_grand.py --dataset cora --epochs 1200 --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {t} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy --log_path test_grand_log.txt --seed {seed} --sample {sample} --tem {tem} --lam {lam} --dropnode_rate {dropnode_rate}"
                            commands.append(command)

    t = 1
    for hidden in [64, 128]: 
        for dropnode_rate in [0.5, 0.7]:
            for seed in [1,2,3,4]:
                            command = f"CUDA_VISIBLE_DEVICES=2 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data_with_grand.py --dataset cora --epochs 1200 --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {t} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy --log_path test_grand_log.txt --seed {seed} --sample {sample} --tem {tem} --lam {lam} --dropnode_rate {dropnode_rate}"
                            commands.append(command)

    for cmd in commands:
        print(cmd)
        os.system(cmd)

    

    

