import os
import copy
import numpy as np
import argparse
from datetime import datetime

if __name__ == '__main__':
    commands = []
    for dropoutgcn in [0.4, 0.6, 0.8]:
        for dropoutgat in [0.3, 0.5, 0.7]:
            for hidden in [8, 16]:
                for weight_decay in [0.0005, 0.001]:
                    command = f"CUDA_VISIBLE_DEVICES=1 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data.py --dataset cora --epochs 1200 --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes 0 --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path /home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gcn_cora_82.npy --log_path test_syn_log.txt --seed 1"
                    commands.append(command)
    for cmd in commands:
        print(cmd)
        os.system(cmd)