import os
import copy
import numpy as np
import argparse
from datetime import datetime

if __name__ == '__main__':
    commands = []
    epoch = 2000
    sample_per_class = 300
    tiled = 2
    log_path = '/home/local/ASUAD/changyu2/GIB/citeseer_syn_log.txt'
    dataset = 'citeseer'
    ps_labels_path = '/home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/citeseer/all_ps_labels.npy'
    emb_path = '/data-drive/data/graph_data/GCL_embedding/citeseer_512/all_embs.npy'
    gen_emb_path = '/data-drive/backup/changyu/expe/gge/graphvae_citeseer_freeze_enc_feat_map1_lr2.4/citeseer_latents_33000_gvae_10000_64_decode_feat.npy'
    gen_label_path = '/data-drive/backup/changyu/expe/gge/unet_1d_citeseer64_gvae_encode_all_norm_ema/labels_33000_diffusion_3000_1.8.npy'
    gen_neibor_path =  '/data-drive/backup/changyu/expe/gge/graphvae_citeseer_freeze_enc_feat_map1_lr2.4/citeseer_latents_33000_gvae_10000_64_decode_map.npy'
    for seed in [1,2,3,4,5,6,7,8,9,10]:
        for dropoutgcn in [0.6, 0.5, 0.7]:
            for dropoutgat in [0.5]:
                for hidden in [ 16, 8, 24]:
                    for weight_decay in [0.0005]:
                        command = f"CUDA_VISIBLE_DEVICES=3 /home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/GIB/train_with_synthetic_data.py --dataset {dataset} --epochs {epoch} --dropoutgcn {dropoutgcn} --dropoutgat {dropoutgat} --warmup 100 --update_frequency 1 --tildes {tiled} --hidden {hidden} --weight_decay {weight_decay} --ps_labels_path {ps_labels_path} --log_path {log_path} --seed {seed} --sample_per_class {sample_per_class} --emb_path {emb_path} --gen_emb_path {gen_emb_path} --gen_label_path {gen_label_path} --gen_neibor_path {gen_neibor_path}"
                        commands.append(command)
    for cmd in commands:
        print(cmd)
        os.system(cmd)

    