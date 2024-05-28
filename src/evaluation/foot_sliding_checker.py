import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_dir)

from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, default='data/humanml3d/new_joints_vecs')
    parser.add_argument('--data_rep', type=str, default='xyz', choices=['xyz', 'humanml', 'brax_ik'])
    parser.add_argument('--feet_threshold', type=float, default=10.0)
    args = parser.parse_args()

    # find all npy files in sample_dir but not its subdirectories
    npy_files = []
    for file in os.listdir(args.sample_dir):
        if file.endswith('.npy'):
            npy_files.append(file)
    # random permute
    npy_files = np.random.permutation(npy_files)

    # calculate metrics
    loss_pn, loss_fl, loss_sk, metr_act = [], [], [], []

    for npy_file in tqdm(npy_files):
        motion = np.load(os.path.join(args.sample_dir, npy_file))

        
        if args.data_rep == 'humanml':
            assert motion.shape[-1] == 263, f"Given data not in humanml format. Should have 263 dimensions found {motion.shape[-1]}" 
            raise NotImplementedError  

        elif args.data_rep == 'xyz':
            raise NotImplementedError  
        else:
            raise NotImplementedError("Data representation not implemented. Please choose from 'xyz', 'humanml' or 'brax_ik'")
        
        y_translation = 520.0
        motion = torch.from_numpy(motion).float()
        min_height, idx = motion[..., 1].min(dim=-1)
        min_height = min_height + y_translation
        pn = -torch.minimum(min_height, torch.zeros_like(min_height))  # penetration
        pn[pn < args.feet_threshold] = 0.0
        fl = torch.maximum(min_height, torch.zeros_like(min_height))  # float
        fl[fl < args.feet_threshold] = 0.0
        bs, t = idx.shape
        I = torch.arange(bs).view(bs, 1).expand(-1, t-1).long()
        J = torch.arange(t-1).view(1, t-1).expand(bs, -1).long()
        J_next = J + 1
        feet_motion = motion[I, J, idx[:, :-1]]
        feet_motion_next = motion[I, J_next, idx[:, :-1]]
        sk = torch.norm(feet_motion - feet_motion_next, dim=-1)
        contact = fl[:, :t] < args.feet_threshold
        sk = sk[contact[:, :-1]]  # skating
        # action: measure the continuity between frames
        vel = motion[:, 1:] - motion[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        acc = torch.norm(acc, dim=-1)
        # all losses
        loss_pn.append(pn[:, :t].view(-1))
        loss_fl.append(fl[:, :t].view(-1))
        loss_sk.append(sk.view(-1))
        metr_act.append(acc[:, :t].view(-1))

        
    loss_pn = torch.cat(loss_pn, dim=0)
    loss_fl = torch.cat(loss_fl, dim=0)
    loss_sk = torch.cat(loss_sk, dim=0)
    metr_act = torch.cat(metr_act, dim=0)
    print('PN: %.4f, FL: %.4f, SK: %.4f, AC: %.4f' % (loss_pn.mean(), loss_fl.mean(), loss_sk.mean(), metr_act.mean()))
    