import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_dir)

from utils import *

from HumanML3D.common.motion_process import recover_from_ric  # For 263 representation


def load_data(args,npy_file):    
    if args.data_rep == 'LIMO':
        motion = np.load(os.path.join(args.sample_dir, npy_file))
        assert motion.shape[-1] == 263, f"Given data not in humanml format. Should have 263 dimensions found {motion.shape[-1]}" 
    
        motion = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()
        return motion[None]
        

    elif args.data_rep == 'humanml':
        motion = np.load(os.path.join(args.sample_dir, npy_file))
        assert motion.shape[-1] == 263, f"Given data not in humanml format. Should have 263 dimensions found {motion.shape[-1]}" 
        raise NotImplementedError  

    elif args.data_rep == 'xyz' or args.data_rep == 't2m':
        motion = np.load(os.path.join(args.sample_dir, npy_file))
    elif args.data_rep == 'mdm':
        with open(os.path.join(args.sample_dir, npy_file), 'rb') as f:
            motion = np.load(f, allow_pickle=True).item()
            assert np.all(motion['lengths'] == motion['lengths'][0]), f"Given data not in of same motion length format. Found {motion['lengths']}" 
            motion = motion['motion'][:, :, :,:motion['lengths'][0]].transpose((0,3,1,2))
    else:
        raise NotImplementedError("Data representation not implemented. Please choose from 'xyz', 'humanml' or 'brax_ik'")

    return motion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, default='data/humanml3d/new_joints_vecs')
    parser.add_argument('--data_rep', type=str, default='xyz', choices=['xyz', 'humanml', 'brax_ik','mdm','t2m', 'LIMO'])
    parser.add_argument('--feet_threshold', type=float, default=0.01)
    parser.add_argument('--framerate', type=float, default=60)
    args = parser.parse_args()

    # find all npy files in sample_dir but not its subdirectories
    npy_files = []
    for file in os.listdir(args.sample_dir):
        if args.data_rep == 'LIMO':
            if os.path.isdir(os.path.join(args.sample_dir,file)):
                for f in os.listdir(os.path.join(args.sample_dir,file)): 
                    if f.endswith('.npy'): 
                        npy_files.append(os.path.join(args.sample_dir,file,f))
        else: 
            if file.endswith('.npy'):
                npy_files.append(file)

    # random permute
    npy_files = np.random.permutation(npy_files)

    # calculate metrics
    loss_pn, loss_fl, loss_sk, metr_act = [], [], [], []

    # Plot the min height to access the ground floor location
    min_height_list = []
    for npy_file in tqdm(npy_files):
        motion = load_data(args,npy_file)

        motion = torch.from_numpy(motion).float()
        min_height, idx = motion[..., 1].min(dim=-1)
        min_height_list.extend(list(min_height.reshape(-1).data.cpu().numpy()))

    # print(min_height_list)    
    # import matplotlib.pyplot as plt
    # plt.hist(min_height_list,bins=100)
    # plt.show()

    y_translation = -np.median(min_height_list)
    # y_translation = 0

    print(f"Ground height:",y_translation, "calculated as the median of the min height across all samples")

    for npy_file in tqdm(npy_files):
        motion = load_data(args,npy_file)
        if args.data_rep == 'mdm':
            motion = np.random.permutation(motion)


        # import polyscope as ps
        # ps.init()
        # ps.register_point_cloud("motion",motion[0,0])
        # ps.set_ground_plane_height_factor(y_translation)
        # ps.show()

        # y_translation = 0.0
        motion = torch.from_numpy(motion).float()
        min_height, idx = motion[..., 1].min(dim=-1)
        # print(min_height,idx,motion[..., 1].shape)
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
    print('Mean: PN: %.4f (meters), FL: %.4f (meters), SK: %.4f (meters/seconds), AC: %.4f' % (loss_pn.mean(), loss_fl.mean(), args.framerate*loss_sk.mean(), metr_act.mean()))
    print('STD : PN: %.4f (meters), FL: %.4f (meters), SK: %.4f (meters/seconds), AC: %.4f' % (loss_pn.std(), loss_fl.std(), args.framerate*loss_sk.std(), metr_act.std()))
    