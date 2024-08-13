import os
import sys
import numpy as np
import json
import pickle
from os.path import join as pjoin
# import geometry as geometry
import torch
from common.skeleton import Skeleton
from common.quaternion import *
import joblib
# from human_body_prior.body_model.body_model import BodyModel
# from human_body_prior.tools.omni_tools import copy2cpu as c2c
from tqdm import tqdm

from scipy.ndimage import gaussian_filter

sys.path.append

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_dir)

from utils import *



os.environ['PYOPENGL_PLATFORM'] = 'egl'
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
ex_fps = 20

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# (joints_num, 3)

l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(positions, feet_thre, tgt_offsets):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)


    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data, global_positions, positions, l_velocity

def rotmat_to_rot6d(x):
    return x[...,:2]

def rots_to_h3d(rots, pose_seq_np):
    fps = 30    
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    pose_seq_np_n[:,:,0]*=-1    #IMPORTANT STEP = DONT KNOW WHY WE NEED TO DO but observed that to match with h3d rawposeprocessing output we need to do this step

    ### Don't know why this is happening 
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pose_seq_np_n[0]))


    data, ground_positions, positions, l_velocity = process_file(pose_seq_np_n[:,:22,:], 0.002,tgt_offsets)
    return data


# def decompose_h3d(data): 
#     sample = 
#     data = root_data
#     data = np.concatenate([data, ric_data[:-1]], axis=-1)
#     data = np.concatenate([data, rot_data[:-1]], axis=-1)
#     data = np.concatenate([data, local_vel], axis=-1)
#     data = np.concatenate([data, feet_l, feet_r], axis=-1)

#     return 

def read_file_to_list(filename):
    with open(filename, 'r') as file:
        # Read all lines from the file, strip newline characters, and return as a list
        return [line.strip() for line in file.readlines()]


# Create a mapping from opencap id to mcs score for each category 
def load_mcs_scores(csv_path):
    import pandas as pd
    mcs_sheet = pd.read_csv(csv_path,skiprows=1)
    mcs_sheet = mcs_sheet.rename(columns={'Unnamed: 5': 'LLT-Twist', 'Unnamed: 7': 'RLT-Twist', 'Unnamed: 9': 'LLTF-Twist', 'Unnamed: 11': 'RLTF-Twist', 'Unnamed: 15': 'BAP-Pull' , 'Unnamed: 17': 'BAPF-Pull' })
    mcs_scores = mcs_sheet.to_dict('index') 
    mcs_scores = dict([ (mcs_scores[index]['OpenCapID'], mcs_scores[index]) for index in mcs_scores if type(mcs_scores[index]['OpenCapID']) == str])

    for subject_id in mcs_scores: 
        for excer in mcs_scores[subject_id]:
            try: 
                mcs_scores[subject_id][excer] = int(mcs_scores[subject_id][excer])
            except ValueError as e: 
                mcs_scores[subject_id][excer] = None



    return mcs_scores

if __name__ == "__main__":
    text = "text"
    action_to_desc = {
        "BAPF":"bend and pull full",
        "CMJ":"countermovement jump",
        "LCMJ":"left countermovement jump",
        "LLT":"left lunge and twist",
        "LLTF":"left lunge and twist full",
        "RCMJ":"right countermovement jump",
        "RLT":"right lunge and twist",
        "RLTF":"right lunge and twist full",
        "RSLS":"right single leg squat",
        "SQT":"squat",
        "BAP":"bend and pull",
        "LSLS":"left single leg squat",
        "PU":"push up"
    }
    pkl_data = joblib.load(open(os.path.join(DATA_DIR,"pkl","mcs_data_v3.pkl"),"rb"))
    for k,v in pkl_data.items():
        try:
            print(k,v.shape)
        except:
            print(k,len(v))
    
    mcs_scores = load_mcs_scores(os.path.join(DATA_DIR,'mcs.csv'))
    mode = "eval"
    save_dir1 = mode + "/new_joints/"
    save_dir2 = mode +  "/new_joints_vecs/"
    save_dir3 = mode +  "/original_texts/"
    save_dir4 = mode +  "/mot_data/"
    mcs_dir = mode +  "/mcs/"
    
    os.makedirs(mode,exist_ok=True)
    os.makedirs(save_dir1,exist_ok=True)
    os.makedirs(save_dir2,exist_ok=True)
    os.makedirs(save_dir3,exist_ok=True)
    os.makedirs(save_dir4,exist_ok=True)
    os.makedirs(mcs_dir,exist_ok=True)


    file_set = read_file_to_list(mode + ".txt")

    for i,(rots_axisangle, pose_seq_np, label, subject_id, mot) in tqdm(enumerate(zip(pkl_data["poses"], pkl_data["joints_3d"],pkl_data["label"],pkl_data["subject_id"],pkl_data["mot"]))):
        if subject_id not in file_set:
            continue
        h3d_format = rots_to_h3d(rots_axisangle, pose_seq_np)
        rec_ric_data = recover_from_ric(torch.from_numpy(h3d_format).unsqueeze(0).float(), joints_num)
        action = next(k for k,v in pkl_data["label_dict"].items() if v==label)
        if action.upper() not in action_to_desc:
            continue
        np.save(pjoin(save_dir1, str(i)+".npy"), rec_ric_data.squeeze().numpy())
        np.save(pjoin(save_dir2, str(i)+".npy"), h3d_format)
        
        
        act = action_to_desc[action.upper()]
        with open(save_dir3+str(i)+".txt",'w') as f:
            f.write(act)

        if subject_id in mcs_scores and action in mcs_scores[subject_id] and mcs_scores[subject_id][action] is not None: 
            with open(mcs_dir+str(i)+".txt",'w') as f:
                f.write(str(mcs_scores[subject_id][action]))   

        # if subject_id in mcs_scores
        
        np.save(pjoin(save_dir4, str(i)+".npy"), mot)