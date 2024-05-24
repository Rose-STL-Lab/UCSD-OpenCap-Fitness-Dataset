import os
import sys
import joblib
import numpy as np
from utils import *
import time 
from renderer import Visualizer
from dataloader import OpenCapDataLoader, MultiviewRGB
from retarget2smpl import SMPLRetarget
import traceback

def create_pickle():
    video_dir = RENDER_DIR
    
    vis = Visualizer()
    data = {}
    data['poses'] = []
    data['joints_3d'] = []
    data['label'] = []
    data['subject_id'] = []
    
    for subject in os.listdir(INPUT_DIR):
        print("Subject:",subject)
        for sample_path in os.listdir(os.path.join(INPUT_DIR,subject,'MarkerData')):
            # Input 
            if sample_path == "Settings": continue
            
            sample_path = os.path.join(INPUT_DIR,subject,'MarkerData',sample_path)
            try: 
                poses, joints_3d, label, subject_id = get_pickle_data(sample_path,vis,video_dir=video_dir)
            except Exception as e: 
                print(traceback.format_exc())
                print(f"Error loading sample:{sample_path}:{e}")
                continue
            data['poses'].extend(poses)
    
            data['joints_3d'].extend(joints_3d)
            data['label'].extend(label)
            data['subject_id'].extend(subject_id)

    label_dict = {}
    label_dict = {label:idx for idx,label in enumerate(data['label']) if label not in label_dict}         

    data['label'] = np.array([ label_dict[x]  for x in data['label'] ])  
    data['label_dict'] = label_dict

    with open(os.path.join(PKL_DIR,'mcs_data_v2.pkl'),'wb') as f:
        joblib.dump(data,f)

def get_pickle_data(sample_path,vis,video_dir=None): 
    """
        Render dataset samples 
            
        @params
            sample_path: Filepath of input
            video_dir: Folder to store Render results for the complete worflow  
        
            
        Load input (currently .trc files) and save all the rendering videos + images (retargetting to smp, getting input text, per frame annotations etc.) 
    """
    # print("sample path")
    # print(sample_path)
    sample = OpenCapDataLoader(sample_path)
    
    # Visualize Target skeleton
    # vis.render_skeleton(sample,video_dir=video_dir)


    # Load SMPL
    sample.smpl = SMPLRetarget(sample.joints_np.shape[0],device=None)	
    

    # print("hello")
    # print(sample.name)
    # print(os.path.join(SMPL_DIR,sample.name+'.pkl'))
    sample.smpl.load(os.path.join(SMPL_DIR,sample.name+'.pkl'))

    _, joints3D,_ = sample.smpl()

    # Visualize SMPL
    # vis.render_smpl(sample,sample.smpl,video_dir=video_dir)
    
    
    # Load Video
    # sample.rgb = MultiviewRGB(sample)

    # print(f"SubjectID:{sample.rgb.session_data['subjectID']} Action:{sample.label}")

    # Visualize each view  
    # vis.render_smpl_multi_view(sample,video_dir=None)
    

    # Load Segments
    if os.path.exists(os.path.join(SEGMENT_DIR,sample.name+'.npy')):
        # print("Loading Segments")
        # print(os.path.join(SEGMENT_DIR,sample.name+'.npy'))
        sample.segments = np.load(os.path.join(SEGMENT_DIR,sample.name+'.npy'),allow_pickle=True).item()['segments']
        sample_data = {}
        sample_data['poses'] = []
        sample_data['joints_3d'] = []
        sample_data['label'] = []
        sample_data['subject_id'] = []

        for s_ind, s in enumerate(sample.segments): 
            sample_data['poses'].append(sample.smpl.smpl_params["pose_params"][s[0]:s[1]])
            sample_data['joints_3d'].append(joints3D[s[0]:s[1]])
            sample_data['label'].append(sample.label)
            # sample_data['subject_id'].append(sample.rgb.session_data['subjectID'])
            sample_data['subject_id'].append(sample.openCapID)
    else: 
        # return [],[],[],[]
        sample_data = {}
        sample_data['poses'] = []
        sample_data['joints_3d'] = []
        sample_data['label'] = []
        sample_data['subject_id'] = []
        
        sample_data['poses'].append(sample.smpl.smpl_params["pose_params"])
        sample_data['joints_3d'].append(joints3D)
        sample_data['label'].append(sample.label)
        sample_data['subject_id'].append(sample.openCapID)
    
    # if video_dir is not None:
    # 	video_dir = os.path.join(video_dir,f"{sample.openCapID}_{sample.label}_{sample.recordAttempt}")
    # vis.render_smpl_multi_view(sample,video_dir=video_dir)
    # print(sample.smpl.smpl_params["pose_params"].shape,sample.joints_3d.shape,sample.label,sample.rgb.session_data['subjectID'])

    # sample_data = {}
    # sample_data['poses'] = []
    # sample_data['joints_3d'] = []
    # sample_data['label'] = []
    # sample_data['subject_id'] = []

    # for s_ind, s in enumerate(sample.segments): 
    #     sample_data['poses'].append(sample.smpl.smpl_params["pose_params"][s[0]:s[1]])
    #     sample_data['joints_3d'].append(joints3D[s[0]:s[1]])
    #     sample_data['label'].append(sample.label)
    #     sample_data['subject_id'].append(sample.rgb.session_data['subjectID'])

    return sample_data['poses'],sample_data['joints_3d'],sample_data['label'],sample_data['subject_id']


if __name__ == "__main__": 

    # for smpl_file in os.listdir(SMPL_DIR): 
    #     # if "BAP" not in smpl_file: 
    #         # continue
    #     # if "BAPF" in smpl_file: 
    #         # continue
    #     file_name = os.path.basename(smpl_file).split(".")[0] + '.npy'
    #     if os.path.isfile(os.path.join(SEGMENT_DIR,file_name)): 
    #         continue 

    #     mcs_smpl_path = os.path.join(SMPL_DIR,smpl_file) 
    segments = create_pickle()


        # time.sleep(5)
    
    # print(err_files)
