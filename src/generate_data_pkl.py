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

    mcs_dict = load_mcs_scores(os.path.join(DATA_DIR,'mcs.csv')) 

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
    for idx,label in enumerate(data['label']):
        
        if label == "squats" or label == "St":
            label = "SQT"
        # elif label == 
        # fix any changes  
        label = label.upper()


        if label not in label_dict:
            label_dict[label] = len(label_dict)

        data['label'][idx] = label

    print(label_dict)

    data['label'] = np.array([ label_dict[x]  for x in data['label'] ])  
    data['label_dict'] = label_dict

    os.makedirs(PKL_DIR,exist_ok=True)
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
    
    sample.smpl.load(os.path.join(SMPL_DIR,sample.name+'.pkl'))

    _, joints3D,_ = sample.smpl()
    

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
    
    return sample_data['poses'],sample_data['joints_3d'],sample_data['label'],sample_data['subject_id']

# Create a mapping from opencap id to mcs score for each category 
def load_mcs_scores(csv_path):
    mcs_sheet = pd.read_csv(csv_path,skiprows=1)
    mcs_sheet = mcs_sheet.rename(columns={'Unnamed: 5': 'LLT-Twist', 'Unnamed: 7': 'RLT-Twist', 'Unnamed: 9': 'LLTF-Twist', 'Unnamed: 11': 'RLTF-Twist', 'Unnamed: 15': 'BAP-Pull' , 'Unnamed: 17': 'BAPF-Pull' })
    mcs_scores = mcs_sheet.to_dict('index') 
    mcs_scores = dict([ (mcs_scores[index]['OpenCapID'], mcs_scores[index]) for index in mcs_scores if type(mcs_scores[index]['OpenCapID']) == str])

    return mcs_scores

    return mcs_scores


if __name__ == "__main__": 
    set_logger(task_name="pkl-creation")
    segments = create_pickle()