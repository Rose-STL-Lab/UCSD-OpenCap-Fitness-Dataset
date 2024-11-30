# coding: utf-8

import os
import tqdm
import numpy as np
import pandas as pd
import plotly
import numpy as np
import argparse

import plotly.graph_objects as go
from plotly.subplots import make_subplots



from simulation import load_simulation_data
from startStopDetection import temporal_segementation, time_normalization


# MCS_PATH ='/data/panini/MCS_DATA/'
MCS_PATH = ['/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA', '/data/panini/MCS_DATA/', '/mnt/data/MCS_DATA/']
for mcs_path in MCS_PATH:
    if os.path.exists(mcs_path):
        MCS_PATH = mcs_path
        break

data_path = os.path.join(MCS_PATH, 'Data')

parser = argparse.ArgumentParser(description='Create report on the surrogate results. Usage: ')
parser.add_argument('--surrogates', nargs='+', help='List of surrogate results. Note: Sort by preference. Last experiment given red color', required=True)
parser.add_argument('--name', help='Name of the report', default="MCS-Surrogate.pdf")
parser.add_argument('--pdfs', help='Directory to store pdfs', default="pdfs")
parser.add_argument('--no-isMCS', dest='isMCS', action='store_false', help='Flag to indicate if MCS scores are present')
parser.add_argument('--isMCS', dest='isMCS', action='store_true', help='Flag to indicate if MCS scores are present')


args = parser.parse_args()

isMCS = args.isMCS
report_name = args.name
pdf_dir = args.pdfs

print(f"Creating report:{report_name} in directory:{pdf_dir} isMCS:{isMCS}")

os.makedirs(pdf_dir,exist_ok=True)

# 
# surrogate_results_list = ["transformer_surrogate_v3_activations"] 

surrogate_results_list = args.surrogates
surrogate_results_list = [surrogate_result if os.path.exists(surrogate_result) else os.path.join(MCS_PATH, surrogate_result) for surrogate_result in surrogate_results_list] 
assert all([os.path.exists(surrogate_result) for surrogate_result in surrogate_results_list]), f"All surrogate results should exist:{[os.path.exists(surrogate_result) for surrogate_result in surrogate_results_list]}"




# PPE Files containing with MCS Scores
mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
mcs_scores = [4,4,2,3,2,4,3,3,2,3,4,3,4,2,2,3,4,4,3,3,3 ]
mcs_scores = dict(zip(mcs_sessions,mcs_scores))
PPE_Subjects = ["PPE09182201","PPE09182202","PPE09182203","PPE09182204","PPE09182205","PPE09182206","PPE09182207","PPE09182208","PPE09182209","PPE091822010","PPE09182211","PPE09182212","PPE09182213","PPE09182214","PPE09182215","PPE09182216","PPE09182217","PPE09182218","PPE09182219","PPE09182220","PPE09182221"]
PPE_Subjects = dict(zip(mcs_sessions,PPE_Subjects))

if not isMCS:
    mcs_sessions = os.listdir(data_path)

    subject2opencap = pd.read_table(os.path.join(MCS_PATH, 'subject2opencap.txt'),sep=',')
    PPE_Subjects = dict(zip( subject2opencap[' OpenCap-ID'].tolist(), subject2opencap['PPE'].tolist()))

    for session in PPE_Subjects:
        if session not in mcs_scores:
            mcs_scores[session] = -1  
            print(session, PPE_Subjects[session], mcs_scores[session])

subjects = load_simulation_data(mcs_sessions[::-1], data_path, surrogates=surrogate_results_list)

print(f"Subjects Loaded:", len(subjects.keys()))


# Name Mappings from Github copilot
plot_names_mapping = {
    'lumbar_extension': 'Trunk Tilt',
    'pelvis_tilt': 'Pelvic Tilt',
    'hip_flexion_l': 'Left Hip Flexion/Extension',
    'hip_flexion_r': 'Right Hip Flexion/Extension',
    'knee_angle_l': 'Left Knee Flexion/Extension',
    'knee_angle_r': 'Right Knee Flexion/Extension',
    'ankle_angle_l': 'Left Ankle Dorsi/Plantar',
    'ankle_angle_r': 'Right Ankle Dorsi/Plantar'
}

plot_muscle_activations_mapping = {
    'soleus_l/activation': 'Soleus (Left)',
    'vasint_l/activation': 'Vastus Intermedius (Left)',
    'vaslat_l/activation': 'Vastus Lateralis (Left)',
    'vasmed_l/activation': 'Vastus Medialis (Left)',
    'soleus_r/activation': 'Soleus (Right)',
    'vasint_r/activation': 'Vastus Intermedius (Right)',
    'vaslat_r/activation': 'Vastus Lateralis (Right)',
    'vasmed_r/activation': 'Vastus Medialis (Right)'
}

from collections import defaultdict

# Group muscles by their base name (without _l or _r)
grouped_muscles = defaultdict(list)
for key in plot_muscle_activations_mapping.keys():
    base_name = key.split('_')[0]
    grouped_muscles[base_name].append(key)
# Create a list of muscle pairs
muscle_pairs = []
for base_name, keys in grouped_muscles.items():
    if len(keys) == 2:  # Ensure both left and right muscles are present
        muscle_pairs.append((keys[0], keys[1]))
        
# Number of muscle pairs per page (3 columns * 3 rows)
num_pairs_per_page = 4 * 2

# Split muscle pairs into groups for each page
pages = [muscle_pairs[i:i + num_pairs_per_page] for i in range(0, len(muscle_pairs), num_pairs_per_page)]

# Create dictionaries for each page
plot_muscle_activations_mapping_pages = []
for page in pages:
    page_dict = {}
    for left_key, right_key in page:
        page_dict[left_key] = plot_muscle_activations_mapping[left_key]
        page_dict[right_key] = plot_muscle_activations_mapping[right_key]
    plot_muscle_activations_mapping_pages.append(page_dict)

# Print the dictionaries for each page
for i, page_dict in enumerate(plot_muscle_activations_mapping_pages):
    print(f"Muscle Activation Page {i + 1}:")
    print(page_dict)
    print()

def get_plotting_data(subject,trial,remove_headers=['pelvis_tx','pelvis_ty','pelvis_tz']): 
    # Get the subject data and remove pelvis translation 

    headers = subject['dof_names'] 

    keep_index = [headers.index(header)  for header in plot_names_mapping.keys()]

    plot_headers = [headers[i] for i in keep_index]

    plot_data = {}
    plot_data['kinematics'] = subject[trial]['kinematics'] 
    plot_data['kinematics'] = plot_data['kinematics'][plot_headers]


    plot_headers_kinetics = [ h +'_moment'  for h in plot_headers]
    plot_data['kinetics'] = subject[trial]['kinetics'] 
    plot_data['kinetics'] = plot_data['kinetics'][plot_headers_kinetics]




    for page_index, page_dict in enumerate(plot_muscle_activations_mapping_pages):
        plot_muscle_activations_index = [headers.index(header) for header in page_dict.keys()]
        plot_muscle_activations_headers = [headers[i] for i in plot_muscle_activations_index]


        # Store muscle activations and surrogate results
        plot_data[f'muscle_activations-{page_index}'] = {}
        plot_data[f'muscle_activations-{page_index}']['ground_truth'] = subject[trial]['kinematics'][plot_muscle_activations_headers]

        if 'surrogate' in subject[trial]:
            for surrogate_name  in subject[trial]['surrogate']: 
                plot_data[f'muscle_activations-{page_index}'][surrogate_name] = subject[trial]['surrogate'][surrogate_name][plot_muscle_activations_headers]    



    for k in plot_data:
        if 'muscle_activations' in k:
            mc_page_index = int(k.split('-')[-1])
            for muscle_activations in plot_data[k]:
                assert plot_data[k][muscle_activations].shape[-1] == len(plot_muscle_activations_mapping_pages[mc_page_index]), f"Length of headers should match headers length. Found:{plot_data[k][surrogate_name].shape[-1]} , expected:{plot_muscle_activations_mapping_pages[mc_page_index]}"
                if type(plot_data[k][muscle_activations]) != np.ndarray:
                    plot_data[k][muscle_activations] = plot_data[k][muscle_activations].to_numpy()

        else: 
            assert plot_data[k].shape[-1] == len(plot_headers), f"Length of headers should match headers length. Found:{plot_data[k].shape[-1]} , expected:{len(plot_headers)}"

            if type(plot_data[k]) != np.ndarray:
                plot_data[k] = plot_data[k].to_numpy()



    return plot_headers, plot_data

# get_plotting_data(subjects["015b7571-9f0b-4db4-a854-68e57640640d"],'SQT01_segment_3')






###############################################################

# skip_subjects = ["c08f1d89-c843-4878-8406-b6f9798a558e","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","c28e768f-6e2b-4726-8919-c05b0af61e4a","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","349e4383-da38-4138-8371-9a5fed63a56a"]
# skip_subjects = [mcs_sessions[0], mcs_sessions[1]]
skip_subjects = []
plot_headers = None
for subject_ind, subject_name in tqdm.tqdm(enumerate(mcs_sessions)):
    
    print(f"Evaluating Id: {subject_ind} Name: {subject_name}")
    
    if subject_name in skip_subjects: continue
    
    if subject_name not in subjects: 
        print(f"Subject not found in data:{subject_name}")
        continue
    
    if len(subjects[subject_name]) <= 1:  # If dict is empty skip.
        print(f"Subject is empty:{subjects[subject_name]}")    
        continue
    
    print(f" Data:{subjects[subject_name].keys()}")
    
    
     
    plot_data = {}

    # Get seconds per frame 
    seconds_per_frame = 0 

    for trial_name in subjects[subject_name]: 
        if trial_name == 'dof_names': continue
        
        if trial_name == 'seconds_per_frame': continue  
    

        if len(subjects[subject_name][trial_name]) <= 1:  # If dict is empty skip.
            print(f"Trial is empty:{subjects[subject_name][trial_name]}") 
            continue 
        print(subjects[subject_name][trial_name].keys())
    
        trial_length = subjects[subject_name][trial_name]['kinematics']['time'].iloc[-1] - subjects[subject_name][trial_name]['kinematics']['time'].iloc[0]
        if trial_length < 1: continue # Can't perform squat in less tha a second . 
        
        
        seconds_per_frame += trial_length
        
        plot_headers, plot_data[trial_name] = get_plotting_data(subjects[subject_name],trial_name)
        
        print(seconds_per_frame,subjects[subject_name][trial_name]['kinematics']['time'].iloc[-1],subjects[subject_name][trial_name]['kinematics']['time'].iloc[0])

        print(f"Subject:{subject_name} Trial Index:{trial_name} Length: {trial_length} Headers:{plot_headers}")

    if seconds_per_frame == 0: 
        print("Tracks are empty, skipping subject")
        continue

    assert seconds_per_frame > 0, f"Subject Index:{subject_ind} seconds_per_frame should be greater 0. Likely no trial found to evaluate." 
    
    seconds_per_frame /= sum([len(plot_data[trial_name]['kinematics']) for trial_name in plot_data])


    
    
    fig_title = f"Temporal Segmentation using Knee Kinematics for Subject:{subject_name}"

    num_segments = 1 # Number of segments per trial

    # Temporal Segmentation (using knee angles kinematics since it gave the most reasonable results) 
    segments_fig, segments_all_trials = temporal_segementation(plot_data,plot_headers,\
                                      num_segments=num_segments, seconds_per_frame=seconds_per_frame,\
                                      allowed_height_difference_threshold=0.15,\
                                      isdeg=True,visualize=False,fig_title=fig_title)



    os.makedirs(pdf_dir,exist_ok=True)
    plotly.io.write_image(segments_fig, os.path.join(pdf_dir, f'{PPE_Subjects[subject_name]}_segmentation.pdf'), format='pdf')

    if len(segments_all_trials) == 0: 
        print("Could not find segments")
        continue  
    
    # Update data information
    for trial_name in segments_all_trials:
        subjects[subject_name][trial_name]['segments'] = segments_all_trials[trial_name]
    
    subjects[subject_name]['seconds_per_frame'] = seconds_per_frame
        

# Merge trials across distributions for trials 
def plot_simulation_data(headers,plot_data,title_text="Plot Data",visualize=False,data_type='kinematics',num_cols=4): 
    

    if 'kinematics' in data_type or 'kinetics' in data_type:
        assert plot_data.shape[-1] == 101, "Length of data should be 101" 
        assert len(headers) == plot_data.shape[0], "Length of headers should match headers length"
    
    elif 'muscle_activations' in data_type:
        
        assert 'ground_truth' in plot_data, "Ground truth should be present"
        
        surrogate_exps = [os.path.basename(surrogate) for surrogate in surrogate_results_list]

        for muscle_activations in plot_data:    
            assert muscle_activation in surrogate_exps, f"Surrogate results should be present for muscle activations:{muscle_activations}"
            assert plot_data[muscle_activations].shape[-1] == 101, "Length of data should be 101"
            assert len(headers) == plot_data[muscle_activations].shape[0], "Length of headers should match headers length"

    assert num_cols > 0, "Number of columns should be greater than 0"
    
    
    num_rows = int(np.ceil(len(headers)/num_cols))
 
    if data_type == 'kinematics' or data_type == 'kinetics':
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_names_mapping[header] for header in headers]) 
    elif  'muscle_activations' in data_type:
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_muscle_activations_mapping[header] for header in headers])

        # Colors for left and right sides
        colors = {'ground_truth': 'green', surrogate_exps[-1]: 'red'}
        for surrogate_name in surrogate_exps:
            if surrogate_name not in colors:
                color = (255*np.random.random(3)).astype(int)
                color = 'rgb('+','.join([str(c) for c in color])+')'
                colors[surrogate_name] = 'blue'

    else: 
        raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")
    
    

    # Create each subplot
    for i, header in enumerate(headers):
        row = i // num_cols + 1
        col = i % num_cols + 1

        if 'muscle_activations' in data_type:
            title = plot_muscle_activations_mapping[header]

            # Plot every kinematics data
            x = np.linspace(0,1,num=plot_data['ground_truth'][i].shape[-1])
            for muscle_activation_name in plot_data:

                color = colors[muscle_activation_name]
    
                for j in range(plot_data[muscle_activation_name][i].shape[0]):
                    fig.add_trace(go.Scatter(x=x, y=plot_data[muscle_activation_name][i,j], name=f'{muscle_activation_name}',line=dict(color=color),showlegend=(i==len(headers)-1 and j == 0)), row=row, col=col)

                    fig.add_hline(y=max(plot_data[muscle_activation_name][i,j]), line_width=1, line_dash="dash", line_color=color, row=row, col=col)



        elif data_type == 'kinematics' or data_type == 'kinetics':
            title = plot_names_mapping[header]

            # Plot every kinematics data
            x = np.linspace(0,1,num=plot_data[i].shape[-1])
            for j in range(plot_data[i].shape[0]):
                fig.add_trace(go.Scatter(x=x, y=plot_data[i,j], name=f'{title}',showlegend=False), row=row, col=col)

        else:
            raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")


    
        # Update y-axis label
        if data_type == 'kinematics':
            fig.update_yaxes(title_text='deg', title_standoff=10, row=row, col=col)
        elif data_type == 'kinetics':
            fig.update_yaxes(title_text='Nm', row=row, col=col)
        else:
            fig.update_yaxes(title_text='0-1', row=row, col=col)
            
        # fig.update_yaxes(title_text='deg', row=row, col=col)

    # Update x-axis label for the bottom row
    for col in range(1, num_rows+1):
        for row in range(1,num_cols+1):
            fig.update_xaxes(title_text='% SQT Cycle (Seconds)', row=row, col=col)

    # Update layout
    fig.update_layout(height=1000, width = 2000,
                        showlegend=True,  title_x=0.5,
                        title_text=title_text,
                        font_family="Times New Roman",
                        font_color="black",
                        title_font_family="Times New Roman",
                        title_font_color="black")

    # Show the figure
    if visualize: 
        fig.show()
    
    return fig


# # Plot indivifual sample & Store aggregate (mean, std, list ) values 

# In[10]:


import copy

# Skip following subjects for torque simulation
# skip_subjects = ["c08f1d89-c843-4878-8406-b6f9798a558e","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","c28e768f-6e2b-4726-8919-c05b0af61e4a","349e4383-da38-4138-8371-9a5fed63a56a", "0e10a4e3-a93f-4b4d-9519-d9287d1d74eb",]


# Skip following subjects for muscle simulation
# skip_subjects = ["c08f1d89-c843-4878-8406-b6f9798a558e", "0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45", "c28e768f-6e2b-4726-8919-c05b0af61e4a", "349e4383-da38-4138-8371-9a5fed63a56a", "3d1207bf-192b-486a-b509-d11ca90851d7",   "0e10a4e3-a93f-4b4d-9519-d9287d1d74eb", "2345d831-6038-412e-84a9-971bc04da597", ""] # Skip subject for muscle simulation  
skip_subjects = []
num_segments = 1

# mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]

# mcs_scores = [4,4,2,3,2,4,3,3,2,3,0,3,4,2,2,3,4,4,3,3,3]
# mcs_scores = dict(zip(mcs_sessions,mcs_scores))

# PPE_Subjects = ["PPE09182201","PPE09182202","PPE09182203","PPE09182204","PPE09182205","PPE09182206","PPE09182207","PPE09182208","PPE09182209","PPE091822010","PPE09182211","PPE09182212","PPE09182213","PPE09182214","PPE09182215","PPE09182216","PPE09182217","PPE09182218","PPE09182219","PPE09182220","PPE09182221"]
# PPE_Subjects = dict(zip(mcs_sessions,PPE_Subjects))

############### STORE Manual segmentation results here: 
# manually_segment_subjects_list = [("3d1207bf-192b-486a-b509-d11ca90851d7","SQT01_segment_1"),
#                                   ("3d1207bf-192b-486a-b509-d11ca90851d7","SQT01_segment_2"),
#                                   ("3d1207bf-192b-486a-b509-d11ca90851d7","SQT01_segment_3"), 
                                  
#                                   ("2345d831-6038-412e-84a9-971bc04da597","SQT01_segment_1")]

manual_segments = {} 
# for subject_name,trial_name in manually_segment_subjects_list: 
#     if subject_name not in manual_segments: 
#         manual_segments[subject_name] = {}
#         continue 
    
#     if trial_name not in manual_segments[subject_name]:
#         manual_segments[subject_name][trial_name] = np.zeros((1,2)) # If not segment found, skip the trial
#         continue 
    
#     if 'segments' not in subjects[subject_name][trial_name]: 
#         continue 
    
#     manual_segments[subject_name][trial_name] = copy.deepcopy(subjects[subject_name][trial_name]['segments'])

    
# # Check and update the first set of keys
# if "3d1207bf-192b-486a-b509-d11ca90851d7" in manual_segments:
#     if "SQT01_segment_3" in manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]:
#         manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_3"][0][0] += 15
#         manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_3"][0][1] += 15

#     # if "SQT01_segment_1" in manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]:
#     #     manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_1"][0][0] += 40
#     #     manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_1"][0][1] += 40

# # Check and update the second set of keys
# if "2345d831-6038-412e-84a9-971bc04da597" in manual_segments:
#     if "SQT01_segment_1" in manual_segments["2345d831-6038-412e-84a9-971bc04da597"]:
#         manual_segments["2345d831-6038-412e-84a9-971bc04da597"]["SQT01_segment_1"][0][0] += 40 
##########################################################
aggregate_data = {}
if isMCS:
    mcs_aggregate_data = {2:{}, 3:{}, 4:{}}
else: 
    mcs_aggregate_data = {2:{}, 3:{}, 4:{}, 0:{}, -1:{}}

## Need to compute R2 values for each surrogate result
R2 = {'total_predictions':0, 'SST':np.zeros((0,len(page_dict))),  'SSE':{}}
for surrogate_result in ['ground_truth'] + surrogate_results_list:
    surrogate_result = os.path.basename(surrogate_result)
    R2['SSE'][surrogate_result] = np.zeros((0,len(page_dict)))
    

for plotting_variable in ['kinematics','kinetics']:
    aggregate_data[plotting_variable] = {}
    aggregate_data[plotting_variable]['mean'] = np.zeros((len(plot_names_mapping),101))
    aggregate_data[plotting_variable]['std'] = np.zeros((len(plot_names_mapping),101))

    for mcs_score in mcs_aggregate_data:
        mcs_aggregate_data[mcs_score][plotting_variable] = {}
        mcs_aggregate_data[mcs_score][plotting_variable]['mean'] = np.zeros((len(plot_names_mapping),101))
        mcs_aggregate_data[mcs_score][plotting_variable]['std'] = np.zeros((len(plot_names_mapping),101))
        mcs_aggregate_data[mcs_score][plotting_variable]['list'] = np.zeros((0,len(page_dict),101))
        mcs_aggregate_data[mcs_score][plotting_variable]['ppe_names'] = []
        mcs_aggregate_data[mcs_score][plotting_variable]['ppe_trial'] = []
        mcs_aggregate_data[mcs_score][plotting_variable]['total_trials'] = 0

######## For muscle activations  #####################
for page_index, page_dict in enumerate(plot_muscle_activations_mapping_pages):
    aggregate_data[f'muscle_activations-{page_index}'] = {}

    for mcs_score in mcs_aggregate_data:
        mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'] = {}

    for muscle_activations in ['ground_truth'] + [ os.path.basename(surrogate_result) for surrogate_result in surrogate_results_list]:
        aggregate_data[f'muscle_activations-{page_index}'][muscle_activations] = {}
        aggregate_data[f'muscle_activations-{page_index}'][muscle_activations]['mean'] = np.zeros((len(page_dict),101))
        aggregate_data[f'muscle_activations-{page_index}'][muscle_activations]['std'] = np.zeros((len(page_dict),101))


        
        for mcs_score in mcs_aggregate_data:
            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations] = {}
            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations]['mean'] = np.zeros((len(page_dict),101))
            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations]['std'] = np.zeros((len(page_dict),101))

            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations]['list'] = np.zeros((0,len(page_dict),101))
            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations]['ppe_names'] = []
            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations]['ppe_trial'] = []
            mcs_aggregate_data[mcs_score][f'muscle_activations-{page_index}'][muscle_activations]['total_trials'] = 0







total_trials = 0

for subject_ind, subject_name in tqdm.tqdm(enumerate(subjects)):

    # Check if all the details that have to be plotted exist    
    if subject_name in skip_subjects: 
        continue 
    
    if len(subjects[subject_name]) <= 1:  # If dict is empty skip.
        print(f"Subject is empty:{subjects[subject_name]}")    
        continue
    
    print(f" Data:{subjects[subject_name].keys()}")
    
    plot_headers = None
    plot_data = {}

    for trial_name in subjects[subject_name]: 
        if trial_name == 'dof_names': continue
        if trial_name == 'seconds_per_frame': continue  
        
        if len(subjects[subject_name][trial_name]) <= 1:  # If dict is empty skip.
            print(f"Trial is empty:{subjects[subject_name][trial_name]}") 
            continue 
        print(subjects[subject_name][trial_name].keys())
    
        trial_length = subjects[subject_name][trial_name]['kinematics']['time'].iloc[-1] - subjects[subject_name][trial_name]['kinematics']['time'].iloc[0]
        if trial_length < 1: continue # Can't perform squat in less tha a second. 
        
 
        plot_headers, plot_data_trial = get_plotting_data(subjects[subject_name],trial_name)
        
    
        # Temporal Segmentation (using knee angles kinematics since it gave the most reasonable results) 
        try: 
            if subject_name in manual_segments and trial_name in manual_segments[subject_name]:
                segments = manual_segments[subject_name][trial_name]
            else: 
                segments = subjects[subject_name][trial_name]['segments']            
        except Exception as e: 
            print(f"Error computing segments using temporal segmetnation",e) 
            continue
        
        
        segment_time = sum([  segments[i][1] - segments[i][0] for i in range(len(segments))])*subjects[subject_name]['seconds_per_frame']
        
        print(f"    Subject:{subject_name} Trial Index:{trial_name} Length: {trial_length} Segment Length:{segment_time}  {segments} Headers:{plot_headers}")
        
        for plotting_variable in plot_data_trial:
            
            if plotting_variable == 'kinematics' or plotting_variable == 'kinetics':
                assert len(plot_data_trial[plotting_variable].shape) == 2, "Data should be 2D"

                time_normalized_series = [time_normalization(plot_data_trial[plotting_variable][segment[0]:segment[1]]) for segment in segments if segment[1] > segment[0] ] 
                
                if plotting_variable not in plot_data:
                    plot_data[plotting_variable] = []
                plot_data[plotting_variable].extend(time_normalized_series)

            elif 'muscle_activations' in plotting_variable:

                ### Also compute R2 values for each surrogate result on unsegmented data
                valid_timesteps = plot_data_trial[plotting_variable]['ground_truth'].shape[0]



                R2['SST']  = np.concatenate([R2['SST'],
                                             plot_data_trial[plotting_variable]['ground_truth'][:valid_timesteps]],axis=0)
                R2['total_predictions'] += valid_timesteps


                ### Update plottin data for muscle activations
                for muscle_activation in plot_data_trial[plotting_variable]:
                    assert len(plot_data_trial[plotting_variable][muscle_activation].shape) == 2, "Data should be 2D"

                    surrogate_timesteps = min([plot_data_trial[plotting_variable][muscle_activation].shape[0],valid_timesteps])

                    SSE = (plot_data_trial[plotting_variable][muscle_activation][:surrogate_timesteps] - plot_data_trial[plotting_variable]['ground_truth'][:surrogate_timesteps])**2
                    R2['SSE'][muscle_activation] = np.concatenate([R2['SSE'][muscle_activation],SSE],axis=0)

                    time_normalized_series = [time_normalization(plot_data_trial[plotting_variable][muscle_activation][segment[0]:segment[1]]) for segment in segments if segment[1] > segment[0] ] 
                    if plotting_variable not in plot_data:
                        plot_data[plotting_variable] = {}
                        
                    if muscle_activation not in plot_data[plotting_variable]:
                        plot_data[plotting_variable][muscle_activation] = []

                    plot_data[plotting_variable][muscle_activation].extend(time_normalized_series)







    if len(plot_data) == 0: # Tracks are empty  
        continue
    
    for plotting_variable in plot_data: 
        fig_title = f"{plotting_variable} for Subject:{PPE_Subjects[subject_name]}"
        if plotting_variable == 'kinematics' or plotting_variable == 'kinetics':
            plot_data[plotting_variable] = np.array(plot_data[plotting_variable]).transpose((2,0,1))
            fig = plot_simulation_data(plot_headers, plot_data[plotting_variable], fig_title, visualize=False , data_type=plotting_variable)
        elif 'muscle_activations' in plotting_variable:
            for muscle_activation in plot_data[plotting_variable]:
                plot_data[plotting_variable][muscle_activation] = np.array(plot_data[plotting_variable][muscle_activation]).transpose((2,0,1))

            mc_page_index = int(plotting_variable.split('-')[-1])
            plot_mc_headers = plot_muscle_activations_mapping_pages[mc_page_index]

            fig = plot_simulation_data(plot_mc_headers, plot_data[plotting_variable], fig_title, visualize=False , data_type=plotting_variable,num_cols = 4)

        else: 
            raise ValueError(f"Unknown plotting variable:{plotting_variable}")        
        plotly.io.write_image(fig, os.path.join(pdf_dir, f'{PPE_Subjects[subject_name]}_{plotting_variable}.pdf'), format='pdf')

        if plotting_variable == 'kinematics' or plotting_variable == 'kinetics':
            if not np.isnan(plot_data[plotting_variable]).any(): 
                aggregate_data[plotting_variable]['mean'] += plot_data[plotting_variable].sum(axis=1)
                aggregate_data[plotting_variable]['std'] += (plot_data[plotting_variable]**2).sum(axis=1)
                
                mcs_score = mcs_scores[subject_name]

                # if plotting_variable not in mcs_aggregate_data[mcs_score]: 
                #     mcs_aggregate_data[mcs_score][plotting_variable] = {} 
                    
                #     if 'muscle_activations' in plotting_variable:
                #         mc_page_index = int(plotting_variable.split('-')[-1])
                #         mcs_aggregate_data[mcs_score][plotting_variable]['mean'] = np.zeros((len(plot_muscle_activations_mapping_pages[mc_page_index]),101))
                #         mcs_aggregate_data[mcs_score][plotting_variable]['std'] = np.zeros((len(plot_muscle_activations_mapping_pages[mc_page_index]),101))
                #         mcs_aggregate_data[mcs_score][plotting_variable]['list'] = np.zeros((0,len(plot_muscle_activations_mapping_pages[mc_page_index]),101))
                #     elif plotting_variable == 'kinematics' or plotting_variable == 'kinetics': 
                #         mcs_aggregate_data[mcs_score][plotting_variable]['mean'] = np.zeros((len(plot_names_mapping),101))
                #         mcs_aggregate_data[mcs_score][plotting_variable]['std'] = np.zeros((len(plot_names_mapping),101))
                #         mcs_aggregate_data[mcs_score][plotting_variable]['list'] = np.zeros((0,len(plot_names_mapping),101))

                #     else: 
                #         raise ValueError(f"Unknown plotting variable:{plotting_variable}")

                #     mcs_aggregate_data[mcs_score][plotting_variable]['ppe_names'] = []
                #     mcs_aggregate_data[mcs_score][plotting_variable]['ppe_trial'] = []
                #     mcs_aggregate_data[mcs_score][plotting_variable]['total_trials'] = 0
                
                mcs_aggregate_data[mcs_score][plotting_variable]['mean'] += plot_data[plotting_variable].sum(axis=1)
                mcs_aggregate_data[mcs_score][plotting_variable]['std'] += (plot_data[plotting_variable]**2).sum(axis=1)

                mcs_aggregate_data[mcs_score][plotting_variable]['list'] = np.concatenate([ mcs_aggregate_data[mcs_score][plotting_variable]['list'],
                                                                                        np.transpose(plot_data[plotting_variable], (1,0,2)   ) ])
                
                mcs_aggregate_data[mcs_score][plotting_variable]['ppe_names'].extend([PPE_Subjects[subject_name]]*plot_data[plotting_variable].shape[1])
                # mcs_aggregate_data[mcs_score][plotting_variable]['ppe_trial'].extend([trial_index]*plot_data[plotting_variable].shape[1])
                mcs_aggregate_data[mcs_score][plotting_variable]['total_trials'] += plot_data[plotting_variable].shape[1]
        
        elif 'muscle_activations' in plotting_variable:
            for muscle_activations in plot_data[plotting_variable]:
                if not np.isnan(plot_data[plotting_variable][muscle_activations]).any():
                    aggregate_data[plotting_variable][muscle_activations]['mean'] += plot_data[plotting_variable][muscle_activations].sum(axis=1)
                    aggregate_data[plotting_variable][muscle_activations]['std'] += (plot_data[plotting_variable][muscle_activations]**2).sum(axis=1)

                    mcs_score = mcs_scores[subject_name]

                    mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['mean'] += plot_data[plotting_variable][muscle_activations].sum(axis=1)
                    mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['std'] += (plot_data[plotting_variable][muscle_activations]**2).sum(axis=1)

                    mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['list'] = np.concatenate([ mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['list'],
                                                                                        np.transpose(plot_data[plotting_variable][muscle_activations], (1,0,2)   ) ])

                    mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['ppe_names'].extend([PPE_Subjects[subject_name]]*plot_data[plotting_variable][muscle_activations].shape[1])
                    # mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['ppe_trial'].extend([trial_index]*plot_data[plotting_variable][muscle_activations].shape[1])
                    mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['total_trials'] += plot_data[plotting_variable][muscle_activations].shape[1]

    total_trials += plot_data['kinematics'].shape[1]
            
        
        



for k in aggregate_data:
    if k == 'kinematics' or k == 'kinetics':
        aggregate_data[k]['mean'] /= total_trials
        aggregate_data[k]['std'] = np.sqrt(aggregate_data[k]['std']/total_trials - aggregate_data[k]['mean']**2)
    elif 'muscle_activations' in k:
        for muscle_activation in aggregate_data[k]:
            aggregate_data[k][muscle_activation]['mean'] /= total_trials
            aggregate_data[k][muscle_activation]['std'] = np.sqrt(aggregate_data[k][muscle_activation]['std']/total_trials - aggregate_data[k][muscle_activation]['mean']**2)

    # break

for mcs_score in mcs_aggregate_data:
    for plotting_variable in mcs_aggregate_data[mcs_score]: 
        if plotting_variable == 'kinematics' or plotting_variable == 'kinetics':
            total_trials = mcs_aggregate_data[mcs_score][plotting_variable]['total_trials']

            assert mcs_aggregate_data[mcs_score][plotting_variable]['list'].shape[0] == total_trials
            if total_trials == 0: 
                continue

            mcs_aggregate_data[mcs_score][plotting_variable]['mean'] /= total_trials
            mcs_aggregate_data[mcs_score][plotting_variable]['std'] = np.sqrt(mcs_aggregate_data[mcs_score][plotting_variable]['std']/total_trials - mcs_aggregate_data[mcs_score][plotting_variable]['mean']**2)
        elif 'muscle_activations' in plotting_variable:            
            for muscle_activation in mcs_aggregate_data[mcs_score][plotting_variable]:
                
                total_trials = mcs_aggregate_data[mcs_score][plotting_variable][muscle_activation]['total_trials']
                            
                if total_trials == 0: 
                    continue

                assert mcs_aggregate_data[mcs_score][plotting_variable][muscle_activations]['list'].shape[0] == total_trials



                mcs_aggregate_data[mcs_score][plotting_variable][muscle_activation]['mean'] /= total_trials
                mcs_aggregate_data[mcs_score][plotting_variable][muscle_activation]['std'] = np.sqrt(mcs_aggregate_data[mcs_score][plotting_variable][muscle_activation]['std']/total_trials - mcs_aggregate_data[mcs_score][plotting_variable][muscle_activation]['mean']**2)


############# Compute R2 values for each surrogate result
R2_mean = np.mean(R2['SST'],axis=0,keepdims=True)
R2_SST = np.sum( (R2['SST'] - R2_mean)**2  ,axis=0)

R2['R2'] = {}
R2['RMSE'] = {}
for surrogate_result in R2['SSE']:
    R2['SSE'][surrogate_result] = np.sum(R2['SSE'][surrogate_result],axis=0)
    R2['R2'][surrogate_result] = 1 - R2['SSE'][surrogate_result]/R2_SST

    R2['RMSE'][surrogate_result] = np.sqrt(R2['SSE'][surrogate_result]/R2['total_predictions'])

    results = list(zip(page_dict.values(),R2['R2'][surrogate_result]))
    print(f"R2 Surrogate:{surrogate_result}")
    for muscle in results:
        print(f"{muscle[0]}:{muscle[1]:.3f}")


mcs_scores
plot_data.keys()

assert plot_headers is not None, "Something went wrong. Headers should not be None"

from matplotlib.pyplot import plot


def plot_data_distribution(headers,plot_data_mean,plot_data_std,title_text="Plot Data",data_type="kinematics",num_cols=4, visualize=False): 
    
    # assert plot_data.shape[-1] == 101, "Length of data should be 101"
    num_rows = int(np.ceil(len(headers)/num_cols))


    if data_type == 'kinematics' or data_type == 'kinetics':
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[ plot_names_mapping[header] + ( ' moments' if data_type == 'kinetics' else '' )  for header in headers]) 
    elif  'muscle_activations' in data_type:
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_muscle_activations_mapping[header] for header in headers])
    else: 
        raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")


    # Colors for left and right sides
    colors = {'left': 'blue', 'right': 'red'}

    # Create each subplot
    for i, header in enumerate(headers):
        row = i // num_cols + 1
        col = i % num_cols + 1
        
        if i >= len(headers): 
            break


        if 'muscle_activations' in data_type:
            title = plot_muscle_activations_mapping[header]
        elif data_type == 'kinematics' or data_type == 'kinetics':
            title = plot_names_mapping[header]
        else:
            raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")

        # Plot every kinematics data
        x = np.linspace(0,1,num=plot_data_mean[i].shape[-1])

    
        fig.add_trace(go.Scatter(x=x, y=plot_data_mean[i], showlegend=False, name=f'{title}'), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], 
                                    y=list(plot_data_mean[i] + np.array(plot_data_std[i])) + list(np.array(plot_data_mean[i]) - np.array(plot_data_std[i]))[::-1] ,
                                        mode='lines', line=dict(width=0), name=f'{title} Bounds', showlegend=False, fill='toself',hoverinfo="skip",),
        row=row, col=col)

        # Update y-axis label
        if data_type == 'kinematics':
            fig.update_yaxes(title_text='deg', row=row, col=col)
        elif data_type == 'kinetics':
            fig.update_yaxes(title_text='Nm', row=row, col=col)
        elif 'muscle_activations' in data_type: 
            fig.update_yaxes(title_text='0-1', row=row, col=col)

        fig.update_xaxes(title_text='% SQT Cycle (Seconds)', row=row, col=col)

    

    # Update layout
    # plot_height = 1800 if 'muscle_activations' in data_type else 1000
    fig.update_layout(height=1000, width=2000,
                        showlegend=False,  title_x=0.5,
                        title_text=title_text,
                        font_family="Times New Roman",
                        font_color="black",
                        title_font_family="Times New Roman",
                        title_font_color="black")

    # Show the figure
    if visualize:
        fig.show()
    
    return fig

for k in aggregate_data:
    if k == 'kinematics' or k == 'kinetics':
        fig = plot_data_distribution(plot_headers, aggregate_data[k]['mean'],aggregate_data[k]['std'], f"Aggregate {k} Data",data_type=k)
        plotly.io.write_image(fig, os.path.join(pdf_dir, f'all_subject-{k}.pdf'), format='pdf')
    
    elif 'muscle_activations' in k:
        mc_page_index = int(k.split('-')[-1])
        for muscle_activation_name in aggregate_data[k]:
            fig = plot_data_distribution(plot_muscle_activations_mapping_pages[mc_page_index], aggregate_data[k][muscle_activation_name]['mean'],aggregate_data[k][muscle_activation_name]['std'], f"Aggregate {k} {muscle_activation_name} ",data_type=k)
            plotly.io.write_image(fig, os.path.join(pdf_dir, f'all_subject-{k}-{muscle_activations}.pdf'), format='pdf')
    else: 
        raise ValueError(f"Unknown plotting variable:{k}")

# Plot Per MCS data

def plot_mcs_distribution(headers,mcs_aggregate_data,title_text="Plot Data",data_type="kinematics", num_cols=4, visualize=False): 
    
    # assert plot_data.shape[-1] == 101, "Length of data should be 101"
    num_rows = int(np.ceil(len(headers)/num_cols))

    # Create 4x4 subplots (we'll only use 14 of them)
    if data_type == 'kinematics' or data_type == 'kinetics':
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_names_mapping[header] + ( ' moments' if data_type == 'kinetics' else '' ) for header in headers])
    elif  'muscle_activations' in data_type:
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_muscle_activations_mapping[header] for header in headers])
    else:
        raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")


    # Colors for left and right sides
    colors = {2: 'rgba(255, 0, 0, 0.2)', 3: 'rgba(0, 0, 255, 0.2)' , 4: 'rgba(0, 255, 0, 0.2)', -1: 'rgba(0, 0, 0, 0.2)', 0: 'rgba(0, 0, 0, 0.2)' }
    colors_mean = {2: 'rgba(255, 0, 0, 1.0)', 3: 'rgba(0, 0, 255, 1.0)' , 4: 'rgba(0, 255, 0, 1.0)', -1: 'rgba(0, 0, 0, 1.0)', 0: 'rgba(0, 0, 0, 1.0)'}

    y_max = 0 
    y_min = 0 

    # Create each subplot
    for i, header in enumerate(headers):
        row = i // num_cols + 1
        col = i % num_cols + 1
        
        if i >= len(headers): 
            break
        
        if 'muscle_activations' in data_type:
            title = plot_muscle_activations_mapping[header]
        elif data_type == 'kinematics' or data_type == 'kinetics':
            title = plot_names_mapping[header]
        else:
            raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")

        # Plot every kinematics data
        x = None

        for mcs_score in mcs_aggregate_data: 
            if len(mcs_aggregate_data[mcs_score]) == 0: 
                continue  
            plot_data_mean = mcs_aggregate_data[mcs_score][data_type]['mean']
            plot_data_std = mcs_aggregate_data[mcs_score][data_type]['std']
            # print(plot_data_mean.shape)
            if x is None: 
                x = np.linspace(0,1,num=plot_data_mean[i].shape[-1]) 

            fig.add_trace(go.Scatter(x=x, y=plot_data_mean[i], showlegend = (i==len(headers)-1), 
                                    name=f'MCS:{mcs_score}',line=dict(color=colors_mean[mcs_score])), row=row, col=col)
            fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], 
                                        y=list(plot_data_mean[i] + np.array(plot_data_std[i]/2)) + list(np.array(plot_data_mean[i]) - np.array(plot_data_std[i]/2))[::-1] ,
                                            mode='lines', line=dict(width=0), name=f'MCS:{mcs_score} Bounds', showlegend=False, fill='toself',hoverinfo="skip",fillcolor=colors[mcs_score]), row=row, col=col)

            y_min = min(y_min, np.min(plot_data_mean[i] - plot_data_std[i]/2))
            y_max = max(y_max, np.max(plot_data_mean[i] + plot_data_std[i]/2))
            
        # Update y-axis label
        if data_type == 'kinematics':
            fig.update_yaxes(title_text='deg', row=row, col=col)
        elif data_type == 'kinetics':
            fig.update_yaxes(title_text='Nm', row=row, col=col)
        else:
            fig.update_yaxes(title_text='0-1', row=row, col=col)
        # fig.update_yaxes(title_text='deg', row=row, col=col)

        fig.update_xaxes(title_text='% SQT Cycle (Seconds)', row=row, col=col)

    

    # Update layout
    fig.update_layout(height=1000, width=2000,
                        showlegend=True,  title_x=0.5,
                        title_text=title_text,
                        font_family="Times New Roman",
                        font_color="black",
                        title_font_family="Times New Roman",
                        title_font_color="black")

    # fig.update_yaxes(range=[y_min,y_max])
    
    # Show the figure
    if visualize: 
        fig.show()
    
    return fig

def plot_mcs_data(headers,mcs_aggregate_data,title_text="Plot Data",data_type="kinematics",num_cols=4, visualize=False, mcs_scores = None): 
    
    mcs_scores = mcs_aggregate_data.keys() if mcs_scores is None else mcs_scores

    # assert plot_data.shape[-1] == 101, "Length of data should be 101"
    num_rows = int(np.ceil(len(headers)/num_cols))

    # Create 4x4 subplots (we'll only use 14 of them)
    if data_type == 'kinematics' or data_type == 'kinetics':
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_names_mapping[header] + ( ' moments' if data_type == 'kinetics' else '' ) for header in headers])
    elif  'muscle_activations' in data_type:
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[plot_muscle_activations_mapping[header] for header in headers])
    else:
        raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")
    

    colors = {2: 'rgba(255, 0, 0, 0.2)', 3: 'rgba(0, 0, 255, 0.2)' , 4: 'rgba(0, 255, 0, 0.2)', -1: 'rgba(0, 0, 0, 0.2)', 0: 'rgba(0, 0, 0, 0.2)' }
    colors_mean = {2: 'rgba(255, 0, 0, 1.0)', 3: 'rgba(0, 0, 255, 1.0)' , 4: 'rgba(0, 255, 0, 1.0)', -1: 'rgba(0, 0, 0, 0.1)', 0: 'rgba(0, 0, 0, 0.2)'}
    # Colors for left and right sides
    # colors = {2: 'rgba(255, 0, 0, 0.2)', 3: 'rgba(0, 0, 255, 0.2)' , 4: 'rgba(0, 255, 0, 0.2)', -1: 'rgba(0, 0, 0, 0.2)' }
    # colors_mean = {2: 'rgba(255, 0, 0, 1.0)', 3: 'rgba(0, 0, 255, 1.0)' , 4: 'rgba(0, 255, 0, 1.0)', -1: 'rgba(0, 0, 0, 1.0)'}

    y_min = 0 # Max value for y-axis
    y_max = 0 # Max value for y-axis

    # Create each subplot
    for i, header in enumerate(headers):
        row = i // num_cols + 1
        col = i % num_cols + 1
        
        if i >= len(headers): 
            break   
        
        if 'muscle_activations' in data_type:
            title = plot_muscle_activations_mapping[header]
        elif data_type == 'kinematics' or data_type == 'kinetics':
            title = plot_names_mapping[header]
        else: 
            raise ValueError("Invalid data type. Should be either kinematics or kinetics or muscle_activations")

        # Plot every kinematics data
        x = None

        for mcs_score in mcs_scores: 
            
            if data_type not in mcs_aggregate_data[mcs_score]: 
                continue 
            for y_ind, y in enumerate(mcs_aggregate_data[mcs_score][data_type]['list']):

                if x is None: 
                    x = np.linspace(0,1,num=y.shape[-1])                            

                show_lengend = (i==len(headers)-1) and y_ind==0 
                if show_lengend:
                    plot_name = f"MCS:{mcs_score}"
                else: 
                    plot_name = f'{mcs_aggregate_data[mcs_score][data_type]["ppe_names"][y_ind]}'

                fig.add_trace(go.Scatter(x=x, y=y[i], showlegend=show_lengend, 
                                        name=plot_name,line=dict(color=colors_mean[mcs_score])), row=row, col=col)
                y_min = min(y_min, y[i].min()) 
                y_max = max(y_max, y[i].max())
        # Update y-axis label
        if data_type == 'kinematics':
            fig.update_yaxes(title_text='deg', title_standoff=0, row=row, col=col)
        elif data_type == 'kinetics':
            fig.update_yaxes(title_text='Nm', title_standoff=0, row=row, col=col)
        else:
            fig.update_yaxes(title_text='0-1', title_standoff=0, row=row, col=col)

        fig.update_xaxes(title_text='% SQT Cycle (Seconds)', row=row, col=col)



    # Update layout
    # print("Setting y-axis range:",y_min,y_max)
    # fig.update_yaxes(range=[y_min, y_max])
    
    fig.update_layout(height=1000, width=2000,
                        showlegend=True,  title_x=0.5,
                        title_text=title_text,
                        font_family="Times New Roman",
                        font_color="black",
                        title_font_family="Times New Roman",
                        title_font_color="black",
                        )
    

    # Show the figure
    if visualize:
        fig.show()
    
    return fig



for k in ['kinematics','kinetics']:

    fig = plot_mcs_distribution(plot_headers, mcs_aggregate_data, f"MCS {k} Distributions",data_type=k)
    plotly.io.write_image(fig, os.path.join(pdf_dir, f'mcs_subject_{k}_distribution.pdf'), format='pdf')

    fig = plot_mcs_data(plot_headers, mcs_aggregate_data, f"MCS {k}",data_type=k,visualize=False)
    plotly.io.write_image(fig, os.path.join(pdf_dir, f'mcs_subject_{k}.pdf'), format='pdf')
    
for page_index,ma_page in enumerate(plot_muscle_activations_mapping_pages):
    k = f'muscle_activations-{page_index}'

    for muscle_activation_name in ['ground_truth'] + [ os.path.basename(surrogate_result) for surrogate_result in surrogate_results_list]:
        
        muscle_mcs_aggregate_data = {mcs_score: { k: mcs_aggregate_data[mcs_score][k][muscle_activation_name].copy()} for mcs_score in mcs_aggregate_data}

        fig = plot_mcs_distribution(ma_page.keys(), muscle_mcs_aggregate_data, f"MCS {muscle_activation_name} Distributions",data_type=k)
        plotly.io.write_image(fig, os.path.join(pdf_dir, f'mcs_subject_{k}_{muscle_activation_name}_distribution.pdf'), format='pdf')


        fig = plot_mcs_data(ma_page.keys(), muscle_mcs_aggregate_data, f"MCS {muscle_activation_name}",data_type=k,visualize=False, mcs_scores=[x for x in mcs_aggregate_data.keys() if x > 1])
        plotly.io.write_image(fig, os.path.join(pdf_dir, f'mcs_subject_{k}_{muscle_activation_name}.pdf'), format='pdf')


mcs_aggregate_data[mcs_score].keys()



from compile_report import pdf_compiler


pdf_compiler(pdf_path=pdf_dir, report_name=report_name, PPE_Subjects=PPE_Subjects, mcs_scores=mcs_scores, isMCS=args.isMCS)

# Plot the subject info
for mcs_score in mcs_aggregate_data:
    data_type = 'kinematics'
    if data_type not in mcs_aggregate_data[mcs_score]:
        print(mcs_score, {}, 0)
    else:
        subjects_set = set(mcs_aggregate_data[mcs_score][data_type]["ppe_names"])
        print(mcs_score, sorted(subjects_set), len(mcs_aggregate_data[mcs_score][data_type]["ppe_names"]))



for surrogate_result in R2['SSE']:
    results = list(zip(page_dict.values(),R2['R2'][surrogate_result],R2['RMSE'][surrogate_result]))
    print(f"Metrics: Surrogate:{surrogate_result}, R2, RMSE")
    for muscle in results:
        print(f"{muscle[0]},{muscle[1]:.3f},{muscle[2]:.3f}")