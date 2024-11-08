import os
import math
import tqdm
import numpy as np 
import pandas as pd

# For plotting 
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Temporal Segmentation
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline


# Internal Modules 
from utils import DATA_DIR, set_logger
from opencap_dataloader import OpenCapLoader
from dataloader import OpenCapDataLoader as OldDataLoader

logger = set_logger(task_name="Temporal Segmentation")

def time_normalization(time_series,duration=101): 

    orig_time_space = np.linspace(0,1,len(time_series))
        
    spline = CubicSpline(orig_time_space, time_series)

    spline_input = np.linspace(0,1,duration)
    split_output = spline(spline_input)
            
    return split_output

def find_valleys_in_max_angular_velocity(max_angular_velocity,seconds_per_frame=0.01,allowed_height_difference_threshold=0.1):
    """
        Find peaks in the angular velocity of a time series of rotation vectors.

        Args:
        angular_velocity: A numpy array of shape (N, 24) containing angular velocities in radians per second.
        framerate: The frame rate of the data in Hz.

        Returns: A numpy array of shape (M,) containing the indices of the peaks in the angular velocity.
    """

    diff = np.max(max_angular_velocity) - np.min(max_angular_velocity)
    min_height = np.min(max_angular_velocity) + allowed_height_difference_threshold*diff

    distance_between_valleys = max(1,int(1/(10*seconds_per_frame)))
    # distance_between_valleys = 1

    print(f"distance between valleys={distance_between_valleys}")
    print(f"Max allowed valley height={min_height} allowed_height_difference_threshold={allowed_height_difference_threshold}")
    valleys, meta_data = find_peaks(-max_angular_velocity,distance=distance_between_valleys,height=-min_height)

    print("Valleys",valleys, "Meta data: ",meta_data)

    return valleys

def sort_trial_names(data):
    """
        Sort the trial names in the data
    """
    # If already segmented using dtw
    if all(['_segment_' in trial_name for trial_name in data]): 
        return sorted([k for k in data], key=lambda x : int(x.split('_')[-1]))
    else:
        return sorted([k for k in data], key=lambda x : int(x.lower().replace('sqt','')))


def find_best_n_segments(temporal_segmentation_data, num_segments, duplicate_threshold=0.75, rms_threshold=0.75,seconds_per_frame=0.01):
    """
        Validate segments using DTW

    """

    for trial in temporal_segmentation_data:

        max_pose_velocity = temporal_segmentation_data[trial]['max_pose_velocity']
        valleys = temporal_segmentation_data[trial]['change_points']
        

        if len(valleys) < num_segments -1 : continue

        if len(valleys) > 25: 
            valleys = np.sort(np.random.choice(valleys,25,replace=False))

        # Add first and last frame for completeness
        valleys = [0] + list(valleys) + [len(max_pose_velocity)-1]
        valleys = np.array(valleys)

        # In some cases there could a single valley which defines the start and the stop cycle. 
        # If that is the case, we need to duplicate the valley
        # Check if a particular valley is too far to the adjacent valleys. If so, duplicate it.
        # If the distance between the two adjacent valleys is greater than 1/2*average segment duaration, duplicate the valley 
        valley_copys = []
        valley_threshold = (len(max_pose_velocity)/num_segments)*duplicate_threshold
        for i,v in enumerate(valleys):
            if i == 0: continue
            if i == len(valleys)-1: continue 


            if (valleys[i+1] - valleys[i])  > valley_threshold and (valleys[i] - valleys[i-1])  > valley_threshold: 
                valley_copys.append(valleys[i])
        
        valleys = np.concatenate([valleys,valley_copys]).astype(int)
        valleys.sort()


        # Start-Stop candidates
        print(f"Start-Stop Candidates for {trial_name}",valleys)
        temporal_segmentation_data[trial]['change_points'] = valleys

    # from tslearn.barycenters import dtw_barycenter_averaging
    from itertools import combinations

    best_combination = {} 
    best_combination_score = np.inf
    best_combination_data = ()


    ## Hard tests for filtet out invalid combinations 
    rms_sequence = np.sqrt(np.sum(max_pose_velocity**2)) # RMS for entire sequence. If some segment has motion less than the 1/2*num_segments, probably no motion is happening in it.  
    
    for trial in temporal_segmentation_data:
        valleys = temporal_segmentation_data[trial]['change_points']
        max_pose_velocity = temporal_segmentation_data[trial]['max_pose_velocity']
        
        for cur_comb in combinations(valleys, 2*num_segments):
        
            segments = [    [cur_comb[i*2 + 0],cur_comb[i*2 + 1]] for i in range(num_segments)    ]

            rms_segments = [np.sqrt(np.sum(max_pose_velocity[segment[0]:segment[1]]**2)) for segment in segments]


            # print(f"Testing segments:{trial}",segments,"RMS Sequence:", rms_sequence,"RMS Threshold:",rms_sequence*rms_threshold/num_segments)
            # print(rms_segments)

            if np.min(rms_segments) < rms_sequence*rms_threshold/num_segments:
                continue


            if 'valid_segments' not in temporal_segmentation_data[trial]: 
                temporal_segmentation_data[trial]['valid_segments'] = []
            
            temporal_segmentation_data[trial]['valid_segments'].append(segments)
    
    
    def compute_combination_score(combination): 
        
        # print(f"Computing combination score:{combination}")
        time_series = []
        avg_time = []
        for trial in combination: 
            segments = combination[trial]   
            max_pose_velocity = temporal_segmentation_data[trial]['max_pose_velocity']
            # Normalize Time Series
            time_normalized_series = np.array([time_normalization(max_pose_velocity[segment[0]:segment[1]]) for segment in segments])
            time_series.append(time_normalized_series)
            avg_time.append(np.mean([segment[1] - segment[0] for segment in segments])*seconds_per_frame)
        # Minimize STD at for entire duration
        combination_score = np.sum(np.std(time_series,axis=0)) + max(0, np.mean(avg_time) - 2)*1000 # Penalize for longer than 2 duration
        
        return combination_score        
    
    # Sort the trial names in data
    trials_names = sort_trial_names([k for k in temporal_segmentation_data]) 
    # Run recursion on all trials and return the combinations with the best matching score acrross trials. on all valid combinations. If num_combiations 
    def dfs(ind, combination):
        if ind == len(temporal_segmentation_data): 
            combination_score = compute_combination_score(combination)
            return combination, combination_score
            
        trial_name = trials_names[ind] 
        if 'valid_segments' not in temporal_segmentation_data[trial_name] or len(temporal_segmentation_data[trial_name]['valid_segments']) == 0:
            return dfs(ind+1, combination)
        
        # print(f"{ind} combinations:{combinations}")
        
        best_combination_score = np.inf
        best_combination = None
        for trial_combination in temporal_segmentation_data[trial_name]['valid_segments']: 
            combination[trial_name] = trial_combination
            combination, combination_score = dfs(ind+1, combination)
            
            
            if combination_score < best_combination_score: 
                best_combination_score = combination_score
                best_combination = combination.copy()
        
            del combination[trial_name] # Remove combination, try another combination
        
        # print(f"Best Score for: {ind} {best_combination_score} best_combination:{best_combination} ")
        
        return best_combination, best_combination_score
                     
    best_combination, best_combination_score = dfs(0, {}) 
    print(f"Best Score for: {best_combination_score} best_combination:{best_combination} ")
    
    if np.isinf(best_combination_score):
        print("Could not find a valid segments. All combinations below RMS threshold") 
        return {}, np.inf
        
    return best_combination,best_combination_score


def manual_segments(sessio_id, trial_name, segment_id):
        
    # Check and update the first set of keys
    if "3d1207bf-192b-486a-b509-d11ca90851d7" in manual_segments:
        if "SQT01_segment_3" in manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]:
            manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_3"][0][0] += 15
            manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_3"][0][1] += 15

        # if "SQT01_segment_1" in manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]:
        #     manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_1"][0][0] += 40
        #     manual_segments["3d1207bf-192b-486a-b509-d11ca90851d7"]["SQT01_segment_1"][0][1] += 40

    # Check and update the second set of keys
    if "2345d831-6038-412e-84a9-971bc04da597" in manual_segments:
        if "SQT01_segment_1" in manual_segments["2345d831-6038-412e-84a9-971bc04da597"]:
            manual_segments["2345d831-6038-412e-84a9-971bc04da597"]["SQT01_segment_1"][0][0] += 40 


def temporal_segementation(data,headers, seconds_per_frame=0.01,visualize=True,num_segments=5,allowed_height_difference_threshold=0.1,fig_title=None,isdeg=True):
    """
        Find segments in the angular velocity of a time series of rotation vectors.
        params:
            data: A dictionary containing the pose_params

        returns: 
            segments: A list of tuples containing the start and end indices of the segments.
    """

    # Sort trial names for the plot is easier to read

    trials_names = sort_trial_names(data)


    # Create subplots
    fig = plotly.subplots.make_subplots(rows=2, cols=len(data), subplot_titles=[ f"{trial_index}. Start-Stop:{trial_name} " for trial_index, trial_name in enumerate(trials_names)])
    
    temporal_segmentation_data = {} 
    # Sort the trial names in data
    for trial_index, trial_name in enumerate(trials_names):
        
        print(f"Finding valleys in For {trial_index}:{trial_name}")
        
        pose_velocity = data[trial_name].copy()
        
        print(f"  Time series: {pose_velocity.shape}")

        assert len(headers) == pose_velocity.shape[1], f"Num Headers:{len(headers)} and pose_velocity:{pose_velocity.shape} do not match"

        # Smoothing Filter ## Note window length and polyorder should be adjusted based on the data
        for i in range(len(headers)): 
            pose_velocity[:,i] = savgol_filter(pose_velocity[:,i], window_length=21, polyorder=3)

        # pose_velocity = np.diff(pose_velocity,axis=0)
        # pose_velocity = np.concatenate([pose_velocity[0:1],pose_velocity],axis=0)

        # Only consider knee joints kinematics
        knee_indices = [i for i,header in enumerate(headers) if "knee" in header.lower()]
        max_pose_velocity = np.max(pose_velocity[:,knee_indices],axis=1)

        x = np.arange(len(max_pose_velocity))*seconds_per_frame

        if not isdeg:
            max_pose_velocity = np.rad2deg(max_pose_velocity)

        for i in range(len(headers)): 
            fig.add_trace(go.Scatter(
                x=x,
                y=pose_velocity[:,i],
                mode='lines',
                name=plot_headers[i],
                showlegend=True),row=1,col=trial_index+1)

        fig.add_trace(go.Scatter(
            x=x,
            y=max_pose_velocity,
            mode='lines',
            name='Knee Kinematics',
            showlegend=False
        ),row=2,col=trial_index+1)
        
        change_points = find_valleys_in_max_angular_velocity(max_pose_velocity,seconds_per_frame=seconds_per_frame,allowed_height_difference_threshold=allowed_height_difference_threshold)
        
        fig.add_trace(go.Scatter(
            x=x[change_points],
            y=max_pose_velocity[change_points],
            mode='markers',
            marker=dict(
                color='red',
                size=16,
                symbol='arrow-up'
            ),
            name='change_points'
        ),row=2,col=trial_index+1)
            
        temporal_segmentation_data[trial_name] = {'change_points':change_points, 'max_pose_velocity': max_pose_velocity}
    
    # fig.show()
        
    segments_all_trial,segment_score = find_best_n_segments(temporal_segmentation_data,num_segments=num_segments,seconds_per_frame=seconds_per_frame)
    
    if len(segments_all_trial) != len(trials_names):
        logger.warning(f"Some trials are invalid.  match. Segments:{segments_all_trial.keys()} Trials:{trials_names}")

    for trial_index, trial_name in enumerate(trials_names):

        if trial_name not in segments_all_trial: 
            continue

        change_points = temporal_segmentation_data[trial_name]['change_points']
        max_pose_velocity = temporal_segmentation_data[trial_name]['max_pose_velocity']
        segments = np.array(segments_all_trial[trial_name])

        print(segments,trial_name)

        x = np.arange(len(max_pose_velocity))*seconds_per_frame

        if not isdeg:
            max_pose_velocity = np.rad2deg(max_pose_velocity)
        
        # Plot the line segments 
        if not np.isinf(segment_score): 
            empty_array = np.array([None]*segments.shape[0]).reshape((-1,1))
            plot_y_segments = np.tile( np.max(max_pose_velocity).reshape((1,1)), segments.shape )
            plot_y_segments = np.concatenate([plot_y_segments,empty_array],axis=1).reshape(-1)

            plot_x_segments = np.concatenate([segments*seconds_per_frame,empty_array],axis=1).reshape(-1)

            fig.add_trace(go.Scatter(
                x=plot_x_segments,
                y=plot_y_segments,
                line_shape='linear',
                name='Selected Segments'
            ),row=2,col=trial_index+1)

        else: 
            segments = None


    # Update subplot titles
    fig.update_layout(
        title_text=fig_title if fig_title != '' or fig_title is not None else f'Temporal Segmentation using Knee Kinematics',
        font=dict(family="Times New Roman"),
    )
 
    # Set figure size
    fig.update_layout(width=1000, height=400)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Max Knee Flexion (deg)", row=1, col=1)


    if visualize: 
        # Show the figure
        fig.show()

    # fig.write_image(image_path + "_angular.png")


    return fig, segments_all_trial


if __name__ == "__main__": 


    skip_subjects = []
    # skip_subjects = ["c08f1d89-c843-4878-8406-b6f9798a558e","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","c28e768f-6e2b-4726-8919-c05b0af61e4a","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","349e4383-da38-4138-8371-9a5fed63a56a"]
    sub2sess_pd = pd.read_csv(os.path.join(DATA_DIR, 'subject2opencap.txt') ,sep=',')
    PPE_Subjects  = dict(zip( sub2sess_pd[' OpenCap-ID'].tolist(),sub2sess_pd['PPE'].tolist()  ))

    mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
    

    subjects = {}
    sessions = os.listdir(os.path.join(DATA_DIR, 'Data'))
    # sessions = ["2345d831-6038-412e-84a9-971bc04da597"]

    checked_subject_cnt = len(sessions)

    for subject_ind, subject_name in tqdm.tqdm(enumerate(sessions)):
        print(f"Checking Subject:{subject_ind} Name:{subject_name}")

        if subject_name in skip_subjects: continue
        if not os.path.isdir(os.path.join(DATA_DIR, 'Data',  subject_name)): continue
        

        print(f"Evaluating Id: {subject_ind} Name: {subject_name}")

        subject = OpenCapLoader(subject_name)
        subject_squat = subject.load_squat_trial_kinematics()
        
        if subject_squat is None: 
            continue

        subjects[subject_name] = subject_squat
        plot_headers = subject.mocap_headers
        print(f"Loaded data for:{subject_name} Trials:{subjects[subject_name].keys()} kinematics:{[ subjects[subject_name][trial_name].shape for trial_name in subjects[subject_name]] }")        
        
    
        ## If mcs files, then use dtw segmetation to start with 
        if subject_name in mcs_sessions:   
            for trial_name in subjects[subject_name]: 
                
                if subject_name == "349e4383-da38-4138-8371-9a5fed63a56a" and trial_name == "SQT01": 
                    continue
                
                
                label,recordAttempt_str,recordAttempt = OldDataLoader.get_label(trial_name + '.trc')

                dtw_segment_path = os.path.join( DATA_DIR,'DTW-Segmentation', subject_name + '_' + label + '_' + str(recordAttempt) + '.npy' )
                if not os.path.exists(dtw_segment_path): continue 

                segments_old = np.load(dtw_segment_path,allow_pickle=True).item()['segments']
                print("Segments Old",segments_old)
                
                unsegmented_plot_data  = subjects[subject_name][trial_name].to_numpy()
                plot_data = {} # Store in numpy format for easy access
                for segment_id, segment in enumerate(segments_old): 
                    plot_data['SQT' + str(segment_id)] = unsegmented_plot_data[segment[0]:segment[1]]
            
                seconds_per_frame = (subjects[subject_name][trial_name]['time'].iloc[-1] - subjects[subject_name][trial_name]['time'].iloc[0])/len(unsegmented_plot_data)

                fig_title = f"Temporal Segmentation using Knee Kinematics for Subject:{PPE_Subjects[subject_name]}"
                num_segments = 1  # Number of segments per trial
                # Temporal Segmentation (using knee angles kinematics since it gave the most reasonable results) 
                segments_fig, segments_all_trials = temporal_segementation(plot_data,plot_headers,\
                                                num_segments=num_segments, seconds_per_frame=seconds_per_frame,\
                                                allowed_height_difference_threshold=0.15,\
                                                isdeg=True,visualize=subject_ind > checked_subject_cnt,fig_title=fig_title)
                
                segments = []
                
                assert len(segments_all_trials) == len(segments_old), f"Segments:{segments_all_trials} Segments Old:{segments_old}" 

                # sort the keys 
                segment_name_sorted = sorted(segments_all_trials.keys(), key=lambda x: int(x.replace('SQT',''))) 

                for segment_id, segment in enumerate(segment_name_sorted): 
                    for k in range(len(segments_all_trials[segment])):
                        segments_all_trials[segment][k][0] += segments_old[segment_id][0]
                        segments_all_trials[segment][k][1] += segments_old[segment_id][0]
                    segments.extend(segments_all_trials[segment]) 
                
                segments_all_trials = {trial_name:segments}

        else: 
            plot_data = {} # Store in numpy format for easy access
            # Get average seconds per frame 
            seconds_per_frame = 0 
            # Filter out unecessary data
            for trial_name in subjects[subject_name]: 
                
                if '_segment_' in trial_name: continue # Skip already segmented data

                trial_length = subjects[subject_name][trial_name]['time'].iloc[-1] - subjects[subject_name][trial_name]['time'].iloc[0]
                if trial_length < 0.5: 
                    print(f"Subject:{subject_name} Trial {trial_name} is too short, skipping")
                    continue # Can't perform squat in less tha a second. 
                
                plot_data[trial_name] = subjects[subject_name][trial_name].to_numpy()

                seconds_per_frame += trial_length
            if seconds_per_frame == 0: 
                print("Tracks are empty, skipping subject")
                continue

            assert seconds_per_frame > 0, f"Subject Index:{subject_ind} seconds_per_frame should be greater 0. Likely no trial found to evaluate." 
            
            seconds_per_frame /= sum([len(plot_data[trial_name]) for trial_name in plot_data])


            fig_title = f"Temporal Segmentation using Knee Kinematics for Subject:{PPE_Subjects[subject_name]}"
            num_segments = 1  # Number of segments per trial
            # Temporal Segmentation (using knee angles kinematics since it gave the most reasonable results) 
            segments_fig, segments_all_trials = temporal_segementation(plot_data,plot_headers,\
                                            num_segments=num_segments, seconds_per_frame=seconds_per_frame,\
                                            allowed_height_difference_threshold=0.15,\
                                            isdeg=True,visualize=subject_ind > checked_subject_cnt,fig_title=fig_title)


        os.makedirs(os.path.join(DATA_DIR,"segmentation-pdfs"),exist_ok=True)
        plotly.io.write_image(segments_fig, os.path.join(DATA_DIR,"segmentation-pdfs",f"{PPE_Subjects[subject_name]}.pdf"), format='pdf')


        if len(segments_all_trials) == 0: 
            print(f"Could not find segments for subject:{subject_name}")
            continue  

        # Save the segmentation data
        os.makedirs(os.path.join(DATA_DIR,"squat-segmentation-data"),exist_ok=True)
        np.save(os.path.join(DATA_DIR,"squat-segmentation-data",f"{subject_name}.npy"),segments_all_trials)

        saved_data = np.load(os.path.join(DATA_DIR,"squat-segmentation-data",f"{subject_name}.npy"),allow_pickle=True).item()
        logger.info(f"Saved Segmentation for Subject:{subject_ind}:{subject_name} Segments:{saved_data}")