
# Temporal Segmentation
import math
import plotly
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Temporal Segmentation
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from scipy.interpolate import CubicSpline


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


def find_best_n_segments(temporal_segmentation_data, num_segments, duplicate_threshold=0.75, rms_threshold=0.75):
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
        print(f"Start-Stop Candidates for {trial}",valleys)
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

            if np.min(rms_segments) < rms_sequence*rms_threshold/num_segments:
                continue

            print("Testing segments",segments,"RMS Sequence:", rms_sequence,"RMS Threshold:",rms_sequence*rms_threshold/num_segments)
            print(rms_segments)

            if 'valid_segments' not in temporal_segmentation_data[trial]: 
                temporal_segmentation_data[trial]['valid_segments'] = []
            
            temporal_segmentation_data[trial]['valid_segments'].append(segments)
    
    
    def compute_combination_score(combination): 
        
        print(f"Computing combination score:{combination}")
        time_series = []
        for trial in combination: 
            segments = combination[trial]   
            max_pose_velocity = temporal_segmentation_data[trial]['max_pose_velocity']
            # Normalize Time Series
            time_normalized_series = np.array([time_normalization(max_pose_velocity[segment[0]:segment[1]]) for segment in segments])
            time_series.append(time_normalized_series)
        # Minimize STD at for entire duration
        combination_score = np.sum(np.std(time_series,axis=0))
        
        return combination_score        
    
    # Sort the trial names in data
    trials_names = sorted([k for k in temporal_segmentation_data], key=lambda x : int(x.split('_')[-1]))
    
    # Run recursion on all trials and return the combinations with the best matching score acrross trials. on all valid combinations. If num_combiations 
    def dfs(ind, combination):
        if ind == len(temporal_segmentation_data): 
            combination_score = compute_combination_score(combination)
            return combination, combination_score
            
        trial_name = trials_names[ind] 
        if 'valid_segments' not in temporal_segmentation_data[trial_name] or len(temporal_segmentation_data[trial_name]['valid_segments']) == 0:
            return dfs(ind+1, combination)
        
        print(f"{ind} combinations:{combinations}")
        
        best_combination_score = np.inf
        best_combination = None
        for trial_combination in temporal_segmentation_data[trial_name]['valid_segments']: 
            combination[trial_name] = trial_combination
            combination, combination_score = dfs(ind+1, combination)
            
            
            if combination_score < best_combination_score: 
                best_combination_score = combination_score
                best_combination = combination.copy()
        
            del combination[trial_name] # Remove combination, try another combination
        
        print(f"Best Score for: {ind} {best_combination_score} best_combination:{best_combination} ")
        
        return best_combination, best_combination_score
                     
    best_combination, best_combination_score = dfs(0, {}) 
    
    if np.isinf(best_combination_score):
        print("Could not find a valid segments. All combinations below RMS threshold") 
        return {}, np.inf
        
    return best_combination,best_combination_score


def temporal_segementation(data,headers, seconds_per_frame=0.01,visualize=False,num_segments=5,allowed_height_difference_threshold=0.1,fig_title=None,isdeg=True):
    """
        Find segments in the angular velocity of a time series of rotation vectors.
        params:
            data: A dictionary containing the pose_params

        returns: 
            segments: A list of tuples containing the start and end indices of the segments.
    """
    trials_names = sorted([k for k in data], key=lambda x : int(x.split('_')[-1])) 
 
    # Create subplots
    fig = plotly.subplots.make_subplots(rows=2, cols=len(data), subplot_titles=[ f"{trial_index}. Start-Stop:{trial_name} " for trial_index, trial_name in enumerate(trials_names)])
    
    temporal_segmentation_data = {} 
    
    # Sort the trial names in data


    for trial_index, trial_name in enumerate(trials_names):
        
        print(f"Finding valleys in For {trial_index}:{trial_name}")
        
        pose_velocity = data[trial_name]['kinematics']
        
        print(f"  Time series: {pose_velocity.shape}")

        
        # Smoothing Filter ## Note window length and polyorder should be adjusted based on the data
        for i in range(len(headers)): 
            pose_velocity[:,i] = savgol_filter(pose_velocity[:,i], window_length=21, polyorder=3)

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
                name=headers[i],
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
        
    segments_all_trial,segment_score = find_best_n_segments(temporal_segmentation_data,num_segments=num_segments)
    
            
    for trial_index, trial_name in enumerate(trials_names):

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
    fig.update_layout(width=2000, height=500)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Max Knee Flexion (deg)", row=1, col=1)


    if visualize: 
        # Show the figure
        fig.show()

    # fig.write_image(image_path + "_angular.png")
    return fig, segments_all_trial