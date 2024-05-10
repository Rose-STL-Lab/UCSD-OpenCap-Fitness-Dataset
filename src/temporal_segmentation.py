import os
import sys
import joblib
import numpy as np

from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R
from utils import smpl_joints,cuda,SMPL_DIR,RENDER_DIR,SEGMENT_DIR

import plotly.subplots
import plotly.graph_objects as go

# TODO not working (unable to load utils from model of digital coach)
def assert_retarget_valid_smpl_format(data): 
    """
        We used a custom module to retarget .trc file format. 
        It is important to ensure the same smpl model is used for training. 
        To validate the above we use the the Rotation2xyz to get the co-oridnates of the 3D joints.
        Lastly use polyscope to render video and confirm with the video in <DATA_DIR>/rendenrerd_videos for the same file  
    """

    import torch
    import polyscope as ps
    
    sys.path.append("/media/shubh/Elements/RoseYu/digital-coach-shubh")

    from model.smpl import SMPL


    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    pose = torch.from_numpy(data['pose_params']).reshape((1,-1,24,3)).permute(0,2,3,1)  
    pose = pose.to(device)
    joints = rotation2xyz(pose, mask=None, pose_rep="rotvec", translation=False, glob=True,
                         jointstype="smpl", vertstrans=False, betas=None, beta=0,
                         glob_rot=None, get_rotations_back=False)
    
    joints = joints.permute(0,3,1,2).squeeze(0).detach().cpu().numpy()


    parent_array = rotation2xyz.smpl_model.parents.detach().cpu().numpy()
    parent_array[0] = 0
    bone_array = np.array([[i,parent_array[i]] for i in range(24)])

    # Render using polyscope
    ps.init()
    ps_skeleton = ps.register_curve_network("SMPL Mesh", joints[0], bone_array)
    
    os.makedirs("/tmp/random", exist_ok=True)
    for i in range(joints.shape[0]): 
        ps_skeleton.update_node_positions(joints[i])
        ps.screenshot(f"/tmp/random/joints_{i}.png")
    os.system("ffmpeg -r 10 -i /tmp/random/joints_%d.png -vcodec mpeg4 -y /tmp/random/joints.mp4")
    
    ps.show()


def get_relative_rotation_vectors(rotation_vectors):
    """
        Calculate the relative rotation matrix between two rotation matrices.
        Args:
        rotation_matrix1: A numpy array of shape (3, 3) representing the first rotation matrix.
        rotation_matrix2: A numpy array of shape (3, 3) representing the second rotation matrix.

        Returns: A numpy array of shape (3, 3) representing the relative rotation matrix from rotation_matrix1 to rotation_matrix2.
    """ 

    T,J,D = rotation_vectors.shape

    rotation_vectors = rotation_vectors.reshape((T*J,D))
    rotation_matrices = R.from_rotvec(rotation_vectors).as_matrix()
    rotation_matrices = rotation_matrices.reshape((T,J,3,3))

    # Calculate relative rotation matrices
    relative_rotation_matrices = rotation_matrices[1:]@(rotation_matrices[:-1].transpose((0,1,3,2)))
    relative_rotation_matrices = relative_rotation_matrices.reshape(( (T-1)*J, 3, 3))
    
    # Convert relative rotation matrices to rotation vectors
    relative_rotation_vectors = R.from_matrix(relative_rotation_matrices).as_rotvec()
    relative_rotation_vectors = relative_rotation_vectors.reshape((T-1,J,3))

    return relative_rotation_vectors



def get_laplacian_rotation_vectors(rotation_vectors):
    """
        Calculate the relative rotation matrix between two rotation matrices.
        Args:
        rotation_matrix1: A numpy array of shape (3, 3) representing the first rotation matrix.
        rotation_matrix2: A numpy array of shape (3, 3) representing the second rotation matrix.

        Returns: A numpy array of shape (3, 3) representing the relative rotation matrix from rotation_matrix1 to rotation_matrix2.
    """ 

    T,J,D = rotation_vectors.shape

    rotation_vectors = rotation_vectors.reshape((T*J,D))
    rotation_matrices = R.from_rotvec(rotation_vectors).as_matrix()
    rotation_matrices = rotation_matrices.reshape((T,J,3,3))

    # Calculate relative rotation matrices
    padded_rotation_matrices = rotation_matrices[1:-1].transpose((0,1,3,2))
    laplacian_rotation_matrices = rotation_matrices[0:-2]@padded_rotation_matrices@padded_rotation_matrices@rotation_matrices[2:]
    laplacian_rotation_matrices = laplacian_rotation_matrices.reshape(( (T-2)*J, 3, 3))
    
    # Convert relative rotation matrices to rotation vectors
    laplacian_rotation_vectors = R.from_matrix(laplacian_rotation_matrices).as_rotvec()
    laplacian_rotation_vectors = laplacian_rotation_vectors.reshape((T-2,J,3))

    return laplacian_rotation_vectors


def calculate_angular_velocity(rotation_vectors,framerate=60):
    """
        To calculate the angular velocity from a time series of rotation vectors, you can follow these steps:

        Convert the rotation vectors to rotation matrices.
        Calculate the relative rotation matrix between each consecutive pair of rotation matrices.
        Convert the relative rotation matrices back to rotation vectors. 
        The magnitude of these vectors represents the angle of rotation between the two time steps.
        Divide the angle of rotation by the time difference between the two steps to get the angular velocity.

        Args:
        rotation_vectors: A numpy array of shape (N, 24, 3) containing rotation vectors.
        framerate: The frame rate of the data in Hz.

        Returns: A numpy array of shape (N-1, 24) containing angular velocities in radians per second.

    
        To test: 
            test_rotation = np.array([R.from_euler('x',0.01*i,degrees=True).as_rotvec() for i in range(100)]).reshape((-1,1,3))
            test_angular_velocity = calculate_angular_velocity(test_rotation,framerate=1)
            assert np.all(test_angular_velocity == 0.01) 
    """


    # Convert rotation vectors to rotation matrices
    
    assert len(rotation_vectors.shape) == 3 and rotation_vectors.shape[-1] == 3, "Invalid shape of rotation vectors. Must be of shape (N, J, 3)"  
    
    relative_rotation_vectors = get_relative_rotation_vectors(rotation_vectors) # For angular velocity calculation
    # relative_rotation_vectors = get_laplacian_rotation_vectors(rotation_vectors) # For angular acceleration calculation
    # Calculate angles of rotation
    relative_angles = np.linalg.norm(relative_rotation_vectors, axis=2)

    # Calculate time difference between 2 frames 
    dt = 1/framerate

    # Calculate angular velocities
    angular_velocities = relative_angles / dt

    # Convert to degrees to better readability

    return np.rad2deg(angular_velocities)


def find_valleys_in_max_angular_velocity(max_angular_velocity,framerate=60):
    """
        Find peaks in the angular velocity of a time series of rotation vectors.

        Args:
        angular_velocity: A numpy array of shape (N, 24) containing angular velocities in radians per second.
        framerate: The frame rate of the data in Hz.

        Returns: A numpy array of shape (M,) containing the indices of the peaks in the angular velocity.
    """

    diff = np.max(max_angular_velocity) - np.min(max_angular_velocity)
    min_height = np.min(max_angular_velocity) + 0.2*diff
    valleys, meta_data = find_peaks(-max_angular_velocity,distance=2*framerate,height=-min_height)

    print("Meta data: ",meta_data)

    return valleys 


def find_all_valleyes_in_max_angular_velocity(max_angular_velocity,framerate=60):
    """
        Find all the peaks and valleys in the angular velocity of a time series of rotation vectors.
        Then find longest common substring between the peaks and valleys to get the segments.

        Args:
        angular_velocity: A numpy array of shape (N, 24) containing angular velocities in radians per second.
        framerate: The frame rate of the data in Hz.

        Returns: A numpy array of shape (M,) containing the indices of the peaks in the angular velocity.
    """

    diff = np.max(max_angular_velocity) - np.min(max_angular_velocity)
    min_height = np.min(max_angular_velocity) + 0.2*diff
    valleys, meta_data = find_peaks(-max_angular_velocity,distance=20,height=-min_height)

    print("Meta data: ",meta_data, "Number of peaks:", len(valleys))
    
    return valleys 



def use_clasp(max_angular_velocity,framerate=60):
    """
        Use CLASP to find segments in the angular velocity of a time series of rotation vectors.

        Args:
        angular_velocity: A numpy array of shape (N, 24) containing angular velocities in radians per second.
        framerate: The frame rate of the data in Hz.

        Returns: A numpy array of shape (M,) containing the indices of the peaks in the angular velocity.
    """
    from claspy.segmentation import BinaryClaSPSegmentation
    import matplotlib.pyplot as plt
    
    clasp = BinaryClaSPSegmentation(n_segments=3,validation=None,window_size=25,distance="euclidean_distance"); 
    peaks = clasp.fit_predict(max_angular_velocity)
    clasp.plot()
    plt.show()
    return peaks

def get_segment_from_change_points(change_points,len_data,framerate=60):
    segments = []
    first = 0 
    for peak in change_points:
        if peak - first < framerate:
            continue 
        segments.append((first,peak))
        first = peak
    if len_data - first > framerate:
        segments.append((first,len_data))

    return np.array(segments)


def validate_segments(max_pose_velocity,segments,fig_path=None,visualize=False):
    """
        Validate segments using DTW

    """

    if len(segments) == 1: return np.array([0],dtype=int),np.array([np.inf])

    from tslearn.metrics import dtw_path
    # from tslearn.barycenters import dtw_barycenter_averaging

    S = segments.shape[0]

    dtw_cost_matrix = np.zeros((S,S)) 
    dtw_path_matrix = [ [ None for _ in range(S)] for _ in range(S)]

    for i,s1 in enumerate(segments):
        for j,s2 in enumerate(segments):
            if i > j:
                dtw_path_matrix[i][j],dtw_cost_matrix[i,j] = dtw_path(max_pose_velocity[s1[0]:s1[1]],max_pose_velocity[s2[0]:s2[1]])
                
                matrix_diagonal = np.sqrt((s1[1]-s1[0])**2 + (s2[1]-s2[0])**2) 

                dtw_cost_matrix[i,j] /= matrix_diagonal
                dtw_cost_matrix[j,i] = dtw_cost_matrix[i,j]

                dtw_path_matrix[i][j] = np.array(dtw_path_matrix[i][j])
                dtw_path_matrix[j][i] = dtw_path_matrix[i][j][:,[1,0]]




    # Rank segments based on the DTW cost matrix
    dtw_score = np.sum(dtw_cost_matrix,axis=1)
    ranked_segments = np.argsort(dtw_score)
    ranked_segments_score = dtw_score[ranked_segments]

    selected_segments = ranked_segments[:3] if len(ranked_segments) >= 3 else ranked_segments

    # fig = go.Figure()
    num_segments = len(selected_segments)
    fig = plotly.subplots.make_subplots(rows=((num_segments-1)*num_segments//2), cols=1,subplot_titles=[ str(i+1) for i in range( ((num_segments-1)*num_segments//2) )   ])
    subplot_titles = {}
    for i in range(num_segments):
        for j in range(num_segments):
            if i > j:
                # fig.add_trace(go.Scatter(x=dtw_path_matrix[selected_segments[i]][selected_segments[j]][:,0], y=dtw_path_matrix[selected_segments[i]][selected_segments[j]][:,1],
                                # line_shape='linear'),row=i,col=j)
                x_i = np.arange(segments[ranked_segments[i]][1]-segments[ranked_segments[i]][0]+1)/framerate
                y_i = max_pose_velocity[segments[ranked_segments[i]][0]:segments[ranked_segments[i]][1]]
                fig.add_trace(go.Scatter(x=x_i, y=y_i, name=f"Segmemt:{segments[ranked_segments[i]][0]/framerate:.2f}-{segments[ranked_segments[i]][1]/framerate:.2f}",
                                    line_shape='linear',
                                    ),row=i+j,col=1)
                
                x_j = np.arange(segments[ranked_segments[j]][1]-segments[ranked_segments[j]][0]+1)/framerate
                y_j = 180 + max_pose_velocity[segments[ranked_segments[j]][0]:segments[ranked_segments[j]][1]]
                fig.add_trace(go.Scatter(x=x_j, y=y_j, name=f"Segmemt:{segments[ranked_segments[j]][0]/framerate:.2f}-{segments[ranked_segments[j]][1]/framerate:.2f}",
                                    line_shape='linear'),row=i+j,col=1)
                
                # fig.update_layout(title="DTW Path")

                # DTW Path 
                x_ij =  [ [x_i[k],x_j[v],None] for (k,v) in dtw_path_matrix[ranked_segments[i]][ranked_segments[j]] ]
                y_ij =  [ [y_i[k],y_j[v],None] for (k,v) in dtw_path_matrix[ranked_segments[i]][ranked_segments[j]] ]
                
                x_ij = np.array(x_ij)[::5].reshape(-1)
                y_ij = np.array(y_ij)[::5].reshape(-1)
                
                fig.add_trace(go.Scatter(x=x_ij, y=y_ij, opacity=0.5, name=f"DTW between Segmemt:{ranked_segments[j]}-{ranked_segments[i]}",
                                    line_shape='linear'),row=i+j,col=1)

                subplot_titles[str(i+j)] = f"DTW betweens Segmemt:{ranked_segments[j]}-{ranked_segments[i]} Score:{dtw_cost_matrix[ranked_segments[i],ranked_segments[j]]:.2f}"

    fig.for_each_annotation(lambda a: a.update(text= subplot_titles[a.text]))
    fig.update_layout(title="DTW Path between selected segments",showlegend=False)
    
    if visualize:   
        fig.show()


    if fig_path is not None:
        fig.update_layout(width=600, height=800)
        fig.write_image(fig_path)
        
    return selected_segments,dtw_score 


def find_best_3_segments(max_pose_velocity,valleys,fig_path=None,visualize=False,normalization_constant=1,rms_threshold=0.75):
    """
        Validate segments using DTW

    """

    if len(valleys) < 4: return valleys, np.inf

    if len(valleys) > 25: 
        valleys = np.sort(np.random.choice(valleys,25,replace=False))

    from tslearn.metrics import dtw_path
    # from tslearn.barycenters import dtw_barycenter_averaging
    from itertools import combinations


    best_combination = -1 
    best_combination_score = np.inf
    best_combination_data = ()

    # valleys = [0] + valleys + [len(max_pose_velocity)-1]
    valleys = np.insert(valleys,0,0)
    valleys = np.insert(valleys,len(valleys),len(max_pose_velocity)-1)
    print(valleys) 

    rms_sequence = np.sqrt(np.sum(max_pose_velocity**2)) # If do not take valleys range but take the complete range then, sometimes rms_threshold fails to find valid sequecne

    for cur_comb in combinations(valleys, 4):
        rms = np.sqrt(np.sum(max_pose_velocity[cur_comb[0]:cur_comb[3]]**2))
        if rms < rms_sequence*rms_threshold:
            continue

        segments = [    
                        [cur_comb[0],cur_comb[1]],
                        [cur_comb[1],cur_comb[2]], 
                        [cur_comb[2],cur_comb[3]] 
                    ]

        segments = np.array(segments)
        dtw_cost_matrix = np.zeros(3) 
        dtw_path_matrix = [ [ ] for _ in range(3)]

        for i,s1 in enumerate(segments):
            for j,s2 in enumerate(segments):
                if i > j:
                    dtw_path_matrix[i+j-1],dtw_cost_matrix[i+j-1] = dtw_path(max_pose_velocity[s1[0]:s1[1]],max_pose_velocity[s2[0]:s2[1]])
                    
                    matrix_diagonal = np.sqrt((s1[1]-s1[0])**2 + (s2[1]-s2[0])**2) 

                    dtw_cost_matrix[i+j-1] /= matrix_diagonal
                    dtw_path_matrix[i+j-1] = np.array(dtw_path_matrix[i+j-1])



        # Rank segments based on the DTW cost matrix
        dtw_score = np.sum(dtw_cost_matrix)

        # print(f"Combination:{cur_comb} Score:{dtw_score}")

        if dtw_score*normalization_constant < best_combination_score: 
            best_combination = cur_comb
            best_combination_score = dtw_score
            best_combination_data = (segments,dtw_cost_matrix, dtw_path_matrix,rms)

    if best_combination == -1: 
        return valleys[:4], np.inf

    dtw_score = best_combination_score
    segments,dtw_cost_matrix,dtw_path_matrix,rms = best_combination_data

    # fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=3, cols=1,subplot_titles=[ str(i+1) for i in range(3)   ])
    subplot_titles = {}
    for i in range(3):
        for j in range(3):
            if i > j:
                # fig.add_trace(go.Scatter(x=dtw_path_matrix[selected_segments[i]][selected_segments[j]][:,0], y=dtw_path_matrix[selected_segments[i]][selected_segments[j]][:,1],
                                # line_shape='linear'),row=i,col=j)
                x_i = np.arange(segments[i][1]-segments[i][0]+1)/framerate
                y_i = max_pose_velocity[segments[i][0]:segments[i][1]]
                fig.add_trace(go.Scatter(x=x_i, y=y_i, name=f"Segmemt:{segments[i][0]/framerate:.2f}-{segments[i][1]/framerate:.2f}",
                                    line_shape='linear',
                                    ),row=i+j,col=1)
                
                x_j = np.arange(segments[j][1]-segments[j][0]+1)/framerate
                y_j = 180 + max_pose_velocity[segments[j][0]:segments[j][1]]
                fig.add_trace(go.Scatter(x=x_j, y=y_j, name=f"Segmemt:{segments[j][0]/framerate:.2f}-{segments[j][1]/framerate:.2f}",
                                    line_shape='linear'),row=i+j,col=1)
                
                # fig.update_layout(title="DTW Path")

                # DTW Path 
                x_ij =  [ [x_i[k],x_j[v],None] for (k,v) in dtw_path_matrix[i+j-1] ]
                y_ij =  [ [y_i[k],y_j[v],None] for (k,v) in dtw_path_matrix[i+j-1] ]
                
                x_ij = np.array(x_ij)[::5].reshape(-1)
                y_ij = np.array(y_ij)[::5].reshape(-1)
                
                fig.add_trace(go.Scatter(x=x_ij, y=y_ij, opacity=0.5, name=f"DTW between Segmemt:{j}-{i}",
                                    line_shape='linear'),row=i+j,col=1)

                subplot_titles[str(i+j)] = f"DTW betweens Segmemt:{j}-{i} Score:{dtw_cost_matrix[i+j-1]:.2f}"

    fig.for_each_annotation(lambda a: a.update(text= subplot_titles[a.text]))
    fig.update_layout(title=f"DTW Path between selected segments. Score:{dtw_score:.2f} RMS:{rms:.2f} RMS Sequence:{rms_sequence:.2f}",showlegend=False)
     
    if visualize:   
        fig.show()


    if fig_path is not None:
        fig.update_layout(width=800, height=800)
        fig.write_image(fig_path)
        
    return segments,dtw_score

err_files = []
def find_segments(mcs_smpl_path,framerate=60,visualize=False):
    """
        Find segments in the angular velocity of a time series of rotation vectors.
        params:
            data: A dictionary containing the pose_params

        returns: 
            segments: A list of tuples containing the start and end indices of the segments.
    """

    data = joblib.load(mcs_smpl_path)
    # assert_retarget_valid_smpl_format(data)
    # pose_velocity = np.rad2deg(np.linalg.norm(data['pose_params'].reshape((-1,24,3)),axis=2))
     
    # test_rotation = np.array([R.from_euler('x',0.01*(i**2),degrees=True).as_rotvec() for i in range(100)]).reshape((-1,1,3))
    # test_angular_velocity = calculate_angular_velocity(test_rotation,framerate=1)    
    
    if data['pose_params'].shape[0]/framerate < 4: # Skipping data cause it is too short
        print(f"Skipping data cause it is too short")
        return []

    pose_velocity = calculate_angular_velocity(data['pose_params'].reshape((-1,24,3))) 
    # max_pose_velocity = 0.5*np.sum(pose_velocity**2,axis=1)
    max_pose_velocity = np.max(pose_velocity,axis=1)
    
    # change_points = find_valleys_in_max_angular_velocity(max_pose_velocity,framerate=framerate)
    # change_points = use_clasp(max_pose_velocity,framerate=framerate)    
    change_points = find_all_valleyes_in_max_angular_velocity(max_pose_velocity,framerate=framerate)



    # segments = get_segment_from_change_points(change_points,len(max_pose_velocity),framerate=framerate)

    file_name = os.path.basename(mcs_smpl_path).split(".")[0]
    os.makedirs(SEGMENT_DIR, exist_ok=True)
    
    # Make sure image dir exists
    os.makedirs(os.path.join(RENDER_DIR,file_name,"images"), exist_ok=True)
    image_path = os.path.join(os.path.join(RENDER_DIR,file_name,"images"),file_name)


    # selected_segments,segment_score = validate_segments(max_pose_velocity,segments,fig_path=file_path.replace(".npy","_dtw.png"),visualize=visualize)
    segments, segment_score = find_best_3_segments(max_pose_velocity,change_points,fig_path=image_path + "_dtw.png",visualize=visualize)
    print(f"Sample:{mcs_smpl_path} image_path:{image_path}  save_path:{file_name} Segments: {segments} DTW Score:{segment_score}")


    # segments = segments[selected_segments]
    # segments_score = segment_score[selected_segments]
    if len(segments) != 3:
        print("Found invalid number of segments")
    # Save segments into npy file
    np.save(os.path.join(SEGMENT_DIR,file_name + '.npy'), {'segments':segments,'score':segment_score})

    # Create subplots
    fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=("(a) Joint Angular Velocity", "(b) Maximal Joint Angular Velocity"))

    x = np.arange(len(pose_velocity))/framerate

    # Plot the actual plot on the figure
    for i in range(24): 
        fig.add_trace(go.Scatter(
            x=x,
            y=pose_velocity[:,i],
            mode='lines',
            name=smpl_joints[i]
        ),row=1,col=1)


    fig.add_trace(go.Scatter(
        x=x,
        y=max_pose_velocity,
        mode='lines',
        name='Maximal Joint Angular Velocity'
    ),row=2,col=1)

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
    ),row=2,col=1)


    # Plot the line segments 
    if not np.isinf(segment_score): 
        empty_arry = np.array([None]*segments.shape[0]).reshape((-1,1))
        plot_y_segments = np.tile( np.max(max_pose_velocity).reshape((1,1)), segments.shape )
        plot_y_segments = np.concatenate([plot_y_segments,empty_arry],axis=1).reshape(-1)

        plot_x_segments = np.concatenate([segments/framerate,empty_arry],axis=1).reshape(-1)

        fig.add_trace(go.Scatter(
            x=plot_x_segments,
            y=plot_y_segments,
            line_shape='linear',
            name='Selected Segemetns'
        ),row=2,col=1)


    # Update subplot titles
    fig.update_layout(
        title_text=f'Angular Velocity analysis of SMPL joints for temportal segmentation for file:{mcs_smpl_path}',
        font=dict(family="Times New Roman"),
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Angular Velocity (deg/s)", row=1, col=1)
    fig.update_yaxes(title_text="Maximal Joint Angular Velocity (deg/s)", row=2, col=1)

    if visualize: 
        # Show the figure
        fig.show()

    fig.update_layout(width=1600, height=800)
    fig.write_image(image_path + "_angular.png")

    if SYSTEM_OS == 'Linux':
        print(f"Runnig command: convert +append {image_path + '_angular.png'} {image_path + '_dtw.png'} {image_path + '.png'}")
        os.system(f"convert +append {image_path + '_angular.png'} {image_path + '_dtw.png'} {image_path + '.png'}")
        os.system(f"rm {image_path + '_angular.png'} {image_path + '_dtw.png'}")
    elif SYSTEM_OS == 'Windows': 
        import subprocess

        # Define the file paths
        angular_png = file_path.replace('.npy', '_angular.png')
        dtw_png = file_path.replace('.npy', '_dtw.png')
        output_png = file_path.replace('.npy', '.png')

        # Construct the command
        # command = f"& \"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe\" +append \"{angular_png}\" \"{dtw_png}\" \"{output_png}\""
        command = [
            "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe",
            "+append",
            angular_png,
            dtw_png,
            output_png
            ]
        try:
            # Run the command
            subprocess.run(command, shell=True, check=True)
            # Remove the temporary files
            subprocess.run(f"del \"{angular_png}\" \"{dtw_png}\"", shell=True, check=True)

            # Print the command being executed
            print(f"Running command: {command}")
        except subprocess.CalledProcessError as e:
            # Print the error message
            print(f"Error: {e}")
            err_files.append(file_path)

        else: 
            raise OSError(f"Unable to use convert function to create visualization plots. Implemented for Linux and Windows. Not for {SYSTEM_OS}")        


if __name__ == "__main__": 

    framerate = 60
    if len(sys.argv) == 2: 
        mcs_smpl_path = sys.argv[1] 
        segments = find_segments(mcs_smpl_path,framerate=framerate,visualize=True)
    else: 
        for smpl_file in os.listdir(SMPL_DIR): 
            # if "BAP" not in smpl_file: 
                # continue
            # if "BAPF" in smpl_file: 
                # continue
            # file_name = os.path.basename(smpl_file).split(".")[0] + '.npy'
            # if os.path.isfile(os.path.join(SEGMENT_DIR,file_name)): 
                # continue 

            mcs_smpl_path = os.path.join(SMPL_DIR,smpl_file) 
            segments = find_segments(mcs_smpl_path,framerate=framerate,visualize=False)


            # time.sleep(5)
    
    print(err_files)
