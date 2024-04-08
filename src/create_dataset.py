import os
import sys
import joblib
import numpy as np

from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R
from utils import smpl_joints,cuda,SMPL_DIR





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


def find_peaks_in_max_angular_velocity(max_angular_velocity,framerate=60):
    """
        Find peaks in the angular velocity of a time series of rotation vectors.

        Args:
        angular_velocity: A numpy array of shape (N, 24) containing angular velocities in radians per second.
        framerate: The frame rate of the data in Hz.

        Returns: A numpy array of shape (M,) containing the indices of the peaks in the angular velocity.
    """

    diff = np.max(max_angular_velocity) - np.min(max_angular_velocity)
    min_height = np.min(max_angular_velocity) + 0.1*diff
    peaks, meta_data = find_peaks(-max_angular_velocity,distance=len(max_angular_velocity)//5,height=-min_height)
    # peaks = peaks + 2*60 # Adjust for the offset

    print("Meta data: ",meta_data)



    # while peaks[0] < len(max_angular_velocity)//4: # Found first peak too early. Better to remove it
    #     print("Peaks: ",peaks)
    #     if len(peaks) <= 2: 
    #         break
        
    #     peaks = peaks[1:]
        


    # while peaks[-1] > 3*len(max_angular_velocity)//4: # Found last peak too late. Better to remove it
    #     if len(peaks) <= 2: 
    #         break
    #     peaks = peaks[:-1]



    return peaks


def find_segments(data,framerate=60,visualize=False):
    """
        Find segments in the angular velocity of a time series of rotation vectors.
        params:
            data: A dictionary containing the pose_params

        returns: 
            segments: A list of tuples containing the start and end indices of the segments.
    """

    # assert_retarget_valid_smpl_format(data)
    # pose_velocity = np.rad2deg(np.linalg.norm(data['pose_params'].reshape((-1,24,3)),axis=2))
     
    # test_rotation = np.array([R.from_euler('x',0.01*(i**2),degrees=True).as_rotvec() for i in range(100)]).reshape((-1,1,3))
    # test_angular_velocity = calculate_angular_velocity(test_rotation,framerate=1)    
    
    if data['pose_params'].shape[0]/framerate < 4: # Skipping data cause it is too short
        print(f"Skipping data cause it is too short")
        return []

    pose_velocity = calculate_angular_velocity(data['pose_params'].reshape((-1,24,3))) 
    max_pose_velocity = np.max(pose_velocity,axis=1)
    
    peaks = find_peaks_in_max_angular_velocity(max_pose_velocity,framerate=framerate)


    segments = []
    first = 0 
    for peak in peaks:
        segments.append((first,peak))
        first = peak
    segments.append((first,len(max_pose_velocity)))

    print("Segments: ",segments)
    if len(segments) != 3:
        print("Found invalid number of segments")

    

    if visualize:
        import plotly.subplots
        import plotly.graph_objects as go

        # Create subplots
        fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=("Joint Angular Velocity", "Maximal Joint Angular Velocity"))

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
            x=x[peaks],
            y=max_pose_velocity[peaks],
            mode='markers',
            marker=dict(
                color='red',
                size=16,
                symbol='arrow-up'
            ),
            name='Peaks'
        ),row=2,col=1)

        # Update subplot titles
        fig.update_layout(title_text=f'Angular Velocity analysis of SMPL joints for temportal segmentation for file:{mcs_smpl_path}')

        # Show the figure
        fig.show()


if __name__ == "__main__": 

    framerate = 60
    if len(sys.argv) == 2: 
        mcs_smpl_path = sys.argv[1] 
        data = joblib.load(mcs_smpl_path)
        segments = find_segments(data,framerate=framerate,visualize=True)
    else: 
        import time 
        for smpl_file in os.listdir(SMPL_DIR): 
            if "BAP" not in smpl_file: 
                continue
            if "BAPF" in smpl_file: 
                continue
            mcs_smpl_path = os.path.join(SMPL_DIR,smpl_file) 
            data = joblib.load(mcs_smpl_path)
            segments = find_segments(data,framerate=framerate,visualize=True)

            time.sleep(5)
