import os 
import re
import sys
import logging
from tqdm import tqdm
import numpy as np 
from functools import lru_cache
from scipy.spatial.transform import Rotation 

# File loaders
import yaml
import pickle 
import joblib

# DL Modules 

# Modules
from utils import * # All hyperparameters and paths are defined here


# Regex parse filename to get category and recordAttempt score
FIND_LABEL_REGEX = r'^([A-Za-z]+)([0-9]+)(_([0-9]+))?\.trc$'
"""
Explanation of the pattern:

^: Matches the start of the string.
[A-Z]+: Matches one or more uppercase letters.
[0-9]+: Matches one or more digits.
(_[0-9]+)?: Matches an optional underscore followed by one or more digits. This accounts for filenames like "LSLS01_1.trc."
\.trc: Matches the ".trc" extension.
$: Matches the end of the string.
This regex pattern should match filenames like the ones you've provided.
"""



# Module to store motion data for each sample 
# TODO: Rename OpenCapDataLoader to TRCLoader
class OpenCapDataLoader:
	# Loads files from opencap and 
	def __init__(self,sample_path): 
		assert os.path.isfile(sample_path), f"File:{sample_path} does not exist"


		self.sample_path = sample_path

		self.openCapID,self.label,self.recordAttempt_str, self.recordAttempt,self.sample = self.load_trc(sample_path)

		assert f"{self.label}{self.recordAttempt_str}.trc" == os.path.basename(sample_path), f"Unable process sample name:{os.path.basename(sample_path)} label:{self.label} rec_str:{self.recordAttempt_str}"

		self.name = f"{self.openCapID}_{self.label}_{self.recordAttempt}"
		self.frames,self.joints,self.joints_np = self.process_trc(self.sample) 

		self.joint2ind = dict([ (x,i) for i,x in enumerate(JOINT_NAMES)])


		self.fps = int(1/np.mean(self.frames[1:]-self.frames[:-1]))
		self.num_frames = len(self.frames)

		# Load MOT file corresponding to the .trc file
		self.mot = self.load_mot_files()



	@staticmethod
	def get_label(sample_path):
		regex = re.compile(FIND_LABEL_REGEX)
		match = regex.match(sample_path)
	
		if match:
			# Extract the components using capture groups
			file_components = list(match.groups())

			label = file_components[0]
			recordAttempt = int(file_components[1])

			recordAttempt_str = sample_path.replace(label,"").replace(".trc","")


			if file_components[2] is not None:
				logger = logging.getLogger(__name__)
				logger.info(f"Weird filename:{sample_path}")
			return label,recordAttempt_str,recordAttempt
	
		else: 
			try: 
				label = sample_path.split('.')[0]
				recordAttempt = 1 
				recordAttempt_str = ""
				return label,recordAttempt_str,recordAttempt
			except Exception as e: 
				raise KeyError(f'{sample_path} does not match regex, {e}')


	@staticmethod
	def load_trc(sample_path):		
		assert '.trc' == sample_path[-4:], f"Filename:{sample_path} not a OpenSim trc file" 
		assert "OpenCapData" not in os.path.basename(sample_path), f"Filepath:{os.path.basename(sample_path)} not sample."
		assert "MarkerData" not in os.path.basename(sample_path), f"Filepath:{os.path.basename(sample_path)} not sample."

		print(sample_path)

		try: 

			# File name details  
			if SYSTEM_OS == 'Linux':
				openCapID = next(filter(lambda x: "OpenCapData" in x,sample_path.split('/')))
			elif SYSTEM_OS == 'Windows': 
				openCapID = next(filter(lambda x: "OpenCapData" in x,sample_path.split('\\')))
			else: 
				raise OSError(f"Unable to split .trc file to find OpenCapID. Implemented for Linux and Windows. Not for {SYSTEM_OS}")
			openCapID = openCapID.split('_')[-1]
		except Exception as os_error: 
			openCapID = os.path.basename(os.path.dirname(os.path.dirname(sample_path))) # Asssuming directory structure sessionID/MarkerData/<sample>.trc, taking session as the parent of parent of 
			print(f"Asssuming directory structure sessionID/MarkerData/<sample>.trc, taking session as the parent of parent of {os_error}")
			print("OpenCap ID:",openCapID) 

		label,recordAttempt_str,recordAttempt = OpenCapDataLoader.get_label(os.path.basename(sample_path))


		# Open file and load xyz co-ordinate for every joint
		sample = []
		with open(sample_path,'r') as f: 
			data = f.read().split('\n')

		for idx, line in enumerate(data):
			if len(line) == 0: 
				continue 
			if idx == 3:
				headers = [t for t in line.strip().split("\t") if t != ""][:22]
			if idx > 5:
				pose = {}
				data = line.split('\t')
				pose[headers[0]] = data[0]
				pose[headers[1]] = data[1]
				data = data[2:]
				for i, header in enumerate(headers[2:]):
					pose[header] = {}
					pose[header]['x'] = float(data[i*3 + 0])
					pose[header]['y'] = float(data[i*3 + 1])
					pose[header]['z'] = float(data[i*3 + 2])
				
				sample.append(pose)
		
		# print(openCapID,label,recordAttempt,sample)

		return openCapID,label,recordAttempt_str, recordAttempt,sample
			

	def process_trc(self,sample):
			
		frames = []
		joints = {}

		for i,pose in enumerate(sample):
			for joint in pose: 
				if joint == 'Frame#': 
					continue 
				elif joint == 'Time': 
					frames.append(float(pose['Time']))
				else: 

					assert type(pose[joint]) == dict\
					 and len(pose[joint]) == 3\
					 and 'x' in pose[joint]\
					 and 'y' in pose[joint]\
					 and 'z' in pose[joint], f"Error in reading data:{pose[joint]}"


					if joint not in joints: 
						joints[joint] = []

					joints[joint].append([float(pose[joint]['x']),float(pose[joint]['y']),float(pose[joint]['z'])])


		# Check the file data matches the OpenCap format
		assert len(joints) == 20, f"Number of joints should be 20 but found:{len(joints)}"
		assert all([len(frames) == len(joints[joint]) for joint in joints]), f"Frames and num pose don't match:{[(str(joint),len(frames)) == len(joints[joint]) for joint in joints]}" 
		assert len(frames) == int(pose['Frame#'])

		frames = np.array(frames)

		joint_np = np.array([joints[joint] for joint in JOINT_NAMES]).transpose((1,0,2))

		return frames,joints,joint_np
	
	def load_mot_files(self): 
		mot_file_path = os.path.dirname(os.path.dirname(self.sample_path))
		mot_file_path = os.path.join(mot_file_path, 'OpenSimData', 'Kinematics', self.label + self.recordAttempt_str + '.mot')

		with open(mot_file_path,'r') as f: 
			file_data = f.read().split('\n') 
			
			data = {'info':'', 'poses': []}
			read_header = False
			read_rows = 0 
			
			for line in file_data:  
				line = line.strip()
				if len(line) == 0:
					continue
				
				if not read_header:
					if line == 'endheader': 
						read_header = True

						logger = logging.getLogger(__name__)
						# Assert all required fields are present in the header

						if 'nColumns' not in data:
							logger.warning(f"data does number of columns:{data} for {mot_file_path}")
						if 'nRows' not in data:
							logger.warning(f"data does not have number of rows:{data} for {mot_file_path}")
					
						continue 
						
					if '=' not in line: 
						data['info'] += line + '\n' 
					else: 
						k,v = line.split('=')
						if v.isnumeric():
							data[k] = int(v)
						else: 
							data[k] = v
				
				else: 
					rows = line.split()

					if len(rows) != data['nColumns']:
						logger.warning(f"row:{read_rows} does not contains the right number of values:{len(rows)} != {data['nColumns']}")

					if read_rows == 0:
						data['headers'] = rows 
					else: 
						rows = [float(row) for row in rows]
						data['poses'].append(rows) 


					read_rows += 1 		

		data['poses'] = np.array(data['poses'])
		return data			

class MultiviewRGB: 
	def __init__(self,sample):
		self.sample = sample
		self.session_data = self.load_subjectID(sample.sample_path)
		self.video_paths = self.get_video_paths()
		self.cameras = self.get_camera_paths(sample.sample_path)

	def load_subjectID(self,sample_path):
		session_path = os.path.dirname(sample_path)
		session_path = os.path.dirname(session_path)
		session_path = os.path.join(session_path,'sessionMetadata.yaml')
		
		assert os.path.isfile(session_path), f"Session path:{session_path} for sample:{session_path} does not exist"

		with open(session_path, "r") as stream:
			try:
				session_data = yaml.safe_load(stream)
			except yaml.YAMLError as exc:
				print(exc)

		return session_data
	
	def get_video_paths(self):
		""""Get video paths for each camera in the session"""
		video_path = os.path.dirname(self.sample.sample_path)
		video_path = os.path.dirname(video_path)
		video_path = os.path.join(video_path,"Videos")
		video_paths = []

		ls_video_path = [ file for file in os.listdir(video_path) if os.path.isdir(os.path.join(video_path,file))]
		ls_video_path = sorted(ls_video_path, key=lambda x: int(x.lower().replace("cam","")))

		for d in ls_video_path:
		
			current_camera_path = os.path.join(video_path,d,"InputMedia", self.sample.label + self.sample.recordAttempt_str)


			mp4_name = [file for file in os.listdir(os.path.join(current_camera_path)) if 'sync' in file ]
			if len(mp4_name) == 0: continue
			mp4_name = mp4_name[0]

			mp4_name = os.path.join(current_camera_path,mp4_name)

			assert os.path.isfile(mp4_name), f"Mov file:{mp4_name} does not exist"

			video_paths.append(mp4_name)
		
		# Make sure video paths are valid 
		assert all([ os.path.isfile(file) for file in video_paths]), f"video paths do not exist. Check:{video_paths[0]} is an mp4" 	

		return video_paths

	def get_camera_paths(self,sample_path): 
		
		# Get filenames corresponding to each video 
		camera_filenames = self.video_paths
		camera_filenames = [os.path.dirname(file) for file in camera_filenames]
		camera_filenames = [os.path.dirname(file) for file in camera_filenames]
		camera_filenames = [os.path.dirname(file) for file in camera_filenames]
		
		# Make sure directories are valid 
		assert all([ os.path.isdir(dir) for dir in camera_filenames]), f"Camera paths not directory. Check:{camera_filenames[0]} is an mp4" 	

		cameras = []
		for camera_ind, camera_filepath in enumerate(camera_filenames): 
			pickle_files = [ file for file in os.listdir(camera_filepath) if 'pickle' in file]
			assert len(pickle_files) == 1, f"Unable to find pickle file in {camera_filepath}" 
 
			camera_filepath = os.path.join(camera_filepath, pickle_files[0])

			assert os.path.isfile(camera_filepath), f"Unable to find pickle file:{camera_filepath}" 
			
			camera = joblib.load(camera_filepath)
			camera = self.process_camera(camera)
			camera['video_paths'] = self.video_paths[camera_ind]
			cameras.append(camera)
		
		return cameras

	@staticmethod
	def process_camera(camera): 
		"""
			Process camera parameters extracted from opencap for openpose parameters 
			- Create field of view from intrinsic matrix 
			- Create projection matrix 

			Inspired by Pose2Sim: https://github.com/davidpagnon/Pose2Sim_Blender/blob/main/Pose2Sim_Blender/cameras.py#L348  

			@params: 
				camera: loaded .pickle file as a dict 

			@returns: 
			 	camera: dict with more parameters

		"""

		# image dimensions
		w, h = camera['imageSize'].reshape(-1)
		
		# Conversation from intrinsic parameters to field of view. Use the width and height of the image. Refer: https://github.com/davidpagnon/Pose2Sim_Blender/blob/13c516dc30d30e1185da53254da781b86b47e7c3/Pose2Sim_Blender/cameras.py#L362 
		fx, fy = camera['intrinsicMat'][0,0], camera['intrinsicMat'][1,1] 
		camera['fov_x'] = 2 * np.arctan2(w, 2 * fx) * 180/np.pi 
		camera['fov_y'] = 2 * np.arctan2(h, 2 * fy) * 180/np.pi
		
		# Conversation from ext matrix to look dir/forward dir and  up vector Refer: https://medium.com/@carmencincotti/lets-look-at-magic-lookat-matrices-c77e53ebdf78
		camera['position'] = -camera['rotation'].T@camera['translation'] 
		camera['position'] /= fx # From milimeters to meters (not sure if this is correct)
		camera['position'] = -camera['position'][::-1] # Swap x and z axis. Also rotate by 180 degrees because of opengl

		camera['look_dir'] = camera['rotation'][2,:]
		camera['look_dir'] = -camera['look_dir'][::-1] # Swap x and z axis. Also rotate by 180 degrees because look dir is opposite of forward dir

		camera['up_dir'] = camera['rotation'][1,:]
		camera['up_dir'] = camera['up_dir'][::-1]




		# set_loc_rotation(camera, np.radians([180,0,0]))
		
		# # principal point # see https://blender.stackexchange.com/a/58236/174689
		# principal_point =  K[c][0,2],  K[c][1,2]
		# max_wh = np.max([w,h])
		
		# camera.data.shift_x = 1/max_wh*(principal_point[0] - w/2)
		# camera.data.shift_y = 1/max_wh*(principal_point[1] - h/2)	

		return camera



# Converts SMPL parameters to Input representation 
class SMPLLoader: 
	def __init__(self):
		
		self.samples = [ self.load_smpl(os.path.join(SMPL_DIR,file)) for file in os.listdir(SMPL_DIR)]
		self.videos = len(self.samples)

		self.ind = 0

	@staticmethod
	def load_smpl(sample_path): 
		assert '.pkl' == sample_path[:-4], f"Filename:{sample_path} not a pickle file" 
		subjectID,label,recordAttempt = sample_path.split('.')[0].split('_')
		name = f"{openCapID}_{label}_{recordAttempt}"
		with open(save_path, 'r') as f:
			data = pickle.load(f)	
		return [subjectID, label,recordAttempt,name, SMPLLoader.process_smpl(data)] 

	@staticmethod 
	def process_smpl(data):
		# Convert smpl pose params to model input representation
		for k in data: 
			print(k,data[k].shape)
		return data

	def __iter__(self):
		self.ind = 0
		return self

	def __next__(self):
		if self.ind < self.frames:
			return self.samples[self.ind]
		else:
			raise StopIteration



# Analyze actions dataset
def analyze_dataset():

	frames_distribution = {}

	for subject in tqdm(os.listdir(INPUT_DIR)):
		for sample_path in os.listdir(os.path.join(INPUT_DIR,subject,'MarkerData')):

			if sample_path == "Settings": continue 

			# TRC File analysis
			sample_path = os.path.join(INPUT_DIR,subject,'MarkerData',sample_path)

			sample = OpenCapDataLoader(sample_path)

			if sample.label not in frames_distribution: 
				frames_distribution[sample.label] = {}
			if sample.recordAttempt not in frames_distribution[sample.label]: 
				frames_distribution[sample.label][sample.recordAttempt] = []

			frames_distribution[sample.label][sample.recordAttempt].append(sample.num_frames)


			# SMPL File analysis
			sample_path = os.path.join(SMPL_DIR,subject,'MarkerData',sample_path)
			sample = OpenCapDataLoader(sample_path)


	classes = list(frames_distribution.keys())

	for c in classes:
		print(f"Class:{c} Frames_distribution:{[(x,np.mean(frames_distribution[c][x])) for x in frames_distribution[c]]}")





if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		analyze_dataset()
	else:
		sample_path = sys.argv[1]
		sample = OpenCapDataLoader(sample_path)
