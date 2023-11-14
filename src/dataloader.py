import os 
import re
import sys
from tqdm import tqdm
import numpy as np 

# File loaders
import pickle 


# DL Modules 

# Modules
from utils import * # All hyperparameters and paths are defined here


# Regex parse filename to get category and mcs score
FIND_LABEL_REGEX = r'^([A-Z]+)([0-9]+)(_([0-9]+))?\.trc$'
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
class OpenCapDataLoader:
	# Loads files from opencap and 
	def __init__(self,sample_path): 
		assert os.path.isfile(sample_path), f"File:{sample_path} does not exist"



		self.openCapID,self.label,self.mcs,self.sample = self.load_trc(sample_path)
		self.name = f"{self.openCapID}_{self.label}_{self.mcs}"
		self.frames,self.joints,self.joints_np = self.process_trc(self.sample) 

		self.joint2ind = dict([ (x,i) for i,x in enumerate(JOINT_NAMES)])


		self.fps = int(1/np.mean(self.frames[1:]-self.frames[:-1]))
		self.num_frames = len(self.frames)



	@staticmethod
	def get_label(sample_path):
		regex = re.compile(FIND_LABEL_REGEX)
		match = regex.match(sample_path)
	
		if match:
			# Extract the components using capture groups
			file_components = list(match.groups())

			label = file_components[0]
			mcs = int(file_components[1])

			if file_components[2] is not None:
				print(f"Weird filename:{sample_path}")
			return label,mcs
	
		else: 
			raise KeyError(f'{sample_path} does not match regex')




	@staticmethod
	def load_trc(sample_path):		
		# print(f'Loading file:{sample_path}')
		assert '.trc' == sample_path[:-4], f"Filename:{sample_path} not a OpenSim trc file" 
		assert "OpenCapData" not in os.path.basename(sample_path), f"Filepath:{os.path.basename(sample_path)} not sample."
		assert "MarkerData" not in os.path.basename(sample_path), f"Filepath:{os.path.basename(sample_path)} not sample."

		# File name details  
		openCapID = next(filter(lambda x: "OpenCapData" in x,sample_path.split('/')))
		openCapID = openCapID.split('_')[-1]

		label,mcs = OpenCapDataLoader.get_label(os.path.basename(sample_path))


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

		return openCapID,label,mcs,sample
			

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


# Converts SMPL parameters to Input representation 


class SMPLLoader: 
	def __init__(self):
		
		self.samples = [ self.load_smpl(os.path.join(SMPL_PATH,file)) for file in os.listdir(SMPL_PATH)]
		self.videos = len(self.samples)

		self.ind = 0

	@staticmethod
	def load_smpl(sample_path): 
		assert '.pkl' == sample_path[:-4], f"Filename:{sample_path} not a pickle file" 
		subjectID,label,mcs = sample_path.split('.')[0].split('_')
		name = f"{openCapID}_{label}_{mcs}"
		with open(save_path, 'r') as f:
			data = pickle.load(f)	
		return [subjectID, label,mcs,name, SMPLLoader.process_smpl(data)] 

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

	for subject in tqdm(os.listdir(DATASET_PATH)):
		for sample_path in os.listdir(os.path.join(DATASET_PATH,subject,'MarkerData')):

			# TRC File analysis
			sample_path = os.path.join(DATASET_PATH,subject,'MarkerData',sample_path)
			sample = OpenCapDataLoader(sample_path)
			
			if sample.label not in frames_distribution: 
				frames_distribution[sample.label] = {}
			if sample.mcs not in frames_distribution[sample.label]: 
				frames_distribution[sample.label][sample.mcs] = []

			frames_distribution[sample.label][sample.mcs].append(sample.num_frames)


			# SMPL File analysis
			sample_path = os.path.join(SMPL_PATH,subject,'MarkerData',sample_path)
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
