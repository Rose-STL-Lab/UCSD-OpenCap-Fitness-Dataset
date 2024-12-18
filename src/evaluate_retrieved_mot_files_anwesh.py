# This file contains code to load MCS subjects and compute joint centers using the retrieved motion files (.mot) for the evaluation.
# Currently works for the 20 subjects only.
# Usage: python src/evaluate_retrieved_mot_files.py evaluate_retrieved_mot_files.py [-h] -m MOT [-d DATA_DIR] [-s SAVE] [-f] [-c CATEGORY]
# optional arguments:
#   -h, --help            show this help message and exit
#   -m MOT, --mot MOT     This option can be used in 4 ways: 1.Single Mot file. 2.
#                         Directory of containing Mot files 3. Single Mot file in the subject
#                         path (./<subject-path>/OpenSimData/<mot>/<file>.mot) 4. Directory
#                         of containing files in the subject path (./<subject-
#                         path>/OpenSim/<mot>/<exp_name>/*.mot) . Mot Files location
#							5. Surrogate results directory  ./<exp_name>/mot_visualization/latents_subject_run_<subject-path>/*.mot
#   -d DATA_DIR, --data-dir DATA_DIR
#                         Path to the database
#   -s SAVE, --save SAVE  Location to save joint motion information
#   -f, --force           forces a re-run on storing the retreived motion even if npy file
#                         containing joint locations data is already present.
#   -c CATEGORY, --category CATEGORY
#                         Match particular
 
# Example: python src/evaluate_retrieved_mot_files.py -m MCS_DATA/mot_visualization/normal_latents_temporal_consistency_v2
# Example: python src/evaluate_retrieved_mot_files.py -m Kinematics --c sqt

import os
import argparse # To parse command line arguments
import utils # Load Constants
import numpy as np # To save the motion files


import pandas as pd # To load MCS Session Ids from the MCS score sheet file 
from osim import load_osim, load_mot, OSIMSequence, NOT_MOT_FILE_ERROR # To load osim files and perform forward kinematics


def get_mcs_session_ids(mcs_score_sheet_file): 
	"""
		Load MCS session ids from the MCS score sheet file
		Args: 
			mcs_score_sheet_file: str
				Path to the MCS score sheet file

		Returns:
			List of MCS session ids
	"""

	# assert os.path.exists(mcs_score_sheet_file), f"MCS score sheet file:{mcs_score_sheet_file} does not exist"


	# mcs_file = pd.read_csv(mcs_score_sheet_file).dropna() # Removing NaN values
	# print("MCS Score Sheet File:",mcs_file.columns)
	# mcs_sessions = mcs_file['OpenCapID'].values
	# return ["000cffd9-e154-4ce5-a075-1b4e1fd66201"]
	# return ["349e4383-da38-4138-8371-9a5fed63a56a"]
	return ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]


def store_and_load_motion(osim_path, osim, mot_file,save_dir='MCS_DATA/RetrievedMotion',force_save=False):

	if not mot_file.endswith('.mot'):
		raise NOT_MOT_FILE_ERROR(f"File:{mot_file} is not a mot file.")

	save_name = os.path.basename(mot_file)
	save_name = save_name.replace('.mot','') + '_joints.npy'
	
	if save_dir is not None:
		save_name = os.path.join(save_dir, save_name)

	save_name = os.path.abspath(save_name) # Store absolute path

	# Store data if it does not already exist
	if not os.path.exists(save_name) or force_save:

		mot = load_mot(osim, mot_file)

		print("Loaded Mot file:",mot_file, mot.shape)
		print("Saving joints locations:",save_name)
		# if mot.shape[-1] != osim.get_num_dofs():
			# raise NOT_MOT_FILE_ERROR(f"Osim file has {osim.get_num_dofs()} dofs, but mot file has {mot.shape[-1]} dofs")		
		osim_fk = OSIMSequence(osim, mot, osim_path=osim_path)
		# Save the motion file		
		np.save(save_name, osim_fk.joints)
	
	return save_name, np.load(save_name)

def get_multiple_motions_for_mcs_data(args):

	assert os.path.exists(args.data_dir), f"Mot files:{args.data_dir} directory does not exist"

	osim_geometry_dir = os.path.join(utils.DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')
 
	print("Loading MCS data from:",args.data_dir)

	# Load only mcs session ids
	mcs_score_sheet_file = os.path.join(args.data_dir, 'OpenCap-MCS-score-sheet-MCS-Scores.csv')	
	session_ids = get_mcs_session_ids(mcs_score_sheet_file)

	
	subject_retreived_motions = {}

	# For every MCS subject: perform forward kinematics and get the joint motion
	for session_index, session_id in enumerate(session_ids):
		print("Session ID:", session_id)
		subject_path = os.path.join(args.data_dir, 'Data',session_id) 
		osim_path = os.path.join(subject_path,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted_contacts.osim')

		assert os.path.exists(osim_path), f"Osim file:{osim_path} does not exist"
		assert os.path.exists(osim_geometry_dir), f"Osim geometry path:{osim_geometry_dir} does not exist"

		# Store a list of motions for each subject 
		subject_retreived_motions[session_id] = []

		# Save location 
		save_dir = os.path.join(args.data_dir, 'Data',session_id, 'OpenSimData','RetrievedMotion')
		os.makedirs(save_dir, exist_ok=True)

		print(os.path.join(subject_path,'OpenSimData',args.mot))

		osim = load_osim(osim_path, osim_geometry_dir,ignore_geometry=False)

		# The mot file can be provided in 4 ways
		# 1. Single Mot file. Load the motion and return it.
		# if os.path.isfile(args.mot): 
		# 	joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, args.mot,save_dir=save_dir, force_save=args.force)
		# 	subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			

		# # 2. Single Mot file in the subject path
		# elif os.path.isfile(os.path.join(subject_path,'OpenSimData','Kinematics',args.mot)):
		# 	joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, args.mot, save_dir=save_dir,force_save=args.force)
		# 	subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))

		# # 3. Directory of containg Mot files
		# elif os.path.isdir(args.mot):
		# 	for mot_file in os.listdir(args.mot): 
		# 		mot_file = os.path.join(args.mot, mot_file)
		# 		try:
		# 			joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
		# 			subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
		# 			# break
		# 		except NOT_MOT_FILE_ERROR as e:
		# 			print(e)
		# 			continue


		# # 4. Directory of containg files in the subject path
		# if os.path.isdir(os.path.join(subject_path,'OpenSimData',args.mot)):
			

		################ Specific rules for loading kinematics data ##############################
		if 'Kinematics' in args.mot:
			for mot_file in os.listdir(os.path.join(subject_path,'OpenSimData',args.mot)):
				if not mot_file.endswith('.mot'): continue

				if args.category.lower() not in mot_file.lower(): continue# If does not match the matching critirea

				mot_file = os.path.join(subject_path,'OpenSimData', args.mot, mot_file)
				try:
					joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
					subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
				except NOT_MOT_FILE_ERROR as e:
					print(e)
					continue

		################ Specific rules for loading kinematics data ##############################
		elif args.mot == 'Dynamics':
			for trial_name in os.listdir(os.path.join(subject_path,'OpenSimData','Dynamics')):
				if args.category.lower() not in trial_name.lower(): continue# If does not match the category matching critirea

				if not os.path.isdir(os.path.join(subject_path,'OpenSimData','Dynamics', trial_name)): continue # Skip if not a directory

				# After running torque driven simulation, the kinematics and kinetics files are stored in the following format	
				mot_file = os.path.join(subject_path,'OpenSimData','Dynamics', trial_name,f"kinematics_activations_{trial_name}_torque_driven.mot")

				if not os.path.exists(mot_file): continue # Skip if directory does not contain the kinematics file. Probably the simulation failed

				try:
					joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
					subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
				except NOT_MOT_FILE_ERROR as e:
					print(e)
					continue

		elif 'PredDynamics' in args.mot:
			for trial_name in os.listdir(os.path.join(subject_path,'OpenSimData',args.mot)):
				if args.category.lower() not in trial_name.lower(): continue# If does not match the category matching critirea

				if not os.path.isdir(os.path.join(subject_path,'OpenSimData',args.mot, trial_name)): continue # Skip if not a directory

				# After running torque driven simulation, the kinematics and kinetics files are stored in the following format	
				mot_file = os.path.join(subject_path,'OpenSimData',args.mot, trial_name,f"kinematics_activations_{trial_name}_muscle_driven.mot")

				if not os.path.exists(mot_file): continue # Skip if directory does not contain the kinematics file. Probably the simulation failed

				try:
					joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
					subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
				except NOT_MOT_FILE_ERROR as e:
					print(e)
					continue
		
		elif args.mot == 'anwesh':
			path = "/home/ubuntu/data/MCS_DATA/LIMO/VQVAE-Generations/mot_visualization/"
			print("Path:",path + session_id+"/")
			if os.path.exists(path + "latents_subject_run_" + session_id+"/"):
				print("PATH EXISTS")
				import glob
				mot_files = glob.glob(path + "latents_subject_run_" + session_id+"/*.mot")
				for mot_file in mot_files:
					try:
						print("MOT File:",mot_file, "OSIM Path", osim_path)
						joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
						subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
					except NOT_MOT_FILE_ERROR as e:
						print(e)
						continue
		
		elif args.mot == 'sim':
			path = "/home/ubuntu/data/MCS_DATA/Data/*/OpenSimData/Dynamics/*_segment_*/kinematics_activations_*_muscle_driven.mot"
			# print("Path:",path + session_id+"/")
			# if os.path.exists(path + "latents_subject_run_" + session_id+"/"):
			# 	print("PATH EXISTS")
			import glob
			mot_files = glob.glob(path)
			for mot_file in mot_files:
				if session_id in mot_file:
					try:
						print("MOT File:",mot_file, "OSIM Path", osim_path)
						joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
						subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
					except NOT_MOT_FILE_ERROR as e:
						print(e)
						continue
  
		elif args.mot == 'mocap':
			path = "/home/ubuntu/data/MCS_DATA/Data/*/OpenSimData/Kinematics/*.mot"
			# print("Path:",path + session_id+"/")
			# if os.path.exists(path + "latents_subject_run_" + session_id+"/"):
			# 	print("PATH EXISTS")
			import glob
			mot_files = glob.glob(path)
			for mot_file in mot_files:
				if session_id in mot_file and ("sqt" in mot_file or "SQT" in mot_file or "Sqt" in mot_file) and ("segment" not in mot_file):
					try:
						print("MOT File:",mot_file, "OSIM Path", osim_path)
						joint_centers_file, joint_centers = store_and_load_motion(osim_path, osim, mot_file, save_dir=save_dir,force_save=args.force)
						subject_retreived_motions[session_id].append((joint_centers_file, joint_centers))			
					
					except NOT_MOT_FILE_ERROR as e:
						print(e)
						continue

		else: 
			# raise NOT_MOT_FILE_ERROR(f"Invalid motion file/directory:{args.mot}")
			continue
 
		
		print(f"Sessions[{session_index}/{len(session_ids)}]:{session_id} Motion Files:{len(subject_retreived_motions[session_id])}")

	save_files_list = []

	# Create a batch of the motion files 
	# max_motion_files = max([len(subject_retreived_motions[x]) for x in  subject_retreived_motions])

	# motion_batch_size = [len(session_ids),max_motion_files] + list(motion.shape)
	# motion_batch = np.zeros(motion_batch_size)

	for session_index,session_id in enumerate(session_ids):
		for motion_index, (joints_centers_file, joint_centers) in enumerate(subject_retreived_motions[session_id]):
			# motion_batch[session_index,motion_index] = motion
			save_files_list.append(joints_centers_file)


	assert len(save_files_list) > 0, "Not joints filed saved"
	

	if args.save is not None:
		# np.save(args.save, motion_batch)
		with open(args.save, 'w') as f:
			for file in save_files_list:
				f.write(file+'\n')
		print(f"Saved motion files list:{args.save} Len:{len(save_files_list)}")

	return save_files_list 

def parse_database_sync_options():

	options = argparse.ArgumentParser()

	# Default source: ubuntu@north.ucsd.edu -p 12700 -i <ssh-key>
	options.add_argument("-m", "--mot", required=True, 
					  help="This option can be used in 4 ways:\n \
							  	1.Single Mot file. \n \
								2. Directory of containg Mot files \n \
								3. Single Mot file in the subject path (./<subject-path>/OpenSimData/<mot>/<file>.mot) \n \
								4. Directory of containg files in the subject path \n (./<subject-path>/OpenSim/<mot>/<exp_name>/*.mot) \n . Mot Files location") 
	options.add_argument("-d", "--data-dir",default="MCS_DATA", help="Path to the database")

	options.add_argument("-s", "--save", default=None, help="Location to save joint motion information") 
	options.add_argument('-f', '--force',action='store_true',help="forces a re-run on storing the retreived motion even if npy file containg joit locations data is already present.")  # on/off flag
	options.add_argument('-c', '--category',default="sqt", help="Match particular ")  # on/off flag

	# options.add_argument("--source-ip-address", default=bc.NORTH['IP_ADDRESS'], help="Source Ip Address (if files not on local system)") 
	# options.add_argument("--source-port", default=12700, help="Source's port number to connect") 
	
	args = options.parse_args()


	# Preprocess the arguments
	if args.save is None:
		args.save = os.path.join(args.data_dir,os.path.basename(args.mot.rstrip("/")).replace('.mot',''))
		print(args.save)
		if not args.save.endswith('.txt'):
			args.save = args.save + '.txt'

	

	return args

if __name__ == "__main__": 
	args = parse_database_sync_options()
	motions = get_multiple_motions_for_mcs_data(args)
