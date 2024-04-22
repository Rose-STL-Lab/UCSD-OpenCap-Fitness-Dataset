import os 
import sys
import argparse
import numpy as np 
from tqdm import tqdm

# Modules to load config file and save generated parameters
import json 
import pickle
from easydict import EasyDict as edict 

# DL Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
 
# Modules
from utils import * # Config details 
from dataloader import OpenCapDataLoader,SMPLLoader # To load TRC file
from smpl_loader import SMPLRetarget
from smplpytorch.pytorch.smpl_layer import SMPL_Layer # SMPL Model
from meters import Meters # Metrics to measure inverse kinematics
from renderer import Visualizer



def retarget_opencap2smpl(sample:OpenCapDataLoader):

	# Log progress
	logger, writer = get_logger(task_name='Retarget')
	logger.info(f"Retargetting file:{sample.openCapID}_{sample.label}")

	# Metrics to measure
	meters = Meters()

	# Visualizer
	vis = Visualizer()

	# GPU mode
	if cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')



	target = torch.from_numpy(sample.joints_np).float()
	target = target.to(device)

	smplRetargetter = SMPLRetarget(sample.joints_np.shape[0],device=device).to(device)
	logger.info(f"OpenCap to SMPL Retargetting details:{smplRetargetter.index}")	
	logger.info(smplRetargetter.cfg.TRAIN)

	# Intialize trans at root joint location
	with torch.no_grad():
		smplRetargetter.smpl_params['trans'][:] = target[:,smplRetargetter.index["dataset_index"][0]]

	# Forward from the SMPL layer
	# if DEBUG: 
		# verts, Jtr, Jtr_offset = smplRetargetter()



	for epoch in tqdm(range(smplRetargetter.cfg.TRAIN.MAX_EPOCH)):

		
		# logger.debug(smplRetargetter)
		verts,Jtr,Jtr_offset = smplRetargetter()


		# print("Per joint loss:",torch.mean(torch.abs(scale*Jtr.index_select(1, index["smpl_index"])-target.index_select(1, index["dataset_index"])),dim=0))

		# verts *= scale


		# DATA Loss Terms
		loss_data = F.smooth_l1_loss(Jtr.index_select(1, smplRetargetter.index["smpl_index"]) ,
								target.index_select(1, smplRetargetter.index["dataset_index"]))



		loss_data_offset = F.smooth_l1_loss(Jtr_offset.index_select(1, smplRetargetter.index["smpl_index"]) ,
								target.index_select(1, smplRetargetter.index["dataset_index"]))
		

		loss_trans = F.smooth_l1_loss(smplRetargetter.smpl_params['trans'],target[:,smplRetargetter.index["dataset_index"][0],:])

		# Regularizers
		loss_temporal_smooth_reg = F.smooth_l1_loss(smplRetargetter.smpl_params['pose_params'][1:],smplRetargetter.smpl_params['pose_params'][:-1])

		loss_offset_min = smplRetargetter.smpl_params['offset'].norm()

		loss_beta = smplRetargetter.smpl_params['shape_params'].norm()


		# logger.debug(f"LAMBDA OFFSET:{smplRetargetter.cfg.TRAIN.LAMBDA_NORM_OFFSET}")
		loss = loss_data 
		loss += loss_data_offset 
		loss += 10*loss_temporal_smooth_reg 
		loss += 1e-6*loss_offset_min  # U
		loss += 0.00001*loss_beta 
		# loss += smplRetargetter.cfg.TRAIN.LAMBDA_TRANS*loss_trans


		if epoch % smplRetargetter.cfg.TRAIN.WRITE == 0  or epoch == smplRetargetter.cfg.TRAIN.MAX_EPOCH-1:
			logger.info("Epoch {}, LR:{:.6f} lossPerBatch={:.6f} Data={:.6f} Offset={:.6f} Trans:{:.6f} Offset Norm={:.6f}  Temporal Reg:{:.6f} Beta Norm:{:.6f}".format(epoch, float(smplRetargetter.scheduler._last_lr[-1]) , float(loss),float(loss_data),float(loss_data_offset),float(loss_trans),float(loss_offset_min),float(loss_temporal_smooth_reg),float(loss_beta)))
			logger.debug(f"scale:{smplRetargetter.smpl_params['scale']}")
			logger.debug(f"Beta:{smplRetargetter.smpl_params['shape_params']}")
			
			for x,y in zip(["LR", "lossPerBatch", "Data", "Offset", "Trans", "Reg Offset", "Reg Temporal", "Reg BETA Norm"],\
						[smplRetargetter.scheduler._last_lr[-1] , loss,loss_data,loss_data_offset,loss_trans,loss_offset_min,loss_temporal_smooth_reg,loss_beta]):
				writer.add_scalar(x, float(y), epoch)
			
			# writer.add_scalar('learning_rate', float(smplRetargetter.optimizer.state_dict()['param_groups'][0]['lr']), epoch)

			smplRetargetter.scheduler.step()

		# criterion = nn.L1Loss(reduction ='none')
		# weights = torch.ones(Jtr.index_select(1, index["smpl_index"]).shape)

		# loss = criterion(scale*Jtr.index_select(1, index["smpl_index"]),
		#                         target.index_select(1, index["dataset_index"])) * weights

		smplRetargetter.optimizer.zero_grad()

		# logger.info(f"Loss:{loss}")

		loss.backward()

		# Don't update all beta parameters
		if smplRetargetter.smpl_params['shape_params'].grad is not None:
			smplRetargetter.smpl_params['shape_params'].grad[smplRetargetter.cfg.TRAIN.MAX_BETA_UPDATE_DIM:] = 0
		
		smplRetargetter.optimizer.step()

		meters.update_early_stop(float(loss))
		# if meters.update_res:

		# if meters.early_stop or loss <= 0.00005:
		#     logger.info("Early stop at epoch {} !".format(epoch))
		#     break




	# smplRetargetter.show(target,verts,Jtr,Jtr_offset)
	if not os.path.isdir(SMPL_DIR):
		os.makedirs(SMPL_DIR,exist_ok=True)

	save_path = os.path.join(SMPL_DIR,sample.name+'.pkl')
	logger.info(f'Saving results at:{save_path}')
	smplRetargetter.save(save_path)	


	# Plot HIP and angle joints to visualize 
	for i in range(smplRetargetter.smpl_params['pose_params'].shape[0]):
		# LHIP 
		writer.add_scalar(f"LHip-Z", float(smplRetargetter.smpl_params['pose_params'][i,1*3 + 0]),i )
		writer.add_scalar(f"LHip-Y", float(smplRetargetter.smpl_params['pose_params'][i,1*3 + 1]),i )
		writer.add_scalar(f"LHip-X", float(smplRetargetter.smpl_params['pose_params'][i,1*3 + 2]),i )
		# RHIP 
		writer.add_scalar(f"RHip-Z", float(smplRetargetter.smpl_params['pose_params'][i,2*3 + 0]),i )
		writer.add_scalar(f"RHip-Y", float(smplRetargetter.smpl_params['pose_params'][i,2*3 + 1]),i )
		writer.add_scalar(f"RHip-X", float(smplRetargetter.smpl_params['pose_params'][i,2*3 + 2]),i )

		# L-Ankle 
		writer.add_scalar(f"LAnkle-Z", float(smplRetargetter.smpl_params['pose_params'][i,7*3 + 0]),i )
		writer.add_scalar(f"LAnkle-Y", float(smplRetargetter.smpl_params['pose_params'][i,7*3 + 1]),i )
		writer.add_scalar(f"LAnkle-X", float(smplRetargetter.smpl_params['pose_params'][i,7*3 + 2]),i )

		writer.add_scalar(f"RAnkle-Z", float(smplRetargetter.smpl_params['pose_params'][i,8*3 + 0]),i )
		writer.add_scalar(f"RAnkle-Y", float(smplRetargetter.smpl_params['pose_params'][i,8*3 + 1]),i )
		writer.add_scalar(f"RAnkle-X", float(smplRetargetter.smpl_params['pose_params'][i,8*3 + 2]),i )

	video_dir = os.path.join(RENDER_DIR,f"{sample.openCapID}_{sample.label}_{sample.recordAttempt}")

	if RENDER:
		vis.render_smpl(sample,smplRetargetter,video_dir=video_dir)        


	logger.info('Train ended, min_loss = {:.4f}'.format(
		float(meters.min_loss)))

	writer.flush()
	writer.close()	


	return smplRetargetter


def retarget_sample(sample_path):
	sample = OpenCapDataLoader(sample_path)

	if not os.path.isfile(os.path.join(SMPL_DIR,sample.name+'.pkl')) or cmd_line_args.force: 
		sample.smpl = retarget_opencap2smpl(sample)
	else:	
		torch_device = torch.device('cuda' if cuda else 'cpu')	
		sample.smpl = SMPLRetarget(sample.joints_np.shape[0],device=torch_device).to(torch_device)	
		sample.smpl.load(os.path.join(SMPL_DIR,sample.name+'.pkl'))

	return sample


# Load file and render skeleton for each video
def retarget_dataset():
	for subject in os.listdir(INPUT_DIR):
		for sample_path in os.listdir(os.path.join(INPUT_DIR,subject,'MarkerData')):
			sample_path = os.path.join(INPUT_DIR,subject,'MarkerData',sample_path)
			sample = retarget_sample(sample_path)

if __name__ == "__main__": 


	############################# Command line Argument Parser #######################################################
	parser = argparse.ArgumentParser(
						prog='Retargetting',
						description='Retargets from SMPL to RaBit',
						epilog='')
	parser.add_argument('--file',type=str,help="Path to .trc file that needs to be retargeted.")  # path to trc file
	parser.add_argument('-f', '--force',action='store_true',help="forces a re-run on retargetting even if pkl file containg smpl data is already present.")  # on/off flag
	parser.add_argument('--render', action='store_true', help="Render a video and save it it in RENDER_DIR. Can also be set in the utils.py")  # on/off flag


	cmd_line_args = parser.parse_args()

	RENDER = RENDER | cmd_line_args.render 

	if cmd_line_args.file is None: 
		retarget_dataset()
	else:
		sample_path = cmd_line_args.file
		sample = retarget_sample(sample_path)

