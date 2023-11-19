import os 
import sys
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
from smplpytorch.pytorch.smpl_layer import SMPL_Layer # SMPL Model
from meters import Meters # Metrics to measure inverse kinematics
from renderer import Visualizer

class SMPLRetarget(nn.Module):
	def __init__(self,batch_size,device='cpu'):
		super(SMPLRetarget, self).__init__()

		# Create the SMPL layer
		self.smpl_layer = SMPL_Layer(center_idx=0,gender='neutral',model_root=os.path.join(HOME_PATH,'smplpytorch/native/models')).to(device)
		self.cfg = self.get_config(os.path.join(HOME_PATH,'Rajagopal_2016.json'))

		# Set utils
		self.batch_size = batch_size

		# Declare/Set/ parameters
		smpl_params = {}
		smpl_params["pose_params"] = torch.zeros(batch_size, 72)

		smpl_params["pose_params"][:,:3] = torch.from_numpy(np.tile(ROOT_INIT_ROTVEC[None,:],(batch_size,1))) # ROTATION VECTOR to initialize root joint orientation 
		smpl_params["pose_params"].requires_grad = True

		smpl_params["trans"] = torch.zeros(batch_size, 3)
		smpl_params["trans"].requires_grad = True

		smpl_params["shape_params"] = 1*torch.ones(10) if bool(self.cfg.TRAIN.OPTIMIZE_SHAPE) else torch.zeros(10)
		# smpl_params["shape_params"][0] = 5
		# smpl_params["shape_params"][1] = 0

		smpl_params["shape_params"][self.cfg.TRAIN.MAX_BETA_UPDATE_DIM:] = 0
		smpl_params["shape_params"].requires_grad = bool(self.cfg.TRAIN.OPTIMIZE_SHAPE)

		smpl_params["scale"] = torch.ones([1])
		smpl_params["scale"].requires_grad = bool(self.cfg.TRAIN.OPTIMIZE_SCALE)

		smpl_params["offset"] = torch.zeros((24,3))
		smpl_params["offset"].requires_grad = bool(self.cfg.TRAIN.OPTIMIZE_OFFSET)


		for k in smpl_params: 
			smpl_params[k] = nn.Parameter(smpl_params[k].to(device),requires_grad=smpl_params[k].requires_grad)
			self.register_parameter(k,smpl_params[k])
		self.smpl_params = smpl_params

		
		index = {}
		smpl_index = []
		dataset_index = []
		for tp in self.cfg.DATASET.DATA_MAP:
			smpl_index.append(tp[0])
			dataset_index.append(tp[1])
			
		index["smpl_index"] = smpl_index
		index["dataset_index"] = dataset_index
		for k in index: 
			index[k] = nn.Parameter(torch.LongTensor(index[k]).to(device),requires_grad=False)
			self.register_buffer(k,index[k])
		# index['smpl_reverse'] = dict([(x,i) for i,x in enum    
		index["parent_array"] = [0,0, 0, 0,1, 2, 3, 4, 5, 6, 7,8,9,9,9,12,13,14,16,17,18,19,20,21] # SMPL Parent Array for bones
		index['dataset_parent_array'] = self.cfg.DATASET.PARENT_ARRAY

		self.index = index



		self.optimizer = optim.Adam([{'params': self.smpl_params["scale"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			{'params': self.smpl_params["shape_params"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			{'params': self.smpl_params["pose_params"], 'lr': self.cfg.TRAIN.LEARNING_RATE},{'params': self.smpl_params["trans"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			{'params': self.smpl_params["offset"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			])
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)


	@staticmethod
	def get_config(config_path):
		with open(config_path, 'r') as f:
			data = json.load(f)
		cfg = edict(data.copy())
		return cfg	

	def forward(self):
		# print("Shape Params:",self.smpl_params['shape_params'])
		shape_params = self.smpl_params['shape_params'].repeat(self.batch_size,1)
		verts, Jtr, Jtr_offset = self.smpl_layer(self.smpl_params['pose_params'], th_betas=shape_params,th_offset=self.smpl_params['offset'])

		verts = verts*self.smpl_params["scale"] + self.smpl_params['trans'].unsqueeze(1)
		Jtr   = Jtr*self.smpl_params["scale"] + self.smpl_params['trans'].unsqueeze(1)
		Jtr_offset   = Jtr_offset*self.smpl_params["scale"] + self.smpl_params['trans'].unsqueeze(1) 

		return verts, Jtr, Jtr_offset


	def save(self,save_path):
		"""
			Save SMPL parameters as a pickle file
		"""	
		assert not os.path.isdir(save_path),f"Location to save file:{save_path} is a directory"

		res = self.smpl_params.copy()
		verts, Jtr, Jtr_offset = self()

		res['joints'] = Jtr


		for r in res: 
			res[r] = res[r].cpu().data.numpy()

		with open(save_path, 'wb') as f:
			pickle.dump(res, f)	

	def load(self,save_path):
			

	def __repr__(self):
		return f"Scale:{self.smpl_params['scale']} Trans:{self.smpl_params['trans'].mean(dim=0)} Betas:{self.smpl_params['shape_params']} Offset:{self.smpl_params['offset']}"        


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
	if not os.path.isdir(SMPL_PATH):
		os.makedirs(SMPL_PATH,exist_ok=True)

	save_path = os.path.join(SMPL_PATH,sample.name+'.pkl')
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

	video_dir = os.path.join(RENDER_PATH,f"{sample.openCapID}_{sample.label}_{sample.mcs}")

	if RENDER:
		vis.render_smpl(sample,smplRetargetter,video_dir=video_dir)        


	logger.info('Train ended, min_loss = {:.4f}'.format(
		float(meters.min_loss)))

	writer.flush()
	writer.close()	


	return smplRetargetter.smpl_params


# Load file and render skeleton for each video
def retarget_dataset():
	for subject in os.listdir(DATASET_PATH):
		for sample_path in os.listdir(os.path.join(DATASET_PATH,subject,'MarkerData')):
			sample_path = os.path.join(DATASET_PATH,subject,'MarkerData',sample_path)
			sample = OpenCapDataLoader(sample_path)

			if not os.path.isfile(os.path.join(SMPL_PATH,sample.name+'.pkl')): 
				smpl_params = retarget_opencap2smpl(sample)
				sample.smpl = smpl_params

			sample.smpl = SMPLLoader(os.path.join(SMPL_PATH,sample.name+'.pkl'))

if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		retarget_dataset()
	else:
		sample_path = sys.argv[1]
		sample = OpenCapDataLoader(sample_path)

		if not os.path.isfile(os.path.join(SMPL_PATH,sample.name+'.pkl')): 
			smpl_params = retarget_opencap2smpl(sample)
			sample.smpl = smpl_params
	
		sample.smpl = SMPLLoader(os.path.join(SMPL_PATH,sample.name+'.pkl'))	

