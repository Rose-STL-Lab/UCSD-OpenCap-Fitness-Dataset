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


 
# Modules
from utils import * # Config details 
from dataloader import OpenCapDataLoader # To load TRC file
from smplpytorch.pytorch.smpl_layer import SMPL_Layer # SMPL Model
from meters import Meters # Metrics to measure inverse kinematics
from renderer import Visualizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True




class SMPLRetarget(nn.Module):
	def __init__(self,batch_size,device='cpu',max_beta_update_dim=2):
		super(SMPLRetarget, self).__init__()

		# Create the SMPL layer
		self.smpl_layer = SMPL_Layer(center_idx=0,gender='neutral',model_root=os.path.join(HOME_PATH,'smplpytorch/native/models')).to(device)
		self.cfg = self.get_config(os.path.join(HOME_PATH,'Rajagopal_2016.json'))

		# Set utils
		self.batch_size = batch_size
		self.max_beta_update_dim = max_beta_update_dim

		# Declare/Set/ parameters
		smpl_params = {}
		smpl_params["pose_params"] = torch.zeros(batch_size, 72)
		smpl_params["pose_params"].requires_grad = True

		smpl_params["trans"] = torch.zeros(batch_size, 3)
		smpl_params["trans"].requires_grad = True

		smpl_params["shape_params"] = 1*torch.ones(10) if bool(self.cfg.TRAIN.OPTIMIZE_SHAPE) else torch.zeros(10)

		smpl_params["shape_params"][max_beta_update_dim:] = 0
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



		self.optimizer = optim.Adam([{'params': smpl_params["scale"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			{'params': smpl_params["shape_params"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			{'params': smpl_params["pose_params"], 'lr': self.cfg.TRAIN.LEARNING_RATE},{'params': smpl_params["trans"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
			{'params': smpl_params["offset"], 'lr': self.cfg.TRAIN.LEARNING_RATE},
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

	def __repr__(self):
		return f"Scale:{self.smpl_params['scale']} Trans:{self.smpl_params['trans'].mean(dim=0)} Betas:{self.smpl_params['shape_params']} Offset:{self.smpl_params['offset']}"        


def retarget_sample(sample:OpenCapDataLoader,save_path=None):

	# Log progress
	logger, writer = get_logger()
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

	smplRetargetter = SMPLRetarget(target.shape[0],device=device).to(device)
	logger.info(f"OpenCap to SMPL Retargetting details:{smplRetargetter.index}")	

	# Forward from the SMPL layer
	verts, Jtr, Jtr_offset = smplRetargetter()
	for epoch in tqdm(range(smplRetargetter.cfg.TRAIN.MAX_EPOCH)):

		
		# logger.debug(smplRetargetter)
		verts,Jtr,Jtr_offset = smplRetargetter()


		# print("Per joint loss:",torch.mean(torch.abs(scale*Jtr.index_select(1, index["smpl_index"])-target.index_select(1, index["dataset_index"])),dim=0))

		# verts *= scale
		
		loss_data = F.smooth_l1_loss(Jtr.index_select(1, smplRetargetter.index["smpl_index"]) ,
								target.index_select(1, smplRetargetter.index["dataset_index"]))

		loss_data_offset = F.smooth_l1_loss(Jtr_offset.index_select(1, smplRetargetter.index["smpl_index"]) ,
								target.index_select(1, smplRetargetter.index["dataset_index"]))
		

		loss_temporal_smooth_reg = F.smooth_l1_loss(smplRetargetter.smpl_params['pose_params'][1:],smplRetargetter.smpl_params['pose_params'][:-1])

		loss_offset_min = smplRetargetter.smpl_params['offset'].norm()

		loss_beta = smplRetargetter.smpl_params['shape_params'].norm()

		loss = loss_data + loss_data_offset + loss_temporal_smooth_reg + 0.01*loss_offset_min + loss_beta

		# criterion = nn.L1Loss(reduction ='none')
		# weights = torch.ones(Jtr.index_select(1, index["smpl_index"]).shape)

		# loss = criterion(scale*Jtr.index_select(1, index["smpl_index"]),
		#                         target.index_select(1, index["dataset_index"])) * weights

		smplRetargetter.optimizer.zero_grad()

		# logger.info(f"Loss:{loss}")

		loss.backward()

		# Don't update all beta parameters
		smplRetargetter.smpl_params['shape_params'].grad[smplRetargetter.max_beta_update_dim:] = 0
		
		smplRetargetter.optimizer.step()

		meters.update_early_stop(float(loss))
		# if meters.update_res:

		# if meters.early_stop or loss <= 0.00005:
		#     logger.info("Early stop at epoch {} !".format(epoch))
		#     break

		if epoch % smplRetargetter.cfg.TRAIN.WRITE == 0 or epoch == smplRetargetter.cfg.TRAIN.MAX_EPOCH-1:
			# logger.info("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
			#         epoch, float(loss),float(scale)))
			print("Epoch {}, lossPerBatch={:.6f} Data={:.6f} Offset={:.6f} Offset Norm={:.6f}  Temporal Reg:{:.6f} Beta Norm:{:.6f}".format(epoch, float(loss),float(loss_data),float(loss_data_offset),float(loss_offset_min),float(loss_temporal_smooth_reg),float(loss_beta)))
			# writer.add_scalar('loss', float(loss), epoch)
			# writer.add_scalar('learning_rate', float(smplRetargetter.optimizer.state_dict()['param_groups'][0]['lr']), epoch)
			# save_single_pic(res,smpl_layer,epoch,logger,args.dataset_name,target)

			# print(smplRetargetter)
			# smplRetargetter.show(target,verts,Jtr,Jtr_offset)
			vis.render_smpl(sample,smplRetargetter,video_dir=None)

			smplRetargetter.scheduler.step()
			res = smplRetargetter.smpl_params
			res['joints'] = Jtr

	# smplRetargetter.show(target,verts,Jtr,Jtr_offset)

	print(f'Saving results at:',save_path)
	if not os.path.isdir(save_path):
		os.makedirs(save_path,exist_ok=True)

	if save_path is not None: 
		save_path = os.path.join(save_path,sample.name+'.pkl')
		smplRetargetter.save(save_path)	
	

	# smplRetargetter.render(sample,smplRetargetter,video_dir=RENDER_PATH)        
	vis.render_smpl(sample,smplRetargetter,video_dir=None)        


	logger.info('Train ended, min_loss = {:.4f}'.format(
		float(meters.min_loss)))


	return smplRetargetter.smpl_params


# Load file and render skeleton for each video
def retarget_dataset():
	save_path = 'SMPL'
	
	for subject in os.listdir(DATASET_PATH):
		for sample_path in os.listdir(os.path.join(DATASET_PATH,subject,'MarkerData')):
			sample_path = os.path.join(DATASET_PATH,subject,'MarkerData',sample_path)
			sample = OpenCapDataLoader(sample_path)
			smpl_params = retarget_sample(sample,save_path=save_path)

if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		retarget_dataset()
	else:
		sample_path = sys.argv[1]
		sample = OpenCapDataLoader(sample_path)
		smpl_params = retarget_sample(sample)
