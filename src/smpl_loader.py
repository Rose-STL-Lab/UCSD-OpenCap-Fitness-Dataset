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
from smplpytorch.pytorch.smpl_layer import SMPL_Layer # SMPL Model
from meters import Meters # Metrics to measure inverse kinematics

class SMPLRetarget(nn.Module):
	def __init__(self,batch_size,device=None):
		super(SMPLRetarget, self).__init__()

		# Create the SMPL layer
		self.smpl_layer = SMPL_Layer(center_idx=0,gender='neutral',model_root=os.path.join(HOME_DIR,'smplpytorch/native/models')).to(device)
		self.cfg = self.get_config(os.path.join(DATA_DIR,'Rajagopal_2016.json'))

		if device is None: 
			device = torch.device('cpu')
		elif device == 'cpu':
			device = torch.device('cpu')
		elif device == 'gpu' or device == 'cuda': 
			device = torch.device('cuda')


		# Set utils
		self.device = device
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

		try: 
			with open(save_path, 'rb') as f:
				smpl_params = pickle.load(f)
		except Exception as e: 
			print(f"Unable to open smpl file:{save_path} Try deleting the file and rerun retargetting. Error:{e}")
			raise

		for k in smpl_params: 
			smpl_params[k] = torch.from_numpy(smpl_params[k]).to(self.device)	

		for k in self.smpl_params: 
			self.smpl_params[k] = smpl_params[k]

	def __repr__(self):
		return f"Scale:{self.smpl_params['scale']} Trans:{self.smpl_params['trans'].mean(dim=0)} Betas:{self.smpl_params['shape_params']} Offset:{self.smpl_params['offset']}"        

