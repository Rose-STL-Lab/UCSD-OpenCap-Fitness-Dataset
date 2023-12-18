import os 
import numpy as np 

# DL Modules 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# Modules 
from utils import * 
from dataloader import SMPLLoader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer # SMPL Model
from meters import Meters # Metrics to measure inverse kinematics
from renderer import Visualizer


class PoseReconstructer: 
	def __init__(self,batch_size,device='cpu'):
		pass

	def fit(self,inp):	

	def reconstruct(self,inp):
		return inp



# Load file and render skeleton for each video
def pca_pose_reconstruction():

	poseReconstructer = PoseReconstructer()
	vis = Visualizer()
	dataloader = SMPLLoader(SMPL_DIR)
	poses = [sample.inp for sample in dataloader.get_in ]
	for sample in : 
		rec = reconstruct(sample.inp)

		vis.render_reconstruction(sample,rec)


	return 	



if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		pca_pose_reconstruction()
	else:
		sample_path = sys.argv[1]
		sample = SMPLLoader(sample_path)
		smpl_params = retarget_sample(sample,save_path='SMPL')



