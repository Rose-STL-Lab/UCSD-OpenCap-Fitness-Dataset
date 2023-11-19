import os
import sys 
import logging
import numpy as np
from tensorboardX import SummaryWriter

########################## SET RANDOM SEEDS #################################
np.random.seed(2)

############################# LIBRARY IMPORTS ####################################################
assert __file__[-1] != '/' , f'File:{__init__}, cannot be parsed' 
SRC_PATH,_ = os.path.split(os.path.abspath(__file__))
HOME_PATH,_ = os.path.split(SRC_PATH)
RABIT_PATH  = os.path.join(HOME_PATH,'RaBit')
STYLEGAN_PATH = os.path.join(RABIT_PATH,'stylegan3')

sys.path.extend([HOME_PATH,RABIT_PATH,STYLEGAN_PATH])


# ############################ FOLDER PATHS #######################################################
# Files
DATASET_PATH = os.path.join(HOME_PATH,'OpenSim') # Path containing all the training data (currently using xyz)
SMPL_PATH = os.path.join(HOME_PATH,'SMPL')
RENDER_PATH = os.path.join(HOME_PATH,'rendered_videos')
LOG_PATH = os.path.join(HOME_PATH,'logs')


# ############################ DATASET CONSTANTS #######################################################
# Excercise categories 
LABELS = ['LSLS', 'CMJ', 'PU', 'SQT', 'RSLS', 'BAPF', 'LLTF', 'LLT', 'RCMJ', 'PUF', 'BAP', 'RLTF', 'RLT', 'LCMJ'] 

# DATASET Skeleton Information
JOINT_NAMES = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

# Skeleton can be represented as a rooted tree. 
# I am rooting the skeleton at the neck. 
# JOINT_PARENT_ARRAY[joint] represens the index of the parent of the joint
JOINT_PARENT_ARRAY = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 13, 13, 13, 10, 10, 10]

ROOT_INIT_ROTVEC = np.array([0,np.pi/2,0])

##################################### RABIT Parameters #######################################
smpl2rabit_mapping = [ 0, # Center hip 
                       3, # spine0
                       6, # spine1
                       9, # spine3 
                       14,# right chest 
                       17,# right shoulder 
                       19,# right elblow 
                       21,# right ability ? 
                       23,# right hand 
                       12,# neck1
                       15,# neck2
                       2, # r hip 
                       5, # r knee
                       1, # l hip 
                       4, # l knee
                       8, # r ankle 
                       11,# r foot 
                       13,# l chest 
                       16,# lshoudler
                       18,# elbow
                       20,# lability 
                       22,# lhand
                       7, # lankle
                       10,# lfoor
                     ]


############################# RETARGETTING HYPERPARAMETERS #######################################################
cuda=True
RENDER=True

############################# LOGGING #######################################################
DEBUG = True
class CustomFormatter(logging.Formatter):

	BLACK = '\033[0;30m'
	RED = '\033[0;31m'
	GREEN = '\033[0;32m'
	BROWN = '\033[0;33m'
	BLUE = '\033[0;34m'
	PURPLE = '\033[0;35m'
	CYAN = '\033[0;36m'
	GREY = '\033[0;37m'

	DARK_GREY = '\033[1;30m'
	LIGHT_RED = '\033[1;31m'
	LIGHT_GREEN = '\033[1;32m'
	YELLOW = '\033[1;33m'
	LIGHT_BLUE = '\033[1;34m'
	LIGHT_PURPLE = '\033[1;35m'
	LIGHT_CYAN = '\033[1;36m'
	WHITE = '\033[1;37m'

	RESET = "\033[0m"

	format = "[%(filename)s:%(lineno)d]: %(message)s (%(asctime)s) "

	FORMATS = {
		logging.DEBUG: YELLOW + format + RESET,
		logging.INFO: GREY + format + RESET,
		logging.WARNING: LIGHT_RED + format + RESET,
		logging.ERROR: RED + format + RESET,
		logging.CRITICAL: RED + format + RESET
	}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)

def get_logger(task_name=None):

	os.makedirs(os.path.join(LOG_PATH, task_name),exist_ok=True)

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)

	handler = logging.FileHandler(os.path.join(LOG_PATH, task_name,"log.txt"))
	handler.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
	formatter = logging.Formatter(
		'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	handler = logging.StreamHandler()
	handler.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
	handler.setFormatter(CustomFormatter())
	logger.addHandler(handler)

	writer = SummaryWriter(os.path.join(LOG_PATH, task_name))

	return logger, writer

