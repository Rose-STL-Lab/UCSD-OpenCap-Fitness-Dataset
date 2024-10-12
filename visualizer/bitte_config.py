####### Hardcode system information for BITTEE ##################### 

import os
import sys 
import random
import numpy as np
import argparse


########################## GET SYSTEM Information #####################
import platform
SYSTEM_OS = platform.system()


######################## NETWORK Information ######################### 
import socket
USER = socket.gethostname()
IP_ADDRESS = socket.gethostbyname(USER)


########################## SET RANDOM SEEDS ###########################
seed = 69
random.seed(seed)
np.random.seed(seed)

############################# RELATIVE PATH ###########################
assert __file__[-1] != '/' , f'File:{__file__}, cannot be parsed' 
SRC_DIR,_ = os.path.split(os.path.abspath(__file__))
HOME_DIR,_ = os.path.split(SRC_DIR)
sys.path.extend([HOME_DIR])

####################### SAVE/DATA FOLDER #############################
BITTE_DIR = os.path.join(HOME_DIR,'BITTE_ENV')
LOG_DIR = os.path.join(BITTE_DIR, 'logs')
SERVER_DIR = os.path.join(BITTE_DIR, 'server')

os.makedirs(SERVER_DIR,exist_ok=True)

######################### SERVER DETAILS ##############################
NORTH = {
    'ENV_DIR' : "/home/ubuntu/data/BITTE_ENV",
    'IP_ADDRESS' : 'north.ucsd.edu',
    'PORT' : 12700 }

YADI = {
    'ENV_DIR' : "/data/shubh/BITTE_ENV",
    'IP_ADDRESS' : 'roselab1.ucsd.edu',
    'USER': 'ubuntu',
    'PORT' : 12700 }