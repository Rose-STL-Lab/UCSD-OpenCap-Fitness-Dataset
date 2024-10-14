import os
import argparse
import bitte_config as bc 

def parse_database_sync_options():

	options = argparse.ArgumentParser()

	# Default source: ubuntu@north.ucsd.edu -p 12700 -i <ssh-key>
	options.add_argument("-s", "--source", default="north", help="Source Enviroment")
	options.add_argument("--source-ssh-key", default=None, help="Source SSH Key to connect") 
	options.add_argument("--source-user", default="ubuntu", help="Username at the source enviroment") 
	options.add_argument("--source-ip-address", default=bc.NORTH['IP_ADDRESS'], help="Source Ip Address") 
	options.add_argument("--source-port", default=12700, help="Source's port number to connect") 
	
	# Default target: "UCSD-OpenCap-Fitness-Dataset/BITTE_ENV/"
	options.add_argument("-t", "--target", default="local", help="Target Enviroment") # By default merge the ROOT_DIR 
	options.add_argument("--target-ssh-key", default=None, help="Target Enviroment") # By default merge the ROOT_DIR 
	options.add_argument("--target-key", default=None, help="Target SSH Key to connect") 
	options.add_argument("--target-user", default=bc.USER, help="Username at the target enviroment") 
	options.add_argument("--target-ip-address", default=bc.IP_ADDRESS, help="target Ip Address") 
	options.add_argument("--target-port", default=12700, help="target's port number to connect") 
	
	args = options.parse_args()
	return args

############################# LOGGING #######################################################
import logging
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

def set_logger(task_name="debug"):


	os.makedirs(os.path.join(bc.LOG_DIR, task_name),exist_ok=True)

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)

	handler = logging.FileHandler(os.path.join(bc.LOG_DIR, task_name,"log.txt"))
	handler.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
	formatter = logging.Formatter(
		'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	handler = logging.StreamHandler()
	handler.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
	handler.setFormatter(CustomFormatter())
	logger.addHandler(handler)

	return logger

def set_wandb_writer(task_name="debug"):
	
	try: 
		from tensorboardX import SummaryWriter
		writer = SummaryWriter(os.path.join(bc.LOG_DIR, task_name))

	except ModuleNotFoundError:
		logger.warning("Unable to load tensorboardX to write summary.")
		writer = None

	return logger, writer