# This file will sync our BITTE environment with the latest environment on the server. 
# Addiotionally it can sync some source enviroment to a target.  
# The changes in the source directory will be transferred to the target directory. 
## The source and target directory need not be on the same system. We will use ssh-keys to transfer using scp
import os
import json
import options_database_sync

#### Enviroment to be synced across systems
class Enviroment: 
	def __init__(self,path):
		self.path = os.path.abspath(path) # Directory location
		assert os.path.isdir(self.path), f"{self.path} must exist and be dir"

	def get_database_info(self): 
		
		# Check if the db.json exists, this file contains 
		return os.walk(self.path)

	def listdir(self):

		return os.listdir(self.path)

class OnlineEnviroment(Enviroment): 
	def __init__(self, path, local_machine=True, ssh_key=None, user="ubuntu", ip_address="127.0.0.1"):
		self.path = path # Directory location

		self.local_machine = local_machine

		# Online access to environement details 
		self.ssh_key = ssh_key
		self.user = user
		self.ip_address = ip_address


		assert local_machine and ssh_key is not None, "Online Enviroment needs ssh_key"







class DataDirectory: 
	def __init__(self,path=None,children={},maybe_dir=True):
		self.path = path
		self.children = children
		self.maybe_dir = maybe_dir

	def __repr__(self): 
		return f"Path:{self.path} children:{self.children.keys()}"

	@classmethod
	def create_tree(cls, tree_command_output): 
		"""
			Given a list of filepaths, `tree_command_output`, 
				- obtained by running the tree command in the terminal or using os.walk
				- length N 

			The output, `self` 
				- is a node of the suffix-tree 
				- describing the folder directory

			@param
				tree_command_output : list

			@return
				DataDirectory     
		"""


		if len(tree_command_output) == 0: # Return none if list is empty 
			return None
		elif len(tree_command_output) == 1: # If only one element, might be dir 
			maybe_dir = False
		else: 
			maybe_dir = True	


		# Set the path/root as the first element of the list (visited the node)
		path = tree_command_output[0]

		children = {}

		# Find subtrees using the child nodes as the intermediate points 
		subtree_list = []
		subtree_st = 1
		N = len(tree_command_output)
		for subtree_en in range(subtree_st,N):   
			d = tree_command_output[subtree_en]

			if os.path.dirname(d) == path: # Child of the current node
				file_name = os.path.basename(d)

				tree_command_output_child_sublist = tree_command_output[subtree_st:subtree_en]
				child = DataDirectory.create_tree(tree_command_output_child_sublist)

				if child is not None: 
					children[file_name] = child

				subtree_st = subtree_en

		assert all([os.path.dirname(children[x].path) == path for x in children]), "Parent of every children should be self.path"	
		
		return cls(path=path, children=children, maybe_dir=maybe_dir)

	@classmethod
	def create_tree_from_file(cls,file_path):
		assert os.path.isfile(file_path), f"File Path:{file_path} does not exist." 

		with open(file_path,'r') as f: 
			tree_command_output = f.read().split('\n')

		return DataDirectory.create_tree(tree_command_output)

	@classmethod
	def create_tree_for_local_data(cls,dir_path):
		assert os.path.isdir(dir_path), f"Folder Path:{dir_path} does not exist." 

		dir_path = os.path.abspath(dir_path)

		tree_command_output = os.popen(f'find -L {dir_path}').read().split("\n")
		# Can also use other commands like tree -if {dir_path}
		return DataDirectory.create_tree(tree_command_output)

	
	def difference(self, other): 
		# Find the difference of the tree with another tree, 
		# 2 nodes are different if the path and the children
		# If all the children are different, then the node is returned
		# If path is different, then the node is returned
		# If some of the children are different, then return the node with the children are returned
		if self.path != other.path: 
			return self.path


def tests_tree_creation(args):

	logger = options_database_sync.set_logger()

	# Test values from remote system  
	logger.info(DataDirectory.create_tree_from_file(os.path.join(args.source,'NorthServer_files.txt')))

	# Test directory on local system  
	logger.info(DataDirectory.create_tree_for_local_data(os.path.join(args.source,'DATA')))

	# Test file containing target directory information
	source_files = os.popen(f'tree -if {args.source}/DATA').read().split("\n")
	logger.info(DataDirectory.create_tree(source_files).__repr__())


if __name__ == "__main__": 	

	import options_database_sync
	args = options_database_sync.parse_database_sync_options()

	tests_tree_creation(args) # Test whether we can load tree for souce


	# Source Environment
	if os.path.isfile(args.source): 
		source_env = Enviroment(args.source,local_machine=True)
	else: 
		source_env = OnlineEnviroment(args.source,local_machine=False, ssh_key=args.source_ssh_key)

	# Target Environment
	if os.path.isfile(args.target): 
		target_env = Enviroment(args.target,local_machine=True)
	else: 
		target_env = OnlineEnviroment(args.target,local_machine=False, ssh_key=args.target_ssh_key)