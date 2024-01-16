# Code to test each module of the pipeline
import os
from utils import * 

def test_smpl_retargetting(): 

	os.system("python3 src/retarget2smpl.py --file ./data/OpenSim/OpenCapData_015b7571-9f0b-4db4-a854-68e57640640d/MarkerData/SQT01.trc  --force")

if __name__ == "__main__": 
	test_smpl_retargetting()