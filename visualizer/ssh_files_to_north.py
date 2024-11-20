import os 
import glob

files_to_copy = glob.glob("/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/Data/*/OpenSimData/Dynamics/*_segment_*/kinematics_activations_*_segment_*_muscle_driven.mot")

for source in files_to_copy:
	
	basename = os.path.basename(source)
	trial = os.path.basename(os.path.dirname(source))
	subject = source
	for i in range(4):
		subject = os.path.dirname(subject)
	subject = os.path.basename(subject)
	# print(source,trial,subject)

	destination = f"/data/panini/MCS_DATA/Data/{subject}/OpenSimData/Dynamics/{trial}/kinematics_activations_{trial}_muscle_driven.mot"

	# print(destination)
	print(f"scp -r -P12700 -i/home/shubh/Desktop/panini {source}  ubuntu@north.ucsd.edu:{destination}")
	os.system(f"scp -r -P12700 -i/home/shubh/Desktop/panini {source}  ubuntu@north.ucsd.edu:{destination}")