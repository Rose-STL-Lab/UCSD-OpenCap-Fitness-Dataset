
# Data loader for .mot files
import os 
import numpy as np
import tqdm
from utils import DATA_DIR, set_logger
import pandas as pd 
import tqdm

logger = set_logger(task_name="opencap_loader")

# Load simulation for data for given list of sessions
class OpenCapLoader: 
    def __init__(self, subject_name):
        self.subject_name = subject_name

    def load_sqauat_trial_kinematics(self):
        """
            Returns a list of sqaut kinematics.
        """

        # Already loaded data before. Sending cached data
        if hasattr(self, 'mocap_headers') and hasattr(self, 'trial_kinematics'):
            return self.trial_kinematics 


        self.mocap_headers = None # Store the co-odrinates of mocap data for each subject
        self.trial_kinematics = {} # [trial_name] = R^{TxD} matrix
        
        kinematics_path = os.path.join(DATA_DIR,'Data', self.subject_name, 'OpenSimData','Kinematics') 

        if not os.path.isdir(kinematics_path):
            logger.warning(f"No kinematics found for subject:{self.subject_name}")
            return None

        for trial_name in os.listdir(kinematics_path): 
            
            if not trial_name.endswith(".mot"): continue  # If not mot file, skip

            if not 'SQT' in trial_name.upper(): continue
                      
            trial_path = os.path.join(kinematics_path, trial_name)
            mot_headers = OpenCapLoader.read_header(trial_path,header_line=10)  
            if len(mot_headers) == 0:
                print(f"Unable to load headers for file:", subject_name,trial_name, trial_path)
                continue 
        
            if self.mocap_headers is None:
                self.mocap_headers = mot_headers.copy()
            else:
                assert self.mocap_headers == mot_headers, "Headers do not match for mocap files"

            mot_headers.remove('time') # Remove time from headers
            kinematics = OpenCapLoader.storage_to_dataframe(trial_path, mot_headers) 
            self.trial_kinematics[trial_name.replace('.mot', '')] = kinematics
                        

        if self.mocap_headers is not None and len(self.trial_kinematics) > 0:
            return self.trial_kinematics
        else:
            logger.warning(f"No trials found for subject:{self.subject_name}")
            return None

    def load_simulation_data(self):    
        subject_name = self.subject_name

        simulation_results_path = os.path.join(DATA_DIR,'Data', subject_name, 'OpenSimData','Dynamics')

        for trial_name in os.listdir(simulation_results_path): 
            
            if trial_name == "SQT01": # Contains same results as segment-1 
                continue
            
            trial = {}
            
            kinetics_path = os.path.join(simulation_results_path, trial_name,f"kinetics_{trial_name}_muscle_driven.mot")
            mot_headers = OpenCapLoader.read_header(kinetics_path,header_line=6)
                        
            if len(mot_headers) == 0:
                print(f"Unable to load headers for file:", subject_name,trial_name, kinetics_path)
                continue 
            
            # Remove time from headers
            # 

            kinetics = OpenCapLoader.storage_to_dataframe(kinetics_path, mot_headers.remove('time'))
            
            trial['kinetics'] = kinetics

            kinematics_path = os.path.join(simulation_results_path, trial_name,f"kinematics_activations_{trial_name}_muscle_driven.mot")
            mot_headers = OpenCapLoader.read_header(kinematics_path,header_line=10)

            if len(mot_headers) == 0:
                print(f"Unable to load headers for file:", subject_name,kinematics_path)
                continue 
            # print("Headers:", mot_headers)
            # Remove time from headers
            mot_headers.remove('time')

            kinematics = OpenCapLoader.storage_to_dataframe(kinematics_path, mot_headers)
            trial['kinematics'] = kinematics

            # subject['dof_names'] = kinematics.columns.tolist()    
            
            # if len(trial) > 0:
                # subject[trial_name] = trial        
                # print("Loaded data for:", subject_name, trial_name, "Kinetics:", subject[trial_name]['kinetics'].shape, "Kinematics:", subject[trial_name]['kinematics'].shape)            

        # if len(subject) > 0:
        #     subjects[subject_name] = subject
        #     return subject 
        # else: 
        #     print("No trials found for subject:", subject_name)
        #     return None


    @staticmethod
    def read_header(mot_file,header_line=10): 
        if not os.path.isfile(mot_file): 
            return []

        try:         
            with open(mot_file, 'r') as f:
                lines = f.readlines()
                headers = lines[header_line]
                headers = headers.split()
                return headers
        except Exception as e:
            print(f"Unable to load headers for file:{mot_file}. Error:{e}")
            return [] 

    @staticmethod
    def storage_to_numpy(storage_file, excess_header_entries=0):
        """Returns the data from a storage file in a numpy format. Skips all lines
        up to and including the line that says 'endheader'.
        Parameters
        ----------
        storage_file : str
            Path to an OpenSim Storage (.sto) file.
        Returns
        -------
        data : np.ndarray (or numpy structure array or something?)
            Contains all columns from the storage file, indexable by column name.
        excess_header_entries : int, optional
            If the header row has more names in it than there are data columns.
            We'll ignore this many header row entries from the end of the header
            row. This argument allows for a hacky fix to an issue that arises from
            Static Optimization '.sto' outputs.
        Examples
        --------
        Columns from the storage file can be obtained as follows:
            >>> data = storage2numpy('<filename>')
            >>> data['ground_force_vy']
        """
        # What's the line number of the line containing 'endheader'?
        f = open(storage_file, 'r')

        header_line = False
        for i, line in enumerate(f):
            if header_line:
                column_names = line.split()
                break
            if line.count('endheader') != 0:
                line_number_of_line_containing_endheader = i + 1
                header_line = True
        f.close()
        # With this information, go get the data.
        if excess_header_entries == 0:
            names = True
            skip_header = line_number_of_line_containing_endheader
        else:
            names = column_names[:-excess_header_entries]
            skip_header = line_number_of_line_containing_endheader + 1
        data = np.genfromtxt(storage_file, names=names,
                skip_header=skip_header)

        new_data = []
        for d in data:
            new_data.append(list(d))
        new_data = np.array(new_data)

        return data


    @staticmethod
    def storage_to_dataframe(storage_file, headers):
        # Extract data
        data = OpenCapLoader.storage_to_numpy(storage_file)
        data = np.array(data)
        new_data = []
        for d in data:
            new_data.append(list(d))
        new_data = np.array(new_data)
        header_mapping = {header:i for i,header in enumerate(headers)}

        out = pd.DataFrame(data=data['time'], columns=['time'])
        for count, header in enumerate(headers):
            out.insert(count + 1, header, new_data[:,count+1])    
        
        return out




if __name__ == '__main__':
    

    logger = set_logger(task_name="temporal_segmentation")

    # PPE Files containing with MCS Scores
    mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]

    mcs_scores = [4,4,2,3,2,4,3,3,2,3,0,3,4,2,2,3,4,4,3,3,3 ]
    mcs_scores = dict(zip(mcs_sessions,mcs_scores))

    PPE_Subjects = ["PPE09182201","PPE09182202","PPE09182203","PPE09182204","PPE09182205","PPE09182206","PPE09182207","PPE09182208","PPE09182209","PPE091822010","PPE09182211","PPE09182212","PPE09182213","PPE09182214","PPE09182215","PPE09182216","PPE09182217","PPE09182218","PPE09182219","PPE09182220","PPE09182221"]
    PPE_Subjects = dict(zip(mcs_sessions,PPE_Subjects))

    subjects = {}

    mcs_sessions = os.listdir(os.path.join(DATA_DIR, 'Data'))
    for subject_name in tqdm.tqdm(mcs_sessions):
        if not os.path.isdir(os.path.join(DATA_DIR, 'Data',  subject_name)): continue
        subject_squat = OpenCapLoader(subject_name).load_sqauat_trial_kinematics()
        if subject_squat is not None:        
            subjects[subject_name] = subject_squat

    