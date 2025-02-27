import os 
import tqdm
import numpy as np 
from motLoader import storage_to_dataframe, read_header





# Load simulation for data for given list of sessions
def load_simulation_data(mcs_sessions,data_path,surrogates=[]):
    subjects = {}

    for subject_name in tqdm.tqdm(mcs_sessions):

        if not os.path.isdir(os.path.join(data_path, subject_name)): continue
        simulation_results_path = os.path.join(data_path, subject_name, 'OpenSimData','Dynamics')

        if not os.path.isdir(simulation_results_path):
            continue
        
        subject = {}
        for trial_name in os.listdir(simulation_results_path): 
            
            if 'segment' not in trial_name: # Contains same results as segment-1 
                continue
            
            trial = {}
            
            kinetics_path = os.path.join(simulation_results_path, trial_name,f"kinetics_{trial_name}_muscle_driven.mot")
            mot_headers = read_header(kinetics_path,header_line=6)            
                        
            if len(mot_headers) == 0:
                print(f"Unable to load headers for file:", subject_name,trial_name, kinetics_path)
                continue 
            
            # Remove time from headers
            mot_headers.remove('time')

            kinetics = storage_to_dataframe(kinetics_path, mot_headers)
            
            trial['kinetics'] = kinetics

            kinematics_path = os.path.join(simulation_results_path, trial_name,f"kinematics_activations_{trial_name}_muscle_driven.mot")
            mot_headers = read_header(kinematics_path,header_line=10)

            if len(mot_headers) == 0:
                print(f"Unable to load headers for file:", subject_name,kinematics_path)
                continue 
            # print("Headers:", mot_headers)
            # Remove time from headers
            mot_headers.remove('time')

            kinematics = storage_to_dataframe(kinematics_path, mot_headers)
            trial['kinematics'] = kinematics

            subject['dof_names'] = kinematics.columns.tolist()    
            
            

            # Load activations predicted by some surrogate model
            for surrogate in surrogates:
                surrogate_results_path = os.path.join(surrogate, f'{subject_name}-{trial_name}.mot')
                if os.path.isfile(surrogate_results_path):
                    mot_headers = read_header(surrogate_results_path,header_line=10)
                    if len(mot_headers) == 0:
                        print(f"Unable to load surrogate headers for file:", subject_name,trial_name, surrogate_results_path)
                        continue 
                    # print("Headers:", mot_headers)
                    # Remove time from headers
                    mot_headers.remove('time')

                    surrogate_data = storage_to_dataframe(surrogate_results_path, mot_headers)

                    if 'surrogate' not in trial:
                        trial['surrogate'] = {}

                    surrogate_name = os.path.basename(surrogate).replace('.mot','')
                    trial['surrogate'][surrogate_name] = surrogate_data

            if 'torques' not in trial:
                trial['torques'] = {}

            for surrogate in ['simulation'] + surrogates:
                print("See file:",os.path.join(simulation_results_path + '_' + os.path.basename(surrogate),trial_name, 'torques.npy'), os.path.exists(os.path.join(simulation_results_path + '_' + os.path.basename(surrogate),trial_name, 'torques.npy'))) 
                if os.path.exists(os.path.join(simulation_results_path + '_' + os.path.basename(surrogate),trial_name, 'torques.npy')): 


                    print("Loading torques for:", subject_name, trial_name)
                    joint_torque = np.load(os.path.join(simulation_results_path + '_' + os.path.basename(surrogate),trial_name, 'torques.npy'),allow_pickle=True).item()

                    opt = np.load(os.path.join(simulation_results_path + '_' + os.path.basename(surrogate),trial_name, 'optimaltrajectories.npy'),allow_pickle=True).item()
                    bh_W = (opt['muscle_driven']['torques_BWht']/opt['muscle_driven']['torques'])[0][0]

                    for joint in joint_torque:
                        for t in joint_torque[joint]:
                            if t == 'ID' or t == 'MT':
                                joint_torque[joint][t] = joint_torque[joint][t] * bh_W

                    surrogate_name = os.path.basename(surrogate).replace('.mot','')
                    trial['torques'][surrogate_name] = joint_torque 
                    # print(trial['torques'][surrogate_name]['hip_flexion_l']['MT'])

                        
            if len(trial) > 0:
                subject[trial_name] = trial        
                print("Loaded data for:", subject_name, trial_name, "Kinetics:", subject[trial_name]['kinetics'].shape, "Kinematics:", subject[trial_name]['kinematics'].shape)            
                # print("Surrogates:", trial['torques']['surrogate_quadruple_activations']['hip_flexion_l']['MT'])
                # print("Surrogates:", trial['torques']['surrogate_double_activations']['hip_flexion_l']['MT'])
        if len(subject) > 0:
            subjects[subject_name] = subject
        else: 
            print("No trials found for subject:", subject_name)
    
    return subjects