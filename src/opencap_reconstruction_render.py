import os 
import sys
import json
import numpy as np 
from tqdm import tqdm
import polyscope as ps 
import polyscope.imgui as psim

from utils import * 
from dataloader import OpenCapDataLoader,MultiviewRGB
from smpl_loader import SMPLRetarget
# from osim import OSIMSequence

class Visualizer: 
	def __init__(self): 
		
		ps.init()

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		# ps.set_view_projection_mode("orthographic")
		ps.set_ground_plane_mode('shadow_only')

	# Initialize 3D objects from a sample and set callback for
	def render_smpl_multi_view_callback(self,sample,video_dir=None):
		self.update_smpl_multi_view_callback(sample,video_dir=video_dir)
		ps.set_user_callback(self.callback)
		ps.show()	

	
	def update_smpl_multi_view_callback(self,sample,video_dir=None):
		assert hasattr(sample,'rgb'), "Error loading RGB Data. Don't know the camera details. Cannot render in multiple views"


		target = sample.joints_np
		verts,Jtr,Jtr_offset = sample.smpl()

		verts = verts.cpu().data.numpy()
		if not hasattr(self,'ps_data'):
			# Initialize Plot SMPL in polyscope
			ps.init()
			self.ps_data = {}
			self.ps_data['bbox'] = verts.max(axis=(0,1)) - verts.min(axis=(0,1))
			self.ps_data['object_position'] = sample.joints_np[0,0]

		ps.remove_all_structures()
		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		camera_position = np.array([5*self.ps_data['bbox'][0],0.5*self.ps_data['bbox'][1],0]) + self.ps_data['object_position']
		look_at_position = np.array([0,0,0]) + self.ps_data['object_position']
		ps.look_at(camera_position,look_at_position)



		Jtr = Jtr.cpu().data.numpy() + np.array([0,0,0])*self.ps_data['bbox']

		verts += np.array([0, 0, 0]) * self.ps_data['bbox']
		# target_joints = target - target[:,7:8,:] + Jtr[:,0:1,:] + np.array([0,0,0])*self.ps_data['bbox']
		target_joints = target + np.array([0,0,0])*self.ps_data['bbox']

		Jtr_offset = Jtr_offset[:,sample.smpl.index['smpl_index']].cpu().data.numpy() + np.array([0.0,0,0])*self.ps_data['bbox']       
		# Jtr_offset = Jtr_offset.cpu().data.numpy() + np.array([0,0,0])*self.ps_data['bbox']       

		ps_mesh = ps.register_surface_mesh('mesh',verts[0],sample.smpl.smpl_layer.smpl_data['f'],transparency=0.7)
		

		target_bone_array = np.array([[i,p] for i,p in enumerate(sample.smpl.index['dataset_parent_array'])])
		ps_target_skeleton = ps.register_curve_network(f"Target Skeleton",target_joints[0],target_bone_array,color=np.array([0,0,1]),enabled=False)

		smpl_bone_array = np.array([[i,p] for i,p in enumerate(sample.smpl.index['parent_array'])])
		ps_smpl_skeleton = ps.register_curve_network(f"Smpl Skeleton",Jtr[0],smpl_bone_array,color=np.array([1,0,0]),enabled=False)

		smpl_index = list(sample.smpl.index['smpl_index'].cpu().data.numpy())    

		offset_skeleton_bones = np.array([[x,smpl_index.index(sample.smpl.index['parent_array'][i])] for x,i in enumerate(smpl_index) if sample.smpl.index['parent_array'][i] in smpl_index])
		ps_offset_skeleton = ps.register_curve_network(f"Offset Skeleton",Jtr_offset[0],offset_skeleton_bones,color=np.array([1,1,0]),enabled=False)


		dataset_index = list(sample.smpl.index['dataset_index'].cpu().data.numpy())    		
		joint_mapping = np.concatenate([target_joints[0,dataset_index],Jtr_offset[0]],axis=0)
		joint_mapping_edges = np.array([(i,joint_mapping.shape[0]//2+i) for i in range(joint_mapping.shape[0]//2)])
		ps_joint_mapping = ps.register_curve_network(f"Mapping (target- smpl) joints",joint_mapping,joint_mapping_edges,radius=0.001,color=np.array([0,1,0]),enabled=False)

		ps_cams = []
		# Set indivdual cameras 
		for i,camera in enumerate(sample.rgb.cameras): 
			intrinsics = ps.CameraIntrinsics(fov_vertical_deg=camera['fov_x'], fov_horizontal_deg=camera['fov_y'])
			# extrinsics = ps.CameraExtrinsics(mat=np.eye(4))
			extrinsics = ps.CameraExtrinsics(root=camera['position'], look_dir=camera['look_dir'], up_dir=camera['up_dir'])
			params = ps.CameraParameters(intrinsics, extrinsics)
			ps_cam = ps.register_camera_view(f"Cam{i}", params)
			print("Camera:",params.get_view_mat())
			ps_cams.append(ps_cam)


		ps_biomechnical = ps.register_surface_mesh("Ground Truth (Skeleton)",sample.osim.vertices[0],sample.osim.faces,transparency=1.0,color=np.array([60,150,60])/255,smooth_shade=True,material='wax')
		ps_biomechnical_joints = ps.register_point_cloud("Ground Truth (Joints)",sample.osim.joints[0],color=np.array([0,0,0]))

		ps_biomechnical_pred = ps.register_surface_mesh("Reconstruction (Skeleton)",sample.osim_pred.vertices[0],sample.osim.faces,transparency=1.0,color=np.array([200,50,50])/255,smooth_shade=True,material='wax')
		ps_biomechnical_joints_pred = ps.register_point_cloud("Reconstruction (Joints)",sample.osim_pred.joints[0],color=np.array([0,0,0]))


		# Create random colors of each segment
		# colors = np.random.random((sample.segments.shape[0],3))
		# mesh_colors = np.zeros((verts.shape[0],3))
		# mesh_colors[:,1] = 0.3 # Default color is light blue
		# mesh_colors[:,2] = 1 # Default color is light blue
		# for i,segment in enumerate(sample.segments):
		# 	mesh_colors[segment[0]:segment[1]] = colors[i:i+1]


		# self.ps_data = {
		# 	"experiment_options": self.exps,
        #     "experiment_options_selected": self.exps[0],

        #     "category_options": self.categories,
        #     "category_options_selected": self.categories[1],

        #     "trial": 1,
		# }

		# self.ps_data = {}

		# Map all polyscope objects to ps_data
		self.ps_data['ps_mesh'] = ps_mesh
		self.ps_data['ps_target_skeleton'] = ps_target_skeleton
		self.ps_data['ps_smpl_skeleton'] = ps_smpl_skeleton
		self.ps_data['ps_offset_skeleton'] = ps_offset_skeleton
		self.ps_data['ps_joint_mapping'] = ps_joint_mapping
		self.ps_data['ps_cams'] = ps_cams
		self.ps_data['ps_biomechnical'] = ps_biomechnical
		self.ps_data['ps_biomechnical_joints'] = ps_biomechnical_joints

		self.ps_data['ps_biomechnical_pred'] = ps_biomechnical_pred
		self.ps_data['ps_biomechnical_joints_pred'] = ps_biomechnical_joints_pred

		# Map all the animation data
		self.ps_data['verts'] = verts
		self.ps_data['target_joints'] = target_joints
		self.ps_data['Jtr'] = Jtr
		self.ps_data['Jtr_offset'] = Jtr_offset
		self.ps_data['dataset_index'] = dataset_index
		self.ps_data['biomechanical'] = sample.osim.vertices
		self.ps_data['biomechanical_joints'] = sample.osim.joints

		self.ps_data['biomechanical_pred'] = sample.osim_pred.vertices
		self.ps_data['biomechanical_joints_pred'] = sample.osim_pred.joints


		# Map all the rendering information
		self.ps_data['video_dir'] = video_dir
		self.ps_data['label'] = sample.label
		self.ps_data['recordAttempt'] = sample.recordAttempt
		self.ps_data['fps'] = sample.fps

		# Animation details  
		self.ps_data['t'] = 0 
		self.ps_data['T'] = min(verts.shape[0], sample.osim.vertices.shape[0], sample.osim_pred.vertices.shape[0])
		self.ps_data['is_paused'] = False
		self.ps_data['ui_text'] = "Enter Instructions here"


		self.ps_data['session_options_selected'] = sample.openCapID
		self.ps_data['session_options'] = [ x.replace("OpenCapData_","") for x in  os.listdir(INPUT_DIR)]

		# Load info about other trial samples.
		other_session_trials_details = [ os.path.join(os.path.dirname(sample.sample_path), x)  for x in os.listdir(os.path.dirname(sample.sample_path))]
		other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]
		
		self.ps_data['category_options_selected'] = sample.label 
		self.ps_data['category_options'] = list(set([x[0] for x in other_session_trials_details]))
		

		self.ps_data['trial_options_selected'] = sample.recordAttempt_str
		self.ps_data['trial_options'] = [ x[1] for x in other_session_trials_details if x[0] == self.ps_data['category_options_selected']]




	def callback_render(self):

		verts = self.ps_data['verts']
		ps_mesh = self.ps_data['ps_mesh']
		target_joints = self.ps_data['target_joints']
		Jtr = self.ps_data['Jtr']
		Jtr_offset = self.ps_data['Jtr_offset']
		biomechanical = self.ps_data['biomechanical']
		biomechanical_joints = self.ps_data['biomechanical_joints']

		biomechanical_pred = self.ps_data['biomechanical_pred']
		biomechanical_joints_pred = self.ps_data['biomechanical_joints_pred']

		ps_target_skeleton = self.ps_data['ps_target_skeleton']
		ps_smpl_skeleton = self.ps_data['ps_smpl_skeleton']
		ps_offset_skeleton = self.ps_data['ps_offset_skeleton']
		ps_joint_mapping = self.ps_data['ps_joint_mapping']	

		ps_biomechnical = self.ps_data['ps_biomechnical']
		ps_biomechnical_joints = self.ps_data['ps_biomechnical_joints']

		ps_biomechnical_pred = self.ps_data['ps_biomechnical_pred']
		ps_biomechnical_joints_pred = self.ps_data['ps_biomechnical_joints_pred']

		dataset_index = self.ps_data['dataset_index']




		video_dir = self.ps_data['video_dir']

		if video_dir is None: 
			ps.warning("Location to render not specefied. Setting to <current working directory>/render as default")
			video_dir = os.path.join(os.getcwd(),"render")

		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)



		print(f'Rendering images:')
		for i in tqdm(range(self.ps_data['T'])):
			
			ps_mesh.update_vertex_positions(verts[i])
			# ps_mesh.set_color(mesh_colors[i])

			ps_target_skeleton.update_node_positions(target_joints[i])
			ps_smpl_skeleton.update_node_positions(Jtr[i])
			ps_offset_skeleton.update_node_positions(Jtr_offset[i])
			ps_joint_mapping.update_node_positions(np.concatenate([target_joints[i,dataset_index],Jtr_offset[i]],axis=0))

			ps_biomechnical.update_vertex_positions(biomechanical[i])
			ps_biomechnical_joints.update_point_positions(biomechanical_joints[i])

			ps_biomechnical_pred.update_vertex_positions(biomechanical_pred[i])
			ps_biomechnical_joints_pred.update_point_positions(biomechanical_joints_pred[i])

			image_path = os.path.join(video_dir,"images",f"smpl_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png")
			ps.screenshot(image_path,transparent_bg=False)
			

		image_path = os.path.join(video_dir,"images",f"smpl_\%d.png")
		video_path = os.path.join(video_dir,"video",f"{self.ps_data['label']}_{self.ps_data['recordAttempt']}_smpl.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = self.ps_data['fps']
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")


	def read_display_size(self):
		polyscope_ini_path = os.path.join(os.getcwd(),'.polyscope.ini')
		if not os.path.exists(polyscope_ini_path):
			self.display_size = (1280,720)
			ps.warning("Polyscope.ini not found, defaulting to 1280x720")
		else:
			with open(polyscope_ini_path) as f: 
				display_data = json.load(f)

			self.display_size = (display_data['windowWidth'], display_data['windowHeight'])

		
		return self.display_size



	def callback(self):
		
		########### Checks ############
		# Ensure self.t lies between 
		self.ps_data['t'] %= self.ps_data['T']


		

		### Update animation based on self.t
		if 'ps_mesh' in self.ps_data:
			t = self.ps_data['t']
			self.ps_data['ps_mesh'].update_vertex_positions(self.ps_data['verts'][t])
			# ps_mesh.set_color(mesh_colors[i])
			self.ps_data['ps_target_skeleton'].update_node_positions(self.ps_data['target_joints'][t])
			self.ps_data['ps_smpl_skeleton'].update_node_positions(self.ps_data['Jtr'][t])
			self.ps_data['ps_offset_skeleton'].update_node_positions(self.ps_data['Jtr_offset'][t])
			self.ps_data['ps_joint_mapping'].update_node_positions(np.concatenate([self.ps_data['target_joints'][t,self.ps_data['dataset_index']],self.ps_data['Jtr_offset'][t]],axis=0))

			self.ps_data['ps_biomechnical'].update_vertex_positions(self.ps_data['biomechanical'][t])
			self.ps_data['ps_biomechnical_joints'].update_point_positions(self.ps_data['biomechanical_joints'][t])

			self.ps_data['ps_biomechnical_pred'].update_vertex_positions(self.ps_data['biomechanical_pred'][t])
			self.ps_data['ps_biomechnical_joints_pred'].update_point_positions(self.ps_data['biomechanical_joints_pred'][t])

		
		if not self.ps_data['is_paused']: 
			self.ps_data['t'] += 1 


		# Check keyboards for inputs
		
		# Check for spacebar press to toggle pause
		if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)):
			
			self.ps_data['is_paused'] = not self.ps_data['is_paused']

		# Left arrow pressed
		if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)):
			self.ps_data['t'] -= 1

		if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)):
			self.ps_data['t'] += 1

		# Input text
		changed, self.ps_data["ui_text"] = psim.InputText("- Coach Instructions", self.ps_data["ui_text"])


		############## Create a seperate window for the GUI to update the animations 
		# psim.Begin("Video Controller",True)

		# Get display size 
		# if not hasattr(self, 'display_size'):
		# 	self.display_size = self.read_display_size()


		# if self.display_size[0] > self.display_size[1]: # With of polyscope is larger than height
		# gui_pos = ( int(self.display_size[0]*0.8), int(self.display_size[1]*0.1) )
		# gui_size = ( int(self.display_size[0]*0.195), int(self.display_size[1]*0.25) )


		# # Ensure that the gui_pos is not an odd number
		# gui_pos = ( gui_pos[0]-1 if gui_pos[0]%2==1 else gui_pos[0], gui_pos[1]-1 if gui_pos[1]%2==1 else gui_pos[1]   ) 
		# gui_size = ( gui_size[0]-1 if gui_size[0]%2==1 else gui_size[0], gui_size[1]-1 if gui_size[1]%2==1 else gui_size[1]   ) 

		# print(psim.GetWindowPos(), psim.GetWindowSize(),self.display_size, gui_pos)
		# psim.SetWindowPos(gui_pos,1) # Set the position the window at the bottom of the GUI
		# psim.SetWindowSize(gui_size,1)

		# Create a floater to show the timestep and adject self.t accordingly
		changed, self.ps_data['t'] = psim.SliderInt("", self.ps_data['t'], v_min=0, v_max=self.ps_data['T'])
		psim.SameLine()

		# Create a render button which when pressed will create a .mp4 file
		if psim.Button("<"):
			self.ps_data['t'] -= 1
		
		psim.SameLine()
		if psim.Button("Play Video" if self.ps_data['is_paused'] else "Pause Video"):
			self.ps_data['is_paused'] = not self.ps_data['is_paused']

		psim.SameLine()
		if psim.Button(">"):
			self.ps_data['t'] += 1

		# psim.SameLine()
		if psim.Button("Render Video"):
			self.callback_render()        

		if(psim.TreeNode("Load other samples")):





		
		

			# psim.TextUnformatted("Load Optimized samples")

			changed = psim.BeginCombo("- Experiement", self.ps_data["session_options_selected"])
			if changed:				
				for val in self.ps_data["session_options"]:
					_, selected = psim.Selectable(val, selected=self.ps_data["session_options_selected"]==val)
					if selected:
						self.ps_data["session_options_selected"] = val

						sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}","MarkerData")
						other_session_trials_details = [ os.path.join(sample_path, x)  for x in os.listdir(sample_path)]
						other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]

						self.ps_data['category_options'] = list(set([x[0] for x in other_session_trials_details]))
						self.ps_data['trial_options'] = [ x[1] for x in other_session_trials_details if x[0] == self.ps_data['category_options_selected']]

				psim.EndCombo()

			changed = psim.BeginCombo("- Excercise Type", self.ps_data["category_options_selected"])
			if changed:
				for val in self.ps_data["category_options"]:
					_, selected = psim.Selectable(val, selected=self.ps_data["category_options_selected"]==val)
					if selected:
						self.ps_data["category_options_selected"] = val

						# Load info about other trial samples.
						sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}","MarkerData")
						other_session_trials_details = [ os.path.join(sample_path, x)  for x in os.listdir(sample_path)]
						other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]
						self.ps_data['category_options'] = list(set([x[0] for x in other_session_trials_details]))
				psim.EndCombo()

			changed = psim.BeginCombo("- Trial", self.ps_data["trial_options_selected"])
			if changed:
				for val in self.ps_data["trial_options"]:
					_, selected = psim.Selectable(val, selected=self.ps_data["trial_options_selected"]==val)
					if selected:
						self.ps_data["trial_options_selected"] = val


						# Load info about other trial samples.
						sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}","MarkerData")
						other_session_trials_details = [ os.path.join(sample_path, x)  for x in os.listdir(sample_path)]
						other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]

						self.ps_data['trial_options'] = [ x[1] for x in other_session_trials_details if x[0] == self.ps_data['category_options_selected']]
				psim.EndCombo()


			
			if(psim.Button("Load Optimized samples")):
				sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}")
				sample_path = os.path.join(sample_path, "MarkerData")
				sample_path = os.path.join(sample_path,f"{self.ps_data['category_options_selected']}{self.ps_data['trial_options_selected']}.trc")
				sample = load_subject(sample_path)
				self.update_smpl_multi_view_callback(sample)
			psim.TreePop()


		# psim.End()



# Load file and render skeleton for each video
def render_dataset():
	video_dir = RENDER_DIR
	
	vis = Visualizer()
	
	for subject in os.listdir(INPUT_DIR):
		for sample_path in os.listdir(os.path.join(INPUT_DIR,subject,'MarkerData')):
			sample_path = os.path.join(INPUT_DIR,subject,'MarkerData',sample_path)
			render_smpl(sample_path,vis,video_dir=video_dir)

def render_smpl(sample_path,vis,video_dir=None): 
	"""
		Render dataset samples 
			
		@params
			sample_path: Filepath of input
			video_dir: Folder to store Render results for the complete worflow  
		
			
		Load input (currently .trc files) and save all the rendering videos + images (retargetting to smp, getting input text, per frame annotations etc.) 
	"""
	sample_path = os.path.abspath(sample_path)

	sample = load_subject(sample_path)
	
	
	if video_dir is not None:
		video_dir = os.path.join(video_dir,f"{sample.openCapID}_{sample.label}_{sample.recordAttempt}")
		if os.path.isfile(os.path.join(video_dir,f"{sample.label}_{sample.recordAttempt}_smpl.mp4")): 
			return
	
	# Visualize each view  
	vis.render_smpl_multi_view_callback(sample,video_dir=video_dir)



def load_subject(sample_path):
	sample = OpenCapDataLoader(sample_path)
	
	# Visualize Target skeleton
	# vis.render_skeleton(sample,video_dir=video_dir)

	# Load SMPL
	sample.smpl = SMPLRetarget(sample.joints_np.shape[0],device=None)	
	sample.smpl.load(os.path.join(SMPL_DIR,sample.name+'.pkl'))

	# Visualize SMPL
	# vis.render_smpl(sample,sample.smpl,video_dir=video_dir)
	# vis.render_smpl(sample,sample.smpl,video_dir=None)
	
	# Load Video
	sample.rgb = MultiviewRGB(sample)

	print(f"Session ID: {sample.name} SubjectID:{sample.rgb.session_data['subjectID']} Action:{sample.label}")

	# Load LaiArnoldModified2017
	from osim import OSIMSequence
	osim_path = os.path.dirname(os.path.dirname(sample.sample_path)) 
	osim_path = os.path.join(osim_path,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')
	osim_geometry_path = os.path.join(DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')
	
	mot_path = os.path.dirname(os.path.dirname(sample.sample_path))
	mot_path = os.path.join(mot_path,'OpenSimData','Kinematics',sample.label+ sample.recordAttempt_str + '.mot')
	print("Loading User motion file:",mot_path)


	# For Plotting dyanmics
	# osim_path = os.path.dirname(os.path.dirname(sample.sample_path)) 
	# osim_path = os.path.join(osim_path,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted_contacts.osim')
	# osim_geometry_path = os.path.join(DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')
	
	# mot_path = os.path.dirname(os.path.dirname(sample.sample_path))
	# mot_path = os.path.join(mot_path,'OpenSimData','Dynamics',sample.label+ sample.recordAttempt_str, f'kinematics_activations_{sample.label+ sample.recordAttempt_str}_torque_driven.mot')
	# mot_path = os.path.join(mot_path,'OpenSimData','Dynamics',sample.label+ sample.recordAttempt_str, f'kinematics_activations_{sample.label+ sample.recordAttempt_str}_muscle_driven.mot')

	sample.osim = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )


	mot_path = os.path.dirname(os.path.dirname(sample.sample_path))
	mot_path = os.path.join(mot_path,'OpenSimData','Pred_Kinematics',sample.label+ sample.recordAttempt_str + '.mot')
	print("Loading Reconstrction file:",mot_path)

	# sample.osim_pred = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )	


	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/train_forward_pass/mot_output/0.mot"
	mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/constrained_latents/entry_0.mot"
	# print("Loading Generatrion file:",mot_path)
	sample.osim_pred = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )	
	print("MOT DATA:",sample.osim_pred.motion.shape)
	print("TIME:",sample.osim_pred.motion[:,0])
	print("Pelivs:",sample.osim_pred.motion[:,1:3])
	# print("New SIM model")


	# Load Segments
	if os.path.exists(os.path.join(SEGMENT_DIR,sample.name+'.npy')):
		sample.segments = np.load(os.path.join(SEGMENT_DIR,sample.name+'.npy'),allow_pickle=True).item()['segments']


	return sample

if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		render_dataset()
	else:
		sample_path = sys.argv[1]
		vis = Visualizer()
		video_dir = sys.argv[2] if len(sys.argv) > 2 else None
		render_smpl(sample_path,vis,video_dir)

