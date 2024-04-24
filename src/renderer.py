import os 
import sys
import numpy as np 
from tqdm import tqdm
import polyscope as ps 
import trimesh 

from utils import * 
from dataloader import OpenCapDataLoader,MultiviewRGB
from smpl_loader import SMPLRetarget

class Visualizer: 
	def __init__(self): 
		
		ps.init()

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		# ps.set_view_projection_mode("orthographic")
		# ps.set_ground_plane_mode('shadow_only')

	def render_skeleton(self,sample,video_dir=None,screen_scale=[1.4,1.0,1.0],frame_rate=60): 
		"""
			
			screen_scale scales the bounding box 
		"""
		joints = sample.joints_np
		bones = np.array([(i,x) for i,x in enumerate(JOINT_PARENT_ARRAY)])

		# Initialise
		self.bbox = joints[0].max(axis=0) - joints[0].min(axis=0)
		self.bbox *= np.array(screen_scale)

		self.object_position = joints[0,0]

		camera_position = np.array([10*self.bbox[0],self.bbox[1],0]) + self.object_position
		look_at_position = np.array([0,0,0]) + self.object_position
		ps.look_at(camera_position,look_at_position)

		ps_joints   = ps.register_point_cloud("Joints", joints[0],enabled=True,color=np.array([1,0,0]),radius=0.01)
		ps_skeleton = ps.register_curve_network("Skeleton", joints[0], bones,color=np.array([0,1,0]),enabled=True)


		if video_dir is None:
			ps.show()
			return 

		video_dir = os.path.join(video_dir,f"{sample.openCapID}_{sample.label}_{sample.recordAttempt}")
		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)

		# Render each frame
		for i,traj in enumerate(joints): 
			ps_joints.update_point_positions(traj)
			ps_skeleton.update_node_positions(traj)


			image_path = os.path.join(video_dir,"images",f"{i}.png")
			print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png");
			ps.screenshot(image_path,transparent_bg=False)
				
		image_path = os.path.join(video_dir,"images",f"\%d.png")
		video_path = os.path.join(video_dir,"video",f"skeleton.mp4")
		palette_path = os.path.join(video_dir,"video",f"skeleton_palette.png")
		frame_rate = sample.fps
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")

	def render_smpl(self,sample,smplRetargetter,video_dir=None):

		target = sample.joints_np
		verts,Jtr,Jtr_offset = smplRetargetter()

		verts = verts.cpu().data.numpy()
		if not hasattr(self,'ps_data'):
			# Initialize Plot SMPL in polyscope
			ps.init()
			self.ps_data = {}
			self.ps_data['bbox'] = verts.max(axis=(0,1)) - verts.min(axis=(0,1))
			self.ps_data['object_position'] = sample.joints_np[0,0]

		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		camera_position = np.array([7*self.ps_data['bbox'][0],0.5*self.ps_data['bbox'][1],0]) + self.ps_data['object_position']
		look_at_position = np.array([0,0,0]) + self.ps_data['object_position']
		ps.look_at(camera_position,look_at_position)

		Jtr = Jtr.cpu().data.numpy() + np.array([0,0,0])*self.ps_data['bbox']

		verts += np.array([0, 0, 0]) * self.ps_data['bbox']
		# target_joints = target - target[:,7:8,:] + Jtr[:,0:1,:] + np.array([0,0,0])*self.ps_data['bbox']
		target_joints = target + np.array([0,0,0])*self.ps_data['bbox']

		Jtr_offset = Jtr_offset[:,smplRetargetter.index['smpl_index']].cpu().data.numpy() + np.array([0.0,0,0])*self.ps_data['bbox']       
		# Jtr_offset = Jtr_offset.cpu().data.numpy() + np.array([0,0,0])*self.ps_data['bbox']       

		ps.remove_all_structures()
		ps_mesh = ps.register_surface_mesh('mesh',verts[0],smplRetargetter.smpl_layer.smpl_data['f'],transparency=0.5)
		

		target_bone_array = np.array([[i,p] for i,p in enumerate(smplRetargetter.index['dataset_parent_array'])])
		ps_target_skeleton = ps.register_curve_network(f"Target Skeleton",target_joints[0],target_bone_array,color=np.array([0,0,1]))

		smpl_bone_array = np.array([[i,p] for i,p in enumerate(smplRetargetter.index['parent_array'])])
		ps_smpl_skeleton = ps.register_curve_network(f"Smpl Skeleton",Jtr[0],smpl_bone_array,color=np.array([1,0,0]))

		smpl_index = list(smplRetargetter.index['smpl_index'].cpu().data.numpy())    

		offset_skeleton_bones = np.array([[x,smpl_index.index(smplRetargetter.index['parent_array'][i])] for x,i in enumerate(smpl_index) if smplRetargetter.index['parent_array'][i] in smpl_index])
		ps_offset_skeleton = ps.register_curve_network(f"Offset Skeleton",Jtr_offset[0],offset_skeleton_bones,color=np.array([1,1,0]))


		dataset_index = list(smplRetargetter.index['dataset_index'].cpu().data.numpy())    		
		joint_mapping = np.concatenate([target_joints[0,dataset_index],Jtr_offset[0]],axis=0)
		joint_mapping_edges = np.array([(i,joint_mapping.shape[0]//2+i) for i in range(joint_mapping.shape[0]//2)])
		ps_joint_mapping = ps.register_curve_network(f"Mapping (target- smpl) joints",joint_mapping,joint_mapping_edges,radius=0.001,color=np.array([0,1,0]))

		if video_dir is None:
			ps.show()
			return 
		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)

		ps.show()
		print(f'Rendering images:')
		for i in tqdm(range(verts.shape[0])):
			ps_mesh.update_vertex_positions(verts[i])
			ps_target_skeleton.update_node_positions(target_joints[i])
			ps_smpl_skeleton.update_node_positions(Jtr[i])
			ps_offset_skeleton.update_node_positions(Jtr_offset[i])
			ps_joint_mapping.update_node_positions(np.concatenate([target_joints[i,dataset_index],Jtr_offset[i]],axis=0))

			image_path = os.path.join(video_dir,"images",f"smpl_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png");
			ps.screenshot(image_path,transparent_bg=False)
			
			# if i > 0.6*verts.shape[0]:
			# if i  % 100 == 99: 
			# 	ps.show()

		image_path = os.path.join(video_dir,"images",f"smpl_\%d.png")
		video_path = os.path.join(video_dir,"video",f"{sample.label}_{sample.recordAttempt}_smpl.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = sample.fps
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")


	def render_smpl_multi_view(self,sample,video_dir=None):

		assert hasattr(sample,'smpl'), "SMPL data not saved. run retarget2smpl.py first"
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
		ps_target_skeleton = ps.register_curve_network(f"Target Skeleton",target_joints[0],target_bone_array,color=np.array([0,0,1]))

		smpl_bone_array = np.array([[i,p] for i,p in enumerate(sample.smpl.index['parent_array'])])
		ps_smpl_skeleton = ps.register_curve_network(f"Smpl Skeleton",Jtr[0],smpl_bone_array,color=np.array([1,0,0]))

		smpl_index = list(sample.smpl.index['smpl_index'].cpu().data.numpy())    

		offset_skeleton_bones = np.array([[x,smpl_index.index(sample.smpl.index['parent_array'][i])] for x,i in enumerate(smpl_index) if sample.smpl.index['parent_array'][i] in smpl_index])
		ps_offset_skeleton = ps.register_curve_network(f"Offset Skeleton",Jtr_offset[0],offset_skeleton_bones,color=np.array([1,1,0]))


		dataset_index = list(sample.smpl.index['dataset_index'].cpu().data.numpy())    		
		joint_mapping = np.concatenate([target_joints[0,dataset_index],Jtr_offset[0]],axis=0)
		joint_mapping_edges = np.array([(i,joint_mapping.shape[0]//2+i) for i in range(joint_mapping.shape[0]//2)])
		ps_joint_mapping = ps.register_curve_network(f"Mapping (target- smpl) joints",joint_mapping,joint_mapping_edges,radius=0.001,color=np.array([0,1,0]))

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


		if video_dir is None:
			ps.show()
			return 
		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)



		# Create random colors of each segment
		colors = np.random.random((sample.segments.shape[0],3))
		mesh_colors = np.zeros((verts.shape[0],3))
		mesh_colors[:,1] = 0.3 # Default color is light blue
		mesh_colors[:,2] = 1 # Default color is light blue
		for i,segment in enumerate(sample.segments):
			mesh_colors[segment[0]:segment[1]] = colors[i:i+1]

		ps.show()	

		print(f'Rendering images:')
		for i in tqdm(range(verts.shape[0])):
			
			ps_mesh.update_vertex_positions(verts[i])
			ps_mesh.set_color(mesh_colors[i])

			ps_target_skeleton.update_node_positions(target_joints[i])
			ps_smpl_skeleton.update_node_positions(Jtr[i])
			ps_offset_skeleton.update_node_positions(Jtr_offset[i])
			ps_joint_mapping.update_node_positions(np.concatenate([target_joints[i,dataset_index],Jtr_offset[i]],axis=0))

			image_path = os.path.join(video_dir,"images",f"smpl_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png")
			ps.screenshot(image_path,transparent_bg=False)
			
			# if i > 0.6*verts.shape[0]:
			# if i  % 100 == 99: 
			# 	ps.show()
			# if i in list(sample.segments.reshape(-1)):
			# 	print(i) 
			# 	ps.show()


		image_path = os.path.join(video_dir,"images",f"smpl_\%d.png")
		video_path = os.path.join(video_dir,"video",f"{sample.label}_{sample.recordAttempt}_smpl.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = sample.fps
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")




# Load file and render skeleton for each video
def render_dataset():
	video_dir = RENDER_DIR
	
	vis = Visualizer()
	
	for subject in os.listdir(INPUT_DIR):
		if '015b7571-9f0b-4db4-a854-68e57640640d' not in subject: 
			continue
		for sample_path in os.listdir(os.path.join(INPUT_DIR,subject,'MarkerData')):
			# Input 
			if "BAPF" in sample_path: 
				continue
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
	sample = OpenCapDataLoader(sample_path)
	
	# Visualize Target skeleton
	# vis.render_skeleton(sample,video_dir=video_dir)


	# Load SMPL
	sample.smpl = SMPLRetarget(sample.joints_np.shape[0],device=None)	
	print(sample.name)
	sample.smpl.load(os.path.join(SMPL_DIR,sample.name+'.pkl'))

	# Visualize SMPL
	# vis.render_smpl(sample,sample.smpl,video_dir=video_dir)
	
	
	# Load Video
	sample.rgb = MultiviewRGB(sample)

	print(f"SubjectID:{sample.rgb.session_data['subjectID']} Action:{sample.label}")

	# Visualize each view  
	# vis.render_smpl_multi_view(sample,video_dir=None)
	

	# Load Segments
	if os.path.exists(os.path.join(SEGMENT_DIR,sample.name+'.npy')):
		sample.segments = np.load(os.path.join(SEGMENT_DIR,sample.name+'.npy'),allow_pickle=True).item()['segments']
	else: 
		return 
	
	if video_dir is not None:
		video_dir = os.path.join(video_dir,f"{sample.openCapID}_{sample.label}_{sample.recordAttempt}")
	vis.render_smpl_multi_view(sample,video_dir=video_dir)





if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		render_dataset()
	else:
		sample_path = sys.argv[1]
		vis = Visualizer()
		video_dir = sys.argv[2] if len(sys.argv) > 2 else None
		render_smpl(sample_path,vis,video_dir)

