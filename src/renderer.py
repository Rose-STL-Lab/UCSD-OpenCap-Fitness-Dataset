import os 
import sys
import numpy as np 
from tqdm import tqdm
import polyscope as ps 
import trimesh 

from utils import * 
from dataloader import OpenCapDataLoader

class Visualizer: 
	def __init__(self): 
		
		ps.init()

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		# ps.set_view_projection_mode("orthographic")
		ps.set_ground_plane_mode('shadow_only')

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

		video_dir = os.path.join(video_dir,f"{sample.openCapID}_{sample.label}_{sample.mcs}")
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

		# ps.show()
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
		video_path = os.path.join(video_dir,"video",f"{sample.label}_{sample.mcs}_smpl.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = sample.fps
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")


	def render_rabit(self,sample,video_dir=None): 

		T = sample.num_frames

		# Get bounding box and object position 
		smpl_verts,smpl_joints,_ = sample.smpl()
		smpl_verts = smpl_verts.cpu().data.numpy()
		smpl_joints = smpl_joints.cpu().data.numpy()

		# Load 0th frame and get params
		sample.rabit.load_smpl_params(sample.smpl.smpl_params,0)
		rabit_verts = sample.rabit.verts
		rabit_joints = sample.rabit.J


		bbox_smpl = smpl_verts.max(axis=(0,1))  - smpl_verts.min(axis=(0,1))
		bbox_target = rabit_verts.max(axis=0)  - rabit_verts.min(axis=0)

		bbox = bbox_smpl if np.linalg.norm(bbox_smpl) > np.linalg.norm(bbox_target) else bbox_target

		object_position = sample.joints_np[0,0]

		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		camera_position = np.array([7*bbox[0],0,0]) + object_position
		look_at_position = np.array([0,0,0]) + object_position
		ps.look_at(camera_position,look_at_position)

		# Translate objects to visualize 
		smpl_joints += (np.array([0,0,+0.5])*bbox).reshape((1,-1,3))  
		smpl_verts += (np.array([0, 0, +0.5]) * bbox).reshape((1,-1,3))

		rabit_joints += (np.array([0,0,-0.5])*bbox).reshape((1,3))  
		rabit_verts += (np.array([0, 0, -0.5]) * bbox).reshape((1,3))


		# Initial plot
		ps.remove_all_structures()
		ps_smpl_mesh = ps.register_surface_mesh('SMPL Mesh',smpl_verts[0],sample.smpl.smpl_layer.smpl_data['f'],transparency=0.5)
		
		smpl_bone_array = np.array([[i,p] for i,p in enumerate(sample.smpl.index['parent_array'])])
		ps_smpl_skeleton = ps.register_curve_network(f"SMPL Skeleton",smpl_joints[0],smpl_bone_array,color=np.array([1,0,0]))


		ps_rabit_mesh = ps.register_surface_mesh('RaBit Mesh',rabit_verts,sample.rabit._faces,transparency=0.5)
		
		rabit_bone_array = np.array([[i,p] for i,p in enumerate(sample.smpl.index['parent_array'])])
		ps_rabit_skeleton = ps.register_curve_network(f"RaBit Skeleton",rabit_joints,rabit_bone_array,color=np.array([1,0,0]))


		if video_dir is None:
			ps.show()
			return 
		os.makedirs(video_dir,exist_ok=True)
		os.makedirs(os.path.join(video_dir,"images"),exist_ok=True)
		os.makedirs(os.path.join(video_dir,"video"),exist_ok=True)

		# ps.show()
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
		video_path = os.path.join(video_dir,"video",f"{sample.label}_{sample.mcs}_smpl.mp4")
		palette_path = os.path.join(video_dir,"video",f"smpl.png")
		frame_rate = sample.fps
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	 {video_path}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")








class VisualizerTrimesh: 
	def __init__(self): 
		
		self.scene = trimesh.Scene()

	def render_with_texture(pose_data):
		# Load Random RaBit Model and render 
		window_conf = gl.Config(double_buffer=True, depth_size=6)
			

		rabit = RaBitModel()
		texture_util.generate_texture("output/m_t.png") # Create a random texture

		beta = np.random.rand(*(500,)) * 10 - 5
		# beta[10:] = smpl.smpl_params['shape']
		# beta[10:] = 0
		
		trans = np.zeros(rabit.trans_shape)


		vis.render_with_texture() # Only possible using trimesh for now 

		os.makedirs("output/renderings", exist_ok=True)
		for i,theta in enumerate(pose_data): 
			if i % 50 == 0: 
				print(f"Rendering:{i} frame")
			rabit.set_params(beta=beta, pose=theta, trans=trans)
			rabit.save_to_obj_with_texture(save_path)
			
			tmesh = trimesh.load(save_path)

			scene.add_geometry(tmesh,"RaBit") 


			# increment the file name
			file_name = "output/renderings/render_" + str(i) + ".png"
			# save a render of the object as a png
			png = scene.save_image(resolution=[1920, 1080], visible=True)
			with open(file_name, "wb") as f:
				f.write(png)
				f.close()

			scene.delete_geometry(['m_t.obj'])


		os.system(f"ffmpeg -y -framerate 60 -i output/renderings/render_\%d.png output/video.mp4")


# Load file and render skeleton for each video
def render_dataset():
	video_dir = 'rendered_videos'
	
	vis = Visualizer()
	
	for subject in os.listdir(INPUT_DIR):
		for sample_path in os.listdir(os.path.join(INPUT_DIR,subject,'MarkerData')):
			sample_path = os.path.join(INPUT_DIR,subject,'MarkerData',sample_path)
			sample = OpenCapDataLoader(sample_path)
			vis.render_skeleton(sample,video_dir=video_dir)
		
	



if __name__ == "__main__": 

	if len(sys.argv) == 1: 
		render_dataset()
	else:
		sample_path = sys.argv[1]
		sample = OpenCapDataLoader(sample_path)

		vis = Visualizer()
		video_dir = sys.argv[2] if len(sys.argv) > 2 else None
		vis.render_skeleton(sample,video_dir=video_dir)		 
