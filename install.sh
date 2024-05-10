
# ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
#     detected_OS := Windows
# else
#     detected_OS := $(shell uname)  # same as "uname -s"
# endif


# Changes to install convert 



# Python packages
pip install polyscope ffmpeg trimesh pyyaml easydict chumpy numpy==1.23.5