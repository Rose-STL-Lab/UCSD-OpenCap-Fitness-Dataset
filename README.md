# UCSD OpenCap Fitness Dataset 
 
[![](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue)](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/)
[![](https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green)](https://www.overleaf.com/project/655aba246db8455baf77edd5)
[![](https://img.shields.io/badge/Code-Github-red?style=flat&logo=github)](https://github.com/shubhMaheshwari/UCSD-Fitness-Dataset)
[![](https://img.shields.io/badge/Dataset-Videos-pink)]()

![](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white)
![](https://img.shields.io/badge/Ubuntu20.04-E95420?style=flat&logo=ubuntu&logoColor=white)

The repository contains information about the MCS dataset, which is used for sports analytics. It provides tools to convert motion capture data from .trc format to SMPL format. Additionally, it includes rendering capabilities and  analysis functionalities.


## Tasks

- Remove SMPLLoader
- Add docs using MKLdocsString 
- Reaplace `convert` command to merge images (raises error when os.system is called. )
- Main installation script to install convert and other libraries
- Change delimiter for OpenCAP loader. make it compatible for linux and windows.



## Dataset Download links

To download the dataset use the following links: 
<!-- 1. [OpenCap Master](https://docs.google.com/spreadsheets/d/1vkZ4-cdH2RjEOTZWhoYnSdXn8ruz9VFXZW7tg9fRYPE/edit?usp=sharing) -->
<!-- 2. [OpenCap]  -->
<!-- 3. [PPE Squat](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/zweatherford_ucsd_edu/EuHlQ1oahHBGgRTADJoImk8BclFRfX5VLFcI0_CbKiZ9Tg?e=q4lBjq)   -->


## Steps to re-create the dataset
```
python3 src/retarget2smpl.py -f 
python3 src/temporal_segmentation.py
python3 src/generate_data_pkl.py
```


### OpenCap File Structure
```
.
├── full_data
│ ├── OpenCapData_015b7571-9f0b-4db4-a854-68e57640640d
│ │ ├── CalibrationImages
│ │ │ ├── calib_imgCam{%d}.jpg
│ │ ├── MarkerData
│ │ │ ├── BAP{%1d}.trc
│ │ ├── OpenSimData
│ │ │ ├── Kinematics
│ │ │ │ ├── {%class}{%d}.mot
│ │ │ └── Model
│ │ │     ├── Geometry
│ │ │     └── LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim 
│ │ ├── README.txt
│ │ ├── sessionMetadata.yaml
│ │ └── Videos
│ │     ├── Cam%d
│ │     │ └── cameraIntrinsicsExtrinsics.pickle
│ │     └── mappingCamDevice.pickle
```
**Opencap processing**:  [Library to process files](https://github.com/stanfordnmbl/opencap-processing )

## Code Structure
```
src
├── analyze_dtw_score.py // Script for analyzing Dynamic Time Warping (DTW) scores
├── dataloader.py // Contains code for loading and preprocessing data
├── evaluation
│   └── foot_sliding_checker.py // Script for checking foot sliding in animations
├── generate_data_pkl.py // Script for generating pickle files from data
├── HumanML3D // Convert dataset to HumanML3D Format
│   ├── rots_to_smpl.py // Script for converting rotations to SMPL format
├── meters.py // Contains code for measuring and logging training progress
├── pose_reconstruction.py // Contains code for reconstructing poses from data
├── renderer.py // Contains code for rendering makerker data(.trc), smpl mesh and skeleton, camera location
├── retarget2smpl.py // Script for retargeting .trc to the SMPL model
├── smpl_loader.py // Contains code for loading SMPL models
├── temporal_segmentation.py // Detects start stop cycle for each sample
├── tests.py // Contains unit tests for the project
└── utils.py // Contains constants and logging functions used across the project
```

Note:- Each python file can be called from any directory. 
```
// All commands that should work only any system
python retarget2smpl.py 
python src/retarget2smpl.py
python UCSD-OpenCap-Fitness-Dataset/src/retarget2smpl.py  
python src/opencap_reconstruction_render.py <subject-path>  <mot-path>  <save-path>
```


## 0. General Setup 

- Clone repo
    ```
        git clone --recursive https://github.com/Rose-STL-Lab/UCSD-OpenCap-Fitness-Dataset.git
        cd UCSD-OpenCap-Fitness-Dataset
    ```

- Creating environment

    ```
    conda create --name bitte -f environment.yml

    ```

- Pip packages
    ```
    pip install polyscope opensim ffmpeg glfw 

    ```

- Raise an issue if you are having trouble installing any of the above packages

## 1. Visualization 




###  Rendering 
Renders a video of the skeleton video using polyscope 
```
python3 python src/opencap_reconstruction_render.py # For complete dataset
```
Or 
```
python3 python src/opencap_reconstruction_render.py <subject-path>  <mot-path>  <save-path> # Specific trc file
```


<details>
<summary> Running on a Remote Server / North Servr / Linux Containers ? </summary>

Polyscope has a lot of trouble installing on the remote server. Below are a few steps that can be taken for fix common errors. 

1. Use `ssh -X` to login 
2. Set `export DISPLAY=:99.0`
3. Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

3. Fix for error: 

    ```
        libGL error: MESA-LOADER: failed to open swrast: /home/ubuntu/.conda/envs/T2M-GPT/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /usr/lib/dri/swrast_dri.so) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
        libGL error: failed to load driver: swrast
        GLFW emitted error: GLX: Failed to create context: GLXBadFBConfig
    ```

    ```
        rm /home/ubuntu/.conda/envs/T2M-GPT/bin/../lib/libstdc++.so.6
        ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6  /home/ubuntu/.conda/envs/T2M-GPT/bin/../lib/libstdc++.so.6 
    ```

4. Fix for `GLFW emitted error: The GLFW library is not initialized`
    ```
        pip install glfw
    ```

5. Fix for error `DISPLAY not found`: 
    ```
        export DISPLAY=:99.0
    ```

6. Segfault: 
    ```
        export DISPLAY=:99.0
    ```

7. Fix for `GLFW emitted error: X11: Failed to open display :99.0`
    ```
        Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    ```



</details>

## 2. Retargetting  
To retarget .trc file to SMPL format  
```
python3 src/retarget2smpl.py # For complete dataset
```
Or 
```
python3 src/retarget2smpl.py  -f --file <sample-filepath> # Specific trc file 
```


`<sample-filepath>` is the path to the trc file containing the xyz co-ordinates of each joint to plot

`-f` forces a re-run on retargetting even if pkl file containg smpl data is already present.  

[Click to download extracted SMPL data from TRC file](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/shmaheshwari_ucsd_edu/EQ41wb0to2pHsLFhXmdTT2sB4jutOKR37ZLo7m6zv_X3hw) 


## 3. Temporal Segmentation 

### Installation
```
pip install 
```

- Raise an issue if you are having trouble installing any of the above packages


<details>
<summary>Convert Installation details </summary>
- Linux

```
sudo apt install imagemagick

```
</details>





```
python src/temporal_segmentation.py 
```


## 3. Motion Aggregration

- To train the generative model we use the 263 representation proposed by Guo et al. () 

### MDM Format
    Store the retargeted smpl data into a single .pkl file for analysis and training.
    ```
        python src/generate_data_pkl.py
    ```

### HumanML3D format (263 dim representation):

    The data from pkl file is converted into 263 HumanML3D format for generation and classifier purposes. 
    ```
        python src/HumanML3D/rots_to_smpl.py
    ```

    Each sample consists of
    - root_rot_velocity $ \in R^{seq\_len \times 1}$
    - root_linear_velocity $\in R^{seq\_len \times 2}$
    - root_y $\in R^{seq\_len \times 1}$
    - ric_data $ \in R^{seq\_len \times 3(joint\_num - 1)}$
    - rot_data $\in R^{seq\_len \times 6(joint\_num - 1)} $
    - local_velocity $\in R^{seq\_len \times 3joint\_num} $
    - foot contact $\in R^{seq\_len \times 4} $

    Here: 1 + 2 + 1 + 21\*3 + 21\*6 + 22\*3 + 4 = 263  
    $seq\_len$ is the number of frame

    $joint\_num=22$ is the number of joints used my HumanML3D SMPL representation. The last 2 joints (left and right hand) are discarded.    


### 4. Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1c-MYPiTuyefkh-6mNw8y9M1BYVCnnR?usp=sharing)

#### To run locally 

<details>
<summary>OpenCap installation details </summary>

Multiple modules need to be installed  


- OpenSim

    ```
    conda install -c opensim-org opensim
    ```

- OpenCap-processing
    
    ```
    git clone https://github.com/stanfordnmbl/opencap-processing.git
    ```

- CasADi

    ```
    cd $UCSD_OPENCAP_DATASET_DIR/deps
    git clone --recursive https://github.com/casadi/casadi.git casadi
    cd casadi
    git checkout 3.5.5
    mkdir -p build
    cp build
    
    apt install swig
    cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON ..
    make
    make install
    cd $UCSD_OPENCAP_DATASET_DIR/deps/opencap-processing
    ```

- References:

    - Step:1 - https://github.com/opensim-org/opensim-core/wiki/Build-Instructions#configuration-1

    - Step 2 - https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python


</details>





#### TODO: Per sample pose reconstruction using PCA 

#### TODO: Per sample per joint fourier analysis 


#### Muscle forces reaction data analysis
```
    Rajagopal, A., Dembia, C.L., DeMers, M.S., Delp, D.D., Hicks, J.L., Delp, S.L. (2016) Full-body musculoskeletal model for muscle-driven simulation of human gait. IEEE Transactions on Biomedical Engineering
```

Additional information about the model can be found on the links below: 
#### Muscular-Skeleton Model information: 
- https://simtk-confluence.stanford.edu:8443/display/OpenSim/OpenSim+Models
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5989715/
- https://github.com/opensim-org/opensim-models

Relevant papers: 

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462


## Developer Utils

### Syncthing 

- To sync database across different systems. Syncthing uses a DEVICE-ID (created during installation) to transfer files between servers. 

- Only drawback is that it requires GUI to access both your system and server. 

- Installation instructions: [here](https://docs.syncthing.net/intro/getting-started.html)



#### Commands for the remote server

```
syncthing
```

port-forwarding to access gui of the remote server

```
ssh -L 9000:127.0.0.1:8384 ubuntu@north.ucsd.edu -i ~/Desktop/panini
```


To sync a folder. First `Add folder` on the and add the name and path to the folder to sync 
Then 
1. open the folder box 
2. click `Edit`
3. Go to tab `Sharing`
4. Select server to sync with 




-------------------------------------- 
Kubernetes syncthing setup manual steps: 
1. open syncthing everywhere 
2. add devices everywhere 
3. folder -> edit everywhere 
    