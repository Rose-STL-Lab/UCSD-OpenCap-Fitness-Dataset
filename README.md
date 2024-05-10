# Sports Analytics Dataset 
 
[![](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue)](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/)
[![](https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green)](https://www.overleaf.com/project/655aba246db8455baf77edd5)
[![](https://img.shields.io/badge/Code-Github-red?style=flat&logo=github)](https://github.com/shubhMaheshwari/UCSD-Fitness-Dataset)
[![](https://img.shields.io/badge/Dataset-Videos-pink)]()


The repository contains information about the MCS dataset, which is used for sports analytics. It provides tools to convert motion capture data from .trc format to SMPL format. Additionally, it includes rendering capabilities using polyscope and data analysis functionalities.



```
- Interpolate between MCS scores 
- Each activity is divided 5 classes 
```


## Tasks
- Remove SMPLLoader
- Download videos using opencap batchDownload.ipynb: `https://github.com/stanfordnmbl/opencap-processing/blob/main/batchDownload.ipynb1`
- Create a single logger for all tasks. And seperate SummaryWriter for plotting metrics. Split get_logger function 
- Add docs using MKLdocsString 
- Create best visualization portal. 
- Integrate MOT file from https://github.com/davidpagnon/Pose2Sim_Blender/
- Change convert (raises error when os.system is called. )
- Main installation script to install convert and other libraries
- Change delimiter for OpenCAP loader. make it compatible for linux and windows.
## Dataset 

To download the dataset use the following links: 
1. [OpenCap Master](https://docs.google.com/spreadsheets/d/1vkZ4-cdH2RjEOTZWhoYnSdXn8ruz9VFXZW7tg9fRYPE/edit?usp=sharing)
2. [OpenCap]

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


## 1. Visualization 

### Installation
```
pip install polyscope opensim ffmpeg
```

- Raise an issue if you are having trouble installing any of the above packages


<details>
<summary>Polyscope installation details </summary>
- Linux

```
```
</details>


<details>
<summary>OpenSim installation details </summary>
 Step:1 - https://github.com/opensim-org/opensim-core/wiki/Build-Instructions#configuration-1
 Step 2 - https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python
</details>



###  Rendering 
Renders a video of the skeleton video using polyscope 
```
python3 renderer.py # For complete dataset
```
Or 
```
python3 renderer.py <sample-filepath> # Specific trc file
```

`<sample-filepath>` is the path to the trc file containing the xyz co-ordinates of each joint to plot


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


<details>
<summary>OpenSim installation details </summary>
 Step:1 - https://github.com/opensim-org/opensim-core/wiki/Build-Instructions#configuration-1
 Step 2 - https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python
</details>


```
python src/temporal_segmentation.py 
```


## 3. Dataset Aggregration 
Store the retargeted smpl data into a single .pkl file for analysis and training.



### 3. Data engineering 
- Input representation 


## Data analysis
- Mocap Capture 
    - [Click to download multi-view RGB Videos and .mot](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/zweatherford_ucsd_edu/EuHlQ1oahHBGgRTADJoImk8BclFRfX5VLFcI0_CbKiZ9Tg?e=q4lBjq)  



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