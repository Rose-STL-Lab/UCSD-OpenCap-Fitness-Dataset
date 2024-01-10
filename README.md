# Sports Analytics Dataset 
 

[![](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue)](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/)
[![](https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green)](https://www.overleaf.com/project/655aba246db8455baf77edd5)
[![](https://img.shields.io/badge/Code-Github-red?style=flat&logo=github)](https://github.com/shubhMaheshwari/UCSD-Fitness-Dataset)
[![](https://img.shields.io/badge/Dataset-Videos-pink)]()


```
- Interpolate between MCS scores 
- Each activity is divided 5 classes 
```

## OpenCap File Structure
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





## 4. Retargetting to RaBit

<details>
<summary>Rabit installation details </summary>
1. Clone RaBit Library

```
    git clone https://github.com/zhongjinluo/RaBit.git
    cd RaBit 
```
2. Download model data from [link](https://drive.google.com/file/d/1yvweTYPKtmuMt5Eu7CHZ4-Do4CRYLFtp/view?usp=sharing) to `<HOME_PATH>/RaBit`

3. Unzip 
```
unzip rabit_data.zip
```
</details>

