# Sports Analytics Dataset 
    - Interpolate between MCS scores 
    - Each activity is divided 5 classes 


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
python3 retarget.py # For complete dataset
```
Or 
```
python3 renderer.py <sample-filepath> # Specific trc file
```

`<sample-filepath>` is the path to the trc file containing the xyz co-ordinates of each joint to plot



### 3. Data engineering 
- Input representation 


## Data analysis
- Mocap Capture 
### TODO: Per sample pose reconstruction using PCA 

### TODO: Per sample per joint fourier analysis 


### Muscle forces reaction data analysis
We are using the following model to evaluate our generartion: 
```
    Rajagopal, A., Dembia, C.L., DeMers, M.S., Delp, D.D., Hicks, J.L., Delp, S.L. (2016) Full-body musculoskeletal model for muscle-driven simulation of human gait. IEEE Transactions on Biomedical Engineering
```

Additional information about the model can be found on the links below: 
### Muscular-Skeleton Model information: 
- https://simtk-confluence.stanford.edu:8443/display/OpenSim/OpenSim+Models
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5989715/
- https://github.com/opensim-org/opensim-models





