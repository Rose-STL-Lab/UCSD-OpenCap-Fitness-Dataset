import os
import sys
import json
import subprocess
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--command", type=str, help="Command to run (create, run, delete)")
argparser.add_argument("--subjects", type=str, required=True, help="Directory containing subject files")

args = argparser.parse_args()


# Directory containing subject files
subject_dir = args.subjects
assert os.path.exists(subject_dir), f"Directory {subject_dir} does not exist."

# Directory to store YAML files
yaml_dir = "digital-coach-yaml"
os.makedirs(yaml_dir, exist_ok=True)

# Namespace for the Pods
namespace = "spatiotemporal-decision-making"


# Template for the Pod YAML
pod_template = """
apiVersion: v1
kind: Pod
metadata:
  namespace: {namespace}
  name: digital-coach-{subject_id}
spec:
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 1
          preference:
            matchExpressions:
            - key: nautilus.io/group
              operator: In
              values:
              - ry
  tolerations:
  - key: "nautilus.io/ry-reservation"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  imagePullSecrets:
    - name: shubh-gitlab-registry
  containers:
  - name: shubh-container
    image: gitlab-registry.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh:latest
    imagePullPolicy: Always
    command: ["/bin/bash", "-c"]
    args: 
    - | 
      conda init bash 
      source ~/.bashrc
      source /root/.bashrc

      conda activate T2M-GPT
      # conda run -n T2M-GPT pip install tqdm

      cd /UCSD-OpenCap-Fitness-Dataset
      git -c http.sslVerify=false  pull origin main
      
      
      # Link the data folder to the mounted data folder
      cd /T2M-GPT
      git -c http.sslVerify=false  pull origin main

      ln -sf /mnt/data/MCS_DATA/Data Data

      export PYTHONUNBUFFERED=1

      conda run -n T2M-GPT python demo/nimble-gui.py visualize --geometry-folder /mnt/data/MCS_DATA/OpenCap_LaiArnoldModified2017_Geometry 
      sleep 1000  

    env: 
    - name: SUBJECT_ID
      value: "{subject_id}"

    volumeMounts:
    - mountPath: /dev/shm
      name: dshm
    - mountPath: /mnt/data
      name: sports-analytics-database
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: "30Gi"
        cpu: "8"
      requests:
        nvidia.com/gpu: "1"
        memory: "20Gi"
        cpu: "4"
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
  - name: sports-analytics-database
    persistentVolumeClaim:
      claimName: sports-analytics-database
  restartPolicy: Never
"""

def create_pod_yaml(subject_id):
    yaml_content = pod_template.format(namespace=namespace, subject_id=subject_id)
    yaml_file = os.path.join(yaml_dir, f"digital_coach_{subject_id}.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    return yaml_file

def is_pod_running(pod_name):
    result = subprocess.run(["kubectl", "get", "pods", pod_name, "-n", namespace, "-o", "json"], capture_output=True, text=True)
    if result.returncode == 0:
        pod_info = json.loads(result.stdout)
        if pod_info["status"]["phase"] in ["Pending", "Running"]:
            return True
    return False

def create_yamls():
    for subject_file in os.listdir(subject_dir):
        subject_id = os.path.splitext(subject_file)[0]
        create_pod_yaml(subject_id)

def run_pods():
    create_yamls()
    cnt = 0
    for yaml_file in os.listdir(yaml_dir):
        yaml_path = os.path.join(yaml_dir, yaml_file)
        pod_name = os.path.splitext(yaml_file)[0].replace("digital_coach_", "digital-coach-")
        if not is_pod_running(pod_name):
            subprocess.run(["kubectl", "apply", "-f", yaml_path])
        else:
            print(f"Pod {pod_name} is already running. Skipping...")


def delete_pods():
    for subject_file in os.listdir(subject_dir):
        subject_id = os.path.splitext(subject_file)[0]
        pod_name = f"digital-coach-{subject_id}"
        subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace])


def is_pod_completed(pod_name):
    result = subprocess.run(["kubectl", "get", "pods", pod_name, "-n", namespace, "-o", "json"], capture_output=True, text=True)
    if result.returncode == 0:
        pod_info = json.loads(result.stdout)
        if pod_info["status"]["phase"] in ["Succeeded", "Failed"]:
            return True
    return False

def delete_completed_pods():
    for subject_file in os.listdir(subject_dir):
        subject_id = os.path.splitext(subject_file)[0]
        pod_name = f"digital-coach-{subject_id}"
        if is_pod_completed(pod_name):
            subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace])
            print(f"Pod {pod_name} deleted as it is completed.")
        else:
            print(f"Pod {pod_name} is not completed. Skipping...")



def is_pod_oomkilled(pod_name):
    result = subprocess.run(["kubectl", "get", "pods", pod_name, "-n", namespace, "-o", "json"], capture_output=True, text=True)
    if result.returncode == 0:
        pod_info = json.loads(result.stdout)
        for container_status in pod_info["status"].get("containerStatuses", []):
            if container_status["state"].get("terminated", {}).get("reason") == "OOMKilled":
                return True
            if container_status["state"].get("terminated", {}).get("reason") == "Error":
                return True
            
    return False

def delete_oom_pods():
    for subject_file in os.listdir(subject_dir):
        subject_id = os.path.splitext(subject_file)[0]
        pod_name = f"digital-coach-{subject_id}"
        if is_pod_oomkilled(pod_name):
            subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace])
            print(f"Pod {pod_name} deleted due to OOMKilled status.")
        else:
            print(f"Pod {pod_name} is not OOMKilled. Skipping...")



command = args.command
if command == "create":
    create_yamls()
elif command == "run":
    run_pods()
elif command == "delete":
    delete_completed_pods()
elif command == "oom-delete":
    delete_oom_pods()
else:
    print("Invalid command. Use 'create', 'run', or 'delete'.")
    sys.exit(1)