# Nautlilus Access 
    These instructions are for UCSD collaborators trying to run the repo using kuberneters



# Load balancer 
- Start load balancer
    ```
    cd bige-demo-website ; 
    kubectl apply -f ingress-loadbalancer.yaml; 
    ```

- Get details
    ```
    kubectl get ingress; 
    kubectl describe ingress bige25f-ingress
    ```



# Data Sharing


- Start syncthing 
    ```
    export DEVICE_ID="<DEVICE-ID>"
    ./sync-data.sh
    ```
- To run syncthing on the persistant volume, run port-forwaring to login 
    ```
    kubectl port-forward pod/data-transfer-pod 32000:8384
    ```

- Other devices
    -------------------------------------- 
    Kubernetes syncthing setup manual steps: 
    1. open syncthing everywhere 
    2. add devices everywhere 
    3. folder -> edit everywhere 


## Run BIGE Server

- YAML file

- Run YAML file on a pod
    ```
    kubectl apply -f digital-coach-yaml/digital_coach_d66330dc-7884-4915-9dbb-0520932294c4.yaml
    ```