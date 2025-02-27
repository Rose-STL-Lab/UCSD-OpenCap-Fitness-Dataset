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



# Data 
- Start syncthing 
    ```
    export DEVICE_ID="<DEVICE-ID>"
    ./sync-data.sh
    ```
- Sync data with other servers
    ```
    

    ```