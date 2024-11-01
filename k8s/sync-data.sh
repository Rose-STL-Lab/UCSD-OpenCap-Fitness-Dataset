#!/bin/bash

# Usage instructions
# This script deploys Syncthing to a Kubernetes cluster to synchronize data between a local machine and a Kubernetes Pod.
# It requires the DEVICE_ID environment variable to be set to the Syncthing device ID of the local machine.
#
# Usage:
#   export DEVICE_ID="your-device-id"
#   ./sync-data.sh

# Check if DEVICE_ID is set
if [ -z "$DEVICE_ID" ]; then
  echo "Error: DEVICE_ID environment variable is not set. Used by Syncthing to identify the master machine."
  echo "Set: export DEVICE_ID=\"your-device-id\""
  exit 1
fi

# Delete the existing Pod if it exists
kubectl delete pod data-transfer-pod --ignore-not-found

# Apply the ConfigMap
kubectl apply -f sync-data/syncthing-config.yaml

# Apply the Pod
kubectl apply -f sync-data/syncthing-setup.yaml

# Wait for the Pod to be running
echo "Waiting for the Pod to be running..."
sleep 20

# Check Pod status
POD_STATUS=$(kubectl get pod data-transfer-pod -o jsonpath='{.status.phase}')
if [ "$POD_STATUS" != "Running" ]; then
  echo "Error: Pod is not running. Current status=$POD_STATUS"
  echo "Describing the Pod..."
  kubectl describe pod data-transfer-pod
  echo "Fetching Pod logs..."
  kubectl logs data-transfer-pod
  exit 1
fi

echo "Syncthing deployment complete."
echo "To access the Syncthing web GUI, run the following command:"
echo "kubectl port-forward pod/data-transfer-pod 32000:8384"
echo "Then open your browser and go to http://localhost:32000"