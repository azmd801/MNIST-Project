In this project, a comprehensive machine learning system is developed using Python to predict handwritten digits using the MNIST dataset.

The code is designed with modularity in mind, ensuring easy maintainability and packaging.

TensorFlow's Convolutional Neural Network (CNN) architecture is utilized to build the predictive model.

MLflow library is employed for several purposes:

Packaging the trained model using one of its flavors, specifically TensorFlow.
Creating prediction endpoints as a RESTful API.
Defining the input and output schema of the model.
A Dockerfile is written to containerize the packaged prediction endpoint.

Subsequently, the endpoint is dockerized using the Dockerfile and pushed to Docker Hub.

The dockerized container is then deployed as a container inside a pod of a local Kubernetes cluster, such as Minikube.



# Perform the following steps to run this project
pull the repo in your local system
## Setting up environment
### Creating environment
``virtualenv venv``
### Activate environment
``source venv/Scripts/activate``
### Install dependencies
```pip install -r requirements.txt```

### Run the project 
`python run_ml_workflow.py`

will fetch the data from source, tranform it, train a model on it , perform evalution of the model and register the model with appropriate version

##### Set environment variable for the tracking URL where the Model Registry resides
`export MLFLOW_TRACKING_URI=http://localhost:5000`

### Local  deplyment for testing purposes   
```mlflow models serve -m "models:/mnist_classifier/1" -p 8000```

### Reequest for verrsion
```curl http://127.0.0.1:8000//version```


### request to generate docker file

```mlflow models generate-dockerfile -m "models:/mnist_classifier/1" -d inferencing_api_deployment```

### Build the docker image
```docker build -t minist/inferencing:prod "inferencing_api_deployment"```

### Pushing the image to docker hub
docker tag 0a8eab4c8858 azmd801/minist-classifier:dev
docker push azmd801/minist-classifier:tagname

## Pushing the image to ECR

### tag the local image using
```docker tag 0a8eab4c8858 637423362191.dkr.ecr.us-east-1.amazonaws.com/minst_classifier:dev```



## Check kubectl help
kube --help

### install hyperhit and minikube on windows machine

`choco install minikube`

`kubectl`

`minikube`

### create minikube cluster
`minikube start --vm-driver=hyperkit`

`kubectl get nodes`

`minikube status`

`kubectl version`
`

### Run following command for depplying the app

`kubectl apply -f "Kubernetes_deployment_yaml\minist-classifier_inference_endpoint.yaml"`


## Steps for deployment id AWS EKS
###  authenticate Docker to an Amazon ECR registry
 `aws ecr get-login-password --region us-east-1`

 `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 637423362191.dkr.ecr.us-east-1.amazonaws.com`

## Configuring AWS EKS for kubernetes deployment
 ### Installing chocklaty for installing eksctl

### Installing eksctl
`choco install -y eksctl`


