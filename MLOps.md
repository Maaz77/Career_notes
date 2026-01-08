
# List of technoloogies of Concepts to learn:

    - MLFlow
    - AirFlow
    - Evidently
    - Grafana
    - pytest
    - CI/CD
    - pre-commit hooks
    - Github Actions
    - AWS services for MLOps
    - terraform
    - Kubernates
    - Docker Compose
    - localstack
    - MakeFile

# postgres connection string example

```
postgresql://mlflow:mlflow@postgres:5432/mlflow
│          │      │       │      │     │
│          │      │       │      │     └─── Database name
│          │      │       │      └───────── Port
│          │      │       └──────────────── Host (service name, container name)
│          │      └──────────────────────── Password
│          └─────────────────────────────── Username
└────────────────────────────────────────── Protocol
```

**Component Breakdown:**

| Component | Value | Description |
|-----------|-------|-------------|
| **Protocol** | `postgresql://` | Database connection protocol |
| **Username** | `mlflow` | Database user credentials |
| **Password** | `mlflow` | Database password |
| **Host** | `postgres` | Service name or container name |
| **Port** | `5432` | PostgreSQL default port |
| **Database** | `mlflow` | Target database name |
    


# CI/CD for MLOps

## The Benefit of GitHub Actions (The "Safety Net")

If you can push broken code, why use GitHub Actions? In a professional MLOps setting, it serves three critical functions:

The "Gold Standard" of Truth: Your local environment might have a specific configuration or a "dirty" state that makes tests pass. GitHub Actions provides a clean, isolated environment (the "Authority") to prove the code works universally.

Branch Protection (The Real Gate): In a professional project, you don't push directly to master. You push to a feature branch and open a Pull Request (PR).

GitHub will show the Red X on the PR.

You can set a Branch Protection Rule that literally "Locks" the Merge button until the Red X turns into a Green Check.

Documentation of Failure: The Red X tells you exactly what broke (Linting? Tests? Infrastructure?) and preserves a log of the failure. This is vital for debugging in a team environment.


## Notes

Real-World Context: Two-Repos Approach
In professional teams, infrastructure and application code are usually separated into:
Infrastructure repo: stable components like VPCs, ECR, databases, etc.
Service/App repo: application-level infrastructure like Lambda or ECS tasks
CI/CD pipelines then ensure that infrastructure is deployed before application services rely on them. For example:
🔁 “If ECR has been provisioned by infra CI, then app CI can deploy Lambda using the image URI.”
But for this mono-repository workshop, we're doing everything together — so we must simulate those dependencies inside Terraform.




# Kubernates  

- K8S = Kubernates 
- automated deployement across different servers
- distribute load across servers (when to scale up and down)
- health check of containers - replace failed containers


- other containers: CRI-o, containerd

## POD 
- Smallest unit, inside it there are shared vol, containers, shared IP address
- single container per pod most common
- one pod -> one server
- created automatically by kubernates
- all containers inside the pod share same namespace ( volume, ip address of that pod)
- could be deleted at anytime

## Kubernates Cluster
- **Consist of Nodes** 
    - Nodes are physical or virtual servers
    - different datacenters
    - **inside nodes there Pods**
    - inside pods there containers

    ### Master Node & Worker Node
    - Master Node Role play
    - Worker Node contains our application
    - communicate with api server
    - Master Node and Worker Node Contains services:
        - Kubelet
        - Kube-proxy
        - container Runtime
    
    - Master Node Contains services:
        - scheduler
        - kuber Controller manager
        - cloude controller manager -> provide load balancer
        - etcd

- ****Kubectl**** -> cli to manage, connect to api server on master node from local 

- Cluster = infrastructure boundary
- Namespace = logical boundary, A namespace is a virtual cluster inside a physical Kubernetes cluster. It provides isolation and organization.

- Node = compute boundary
- Pod = scheduling boundary - **different pods on different nodes could be inside same namespace**
- Container = execution boundary

```
Deployment
    ↓ creates / updates
ReplicaSet
    ↓ creates / deletes
Pod
    ↓ scheduled onto
Node
```

- A **Deployment** manages Pods (how your app runs).
- A **Service** exposes Pods (how your app is reached).
- Pods Can die, Can be recreated, Can change IP addresses
- Without a Service: Clients would break every time a Pod restarts
- Without a Deployment:You’d have no automation, scaling, or self-healing


## Kubernates Practical Imperative

- create cluster localy by ****"minikube"**** -> its better to use virtualbox rather than docker for local 
- "kubectl" cli -> manage cluster outside the nodes 
- >minikube start --driver=='docker or vm'
- >minikube status
- >minikube ip 
    - (get the k8s node ip)
- >the ssh the docker@ip 
    - with the password "tcuser"
- > docker ps 
    - list all containers inside the node
- > kubectl cluster-info
- > kubectl get nodes
    - you see a single node which master 
- > kubectl get pods
    - pods inside the default namespace
- > kubectl get namespaces
    - namesspaces are for grouping different configuration and resources
- > kubectl get pods --namespace=kube-system
    - see all pods inside the kube-system namespace master node
- > kutbectl run ngninx --image=nginx
    - create a pod with nginx container
- > kubectl describe pod nginx
    - describe the pod details
    - which namespace is? 

- inside the node, the pause container is created for each pod to manage the network namespace 
- > docker exec -it <nginx-container-id> bash
    - to access the container terminal
    - connect to the nginx server by curl the ip of the pod inside the container
- > kubectl get pods -o wide
    - get the pod ip address
    - if there are multiple containers inside the pod, they share the same ip address
- > kubectl delete pod nginx
    - delete the pod
- > alias k='kubectl'
    - create alias for kubectl
- > k create deployment nginx-deployment --image=nginx 
    - create deployment
    - now it create a pod with random name 
- > k describe deployment nginx-deployment
    - describe the deployment
    - selector -> label to identify the pod
    - replica set -> manage the number of pods related to this deployment
    - now if you get the k get pods there are two hashes suffixes, the first one is the id of the replica set,      the second one is the id of the pod
     
- > k scale deployment nginx-deployment --replicas=4
    - scale the deployment to 4 pods 
- > k get pods -o wide
    - see 4 pods created
    - each pod has different ip address
    - pods may be on different nodes 

- inside the nodes we can curl the pods but from the local machine we cant access the pods directly

- > minikube ip 
    - get the node ip
    - if there are multiple nodes, each node has its own ip

- pods are managed by deployments. 

- To connect to a specific deployment with a specific ip address we use the service 
    - > k expose deployment nginx-deployment --port=8080 --target-port=80
        - create a service to expose the deployment  
        - now the service will create a stable ip address and port to access the **deployment**
    - > k get services(svc)
        - cluster ip is only accessible inside the cluster 
        - you can access the cluster ip from inside the node
- when we access the service ip address, the service will forward the request to one of the pods behind it (load balancing) and each pod has its own ip address
       
- >  k create deployment hello-node --image=k8s.gcr.io/echoserver:1.4
- >  k expose deployment hello-node --type=NodePort --port=8080 --target-port=8080
    - create a deployment and expose it with NodePort type service
    - NodePort type service will open a specific port on each node to access the service from outside the cluster

- we expose the node through a service of to access the webservice running inside a container inside a pode 

- > minikube service hello-node
    - to access the service from local machine, it will open the browser with the node ip and the node port
    - --url will give only the url without opening the browser

- > k expose deployment nginx-deployment --type=LoadBalancer --port=8080 --target-port=80
    - create a load balancer service to expose the deployment
    - in minikube the load balancer type service will work like NodePort type service

- StrategyType = Rolling Upadate means that when we update the deployment, it will update the pods one by one without downtime

- > k set image deployment nginx-deployment nginx=nginx:1.19.0
    - update the deployment with new image version
- > k rollout status deployment nginx-deployment
    - check the status of the rollout update
- > minikube dashboard
    - open the kubernates dashboard in browser to manage the cluster visually

## Kubernates Practical Declarative

- kubectl apply -f <file.yaml>
    - create resource from yaml file

 
| Concept | Analogy | Visibility | Use Case in MLOps |
|---------|---------|------------|-------------------|
| Namespace | Different Rooms | Logical Isolation | Separating staging vs prod. |
| Service | Receptionist | Abstraction | Stable address for the API, regardless of Pod restarts. |
| ClusterIP | Internal Extension | Cluster Only | The Prediction API (Back-end). |
| LoadBalancer | Public 1-800 Number | Public Internet | The Streamlit UI (Front-end). |




# Useful Commands:
- > lsof -nP -iTCP:PORT | grep LISTEN
    - port checking
- > kubectl port-forward svc/argocd-server -n argocd 8080:443
    - port forward for services on k8s 
