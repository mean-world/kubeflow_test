# kubeflow_test
## Overview
Use kubeflow to quickly create distributed training environment on kubernetes cluster
## outline
```
1.install docker set nvidia container runtime
2.install and set kubeadm
3.create k8s cluster(kubeadm init)
4.install kubeflow and kubeflow python SDK
5.~waiting
```
## 1.install docker set nvidia container runtime
```
#Delete conflicting or old version packages
**for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done**
```
