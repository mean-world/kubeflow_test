
import datetime
from kubernetes import client, config

config.load_incluster_config()


#create ray heads

class kubernetes_control():
    def __init__(self, namespace="default"):
        self.namespace = namespace #user name

    #create ray worker
    def create_deployment(self, cpu, memory, replicas_num):
        cpu_args = "--num-cpus=" + str(cpu)
        memory_args = "--memory=" + str(memory*1000000000)
        memory = str(memory) + "G"
        container = client.V1Container(
        name="ray-worker",
        image="pear1798/ray-test:v2",
        resources=client.V1ResourceRequirements(
            requests={"cpu": cpu, "memory": memory},
            limits={"cpu": cpu, "memory": memory},
        ),
        command = ["ray"],
        args = ["start", "--address=ray-head-svc:6379", "--block", cpu_args, memory_args]
        )

        init_container = client.V1Container(
            name="test-head-status",
            image="busybox:1.28",
            command = ["sleep"],
            args = ["100"]
        )

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"type": "worker"}),
            spec=client.V1PodSpec(containers=[container], init_containers=[init_container]),
        )

        spec = client.V1DeploymentSpec(
            replicas=replicas_num, template=template, selector={
                "matchLabels":
                {"type": "worker"}})


        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name="ray-worker"),
            spec=spec,
        )

        apps_v1 = client.AppsV1Api()
        apps_v1.create_namespaced_deployment(body=deployment, namespace=self.namespace)

    #create ray head
    def create_pod(self):
        #container
        container = client.V1Container(
            name="ray-head",
            image="pear1798/ray-test:v2",
            ports=[client.V1ContainerPort(container_port=6379), client.V1ContainerPort(container_port=8265)],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "4", "memory": "4G"},
                limits={"cpu": "4", "memory": "4G"},
            ),
            command = ["ray"],
            args = ["start", "--block", "--head", "--dashboard-host=0.0.0.0", "--num-cpus=4", "--memory=4000000000"]
        )

        #spec
        spec = client.V1PodSpec(containers=[container], restart_policy="Always")

        #pod
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(name="ray-head-test"),
            spec=spec,
        )

        v1 = client.CoreV1Api()
        v1.create_namespaced_pod(namespace=self.namespace, body=pod)

    #create ray service for ray head dashboard
    def create_service_dashboard(self):
        api_instance = client.CoreV1Api()

        service = client.V1Service()

        service.api_version = "v1"
        service.kind = "Service"
        service.metadata = client.V1ObjectMeta(name="ray-dashboard-svc")

        spec = client.V1ServiceSpec(type="NodePort")
        spec.selector = {"ray-type": "head"}
        spec.ports = [client.V1ServicePort(protocol="TCP", port=8265, target_port=8265, node_port=30006, name="dashboard")]
        service.spec = spec

        api_instance.create_namespaced_service(namespace=self.namespace, body=service)

    #create ray service to connect ray head
    def create_service_head_register(self):
        api_instance = client.CoreV1Api()

        service = client.V1Service()

        service.api_version = "v1"
        service.kind = "Service"
        service.metadata = client.V1ObjectMeta(name="ray-head-svc")

        spec = client.V1ServiceSpec()
        spec.selector = {"ray-type": "head"}
        spec.ports = [client.V1ServicePort(protocol="TCP", port=6379, target_port=6379, name="head")]
        service.spec = spec

        api_instance.create_namespaced_service(namespace=self.namespace, body=service)

    #create user name
    def create_namespace(self):
        v1 = client.CoreV1Api()
        v1.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=self.namespace)))

k8s_cmd = kubernetes_control("user1")
#create ray head
k8s_cmd.create_namespace()
k8s_cmd.create_pod()
k8s_cmd.create_service_dashboard()
k8s_cmd.create_service_head_register()
#create ray worker
k8s_cmd.create_deployment(cpu=2, memory=2, replicas_num=3)
