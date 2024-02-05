from kubernetes import client, config


def create_service_dashboard(namespace):
    api_instance = client.CoreV1Api()

    service = client.V1Service()

    service.api_version = "v1"
    service.kind = "Service"
    service.metadata = client.V1ObjectMeta(name="dashboard-svc")

    spec = client.V1ServiceSpec(type="NodePort")
    spec.selector = {"ray-type": "head"}
    spec.ports = [client.V1ServicePort(protocol="TCP", port=8265, target_port=8265, node_port=30006, name="dashboard")]
    service.spec = spec

    api_instance.create_namespaced_service(namespace=namespace, body=service)

def create_service_head_register(namespace):
    api_instance = client.CoreV1Api()

    service = client.V1Service()

    service.api_version = "v1"
    service.kind = "Service"
    service.metadata = client.V1ObjectMeta(name="ray-head-svc")

    spec = client.V1ServiceSpec()
    spec.selector = {"ray-type": "head"}
    spec.ports = [client.V1ServicePort(protocol="TCP", port=6379, target_port=6379, name="head")]
    service.spec = spec

    api_instance.create_namespaced_service(namespace=namespace, body=service)

config.load_incluster_config()
create_service_dashboard("ray")
create_service_head_register("ray")