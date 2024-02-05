from kubernetes import client, config

def create_pod(namespace):
    #container
    container = client.V1Container(
        name="ray-head",
        image="pear1798/ray-test:v2",
        ports=[client.V1ContainerPort(container_port=6379), client.V1ContainerPort(container_port=8265)],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "2", "memory": "2G"},
            limits={"cpu": "2", "memory": "2G"},
        ),
        command = ["ray"],
        args = ["start", "--block", "--head", "--dashboard-host=0.0.0.0", "--num-cpus=2", "--memory=2000000000"]
    )

    #spec
    spec = client.V1PodSpec(containers=[container], restart_policy="Always", service_account_name="default")

    #pod
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name="ray-head-test"),
        spec=spec,
    )

    v1 = client.CoreV1Api()
    result = v1.create_namespaced_pod(namespace=namespace, body=pod)


#create ray heads
config.load_incluster_config()
create_pod("ray")