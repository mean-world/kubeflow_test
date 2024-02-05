
import datetime

import pytz

from kubernetes import client, config


def create_deployment_object(cpu, memory, replicas_num, DEPLOYMENT_NAME):
    # Configureate Pod template container
    container = client.V1Container(
        name="ray-worker",
        image="pear1798/ray-test:v2",
        resources=client.V1ResourceRequirements(
            requests={"cpu": cpu, "memory": memory},
            limits={"cpu": cpu, "memory": memory},
        ),
        command = ["ray"],
        args = ["start", "--address=ray-head-svc:6379", "--block", "--num-cpus=2", "--memory=2000000000"]

    )

    init_container = client.V1Container(
        name="test-head-status",
        image="busybox:1.28",
        command = ["sleep"],
        args = ["100"]
    )



    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"type": "worker"}),
        spec=client.V1PodSpec(containers=[container], init_containers=[init_container], service_account_name="default"),
    )

    # Create the specification of deployment
    spec = client.V1DeploymentSpec(
        replicas=replicas_num, template=template, selector={
            "matchLabels":
            {"type": "worker"}})

    # Instantiate the deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=DEPLOYMENT_NAME),
        spec=spec,
    )

    return deployment


def create_deployment(api, deployment, namespace="ray"):
    # Create deployment
    resp = api.create_namespaced_deployment(
        body=deployment, namespace=namespace
    )

    print("\n[INFO] deployment `nginx-deployment` created.\n")
    print("%s\t%s\t\t\t%s\t%s" % ("NAMESPACE", "NAME", "REVISION", "IMAGE"))
    print(
        "%s\t\t%s\t%s\t\t%s\n"
        % (
            resp.metadata.namespace,
            resp.metadata.name,
            resp.metadata.generation,
            resp.spec.template.spec.containers[0].image,
        )
    )





#setting
DEPLOYMENT_NAME = "ray-worker"
cpu = "2"
memory = "2G"
replicas_num = 3
namespace = "user-name"

#create ray woker
config.load_incluster_config()
apps_v1 = client.AppsV1Api()
deployment = create_deployment_object(cpu, memory, replicas_num, DEPLOYMENT_NAME)
create_deployment(apps_v1, deployment, namespace)

#create ray heads
