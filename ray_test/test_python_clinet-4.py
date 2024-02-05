from kubernetes import client, config

def create_role(namespace):
    api = client.RbacAuthorizationV1Api()
    role = client.V1Role(
        api_version="rbac.authorization.k8s.io/v1",
        kind="Role",
        metadata=client.V1ObjectMeta(name="control-role"),
        rules=[
            client.V1PolicyRule(
                api_groups=["apps"],
                resources=["deployments"],
                verbs=["get", "watch", "list", "create", "update", "delete"]
            ),
            client.V1PolicyRule(
                api_groups=[""],
                resources=["pods"],
                verbs=["get", "watch", "list", "create", "update", "delete"]
            ),
            client.V1PolicyRule(
                api_groups=[""],
                resources=["services"],
                verbs=["get", "watch", "list", "create", "update", "delete"]
            )
        ]
    )

    api.create_namespaced_role(namespace=namespace, body=role)

def create_rolebinding(namespace):
    api = client.RbacAuthorizationV1Api()
    rolebinding = client.V1RoleBinding(
        api_version="rbac.authorization.k8s.io/v1",
        kind="RoleBinding",
        metadata=client.V1ObjectMeta(name="control-role-rolebinding"),
        subjects=[
            client.RbacV1Subject(
                kind="ServiceAccount",
                name="default"
            )
        ],
        role_ref=client.V1RoleRef(
            kind="Role",
            name="control-role",
            api_group="rbac.authorization.k8s.io"
        )
    )

    api.create_namespaced_role_binding(namespace=namespace, body=rolebinding)

config.load_incluster_config()

create_role("ray")
create_rolebinding("ray")
