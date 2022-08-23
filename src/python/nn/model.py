from sresnet import Model as ModelSRN, BasicBlock
from sresnet import Model_ssl as ModelSRN_ssl
from sresnet import Model_backbone as ModelSRN_backbone
from ssdresnet import Model as ModelSDRN
from ssdresnet import Model_ssl as ModelSDRN_ssl
from ssdresnet import Model_backbone as ModelSDRN_backbone


def fetch_model(
    name="ModelSRN",
    num_classes=2,
    kernel_size=3,
    inject_size=False,
    gpu=False,
):
    if name == "ModelSRN":
        layers = [3, 4, 3]
        model = ModelSRN(
            BasicBlock,
            layers,
            num_classes,
            inject_size,
            kernel_size=kernel_size
        )

    elif name == "ModelSDRN":
        layers = [3, 4, 3]
        model = ModelSDRN(
            BasicBlock,
            layers,
            num_classes,
            inject_size,
            kernel_size=kernel_size,
            scale_factor=3,
            gpu=gpu,
        )
    return model


def fetch_model_ssl(
    name,
    inject_size,
    kernel_size=3,
    feature_dim=64,
    gpu=False,
):
    if name == "ModelSRN":
        layers = [3, 4, 3]
        model = ModelSRN_ssl(
            BasicBlock,
            layers,
            inject_size,
            kernel_size=kernel_size,
            feature_dim=feature_dim,
        )

    elif name == "ModelSDRN":
        layers = [3, 4, 3]
        model = ModelSDRN_ssl(
            BasicBlock,
            layers,
            inject_size,
            kernel_size=kernel_size,
            feature_dim=feature_dim,
            scale_factor=3,
            gpu=gpu,
        )
    return model


def fetch_backbone(
    name="ModelSRN",
    num_classes=2,
    kernel_size=3,
    inject_size=False,
    gpu=False,
):
    if name == "ModelSRN":
        layers = [3, 4, 3]
        model = ModelSRN_backbone(
            BasicBlock,
            layers,
            num_classes,
            inject_size,
            kernel_size=kernel_size,
            gpu=gpu,
        )

    elif name == "ModelSDRN":
        layers = [3, 4, 3]
        model = ModelSDRN_backbone(
            BasicBlock,
            layers,
            num_classes,
            inject_size,
            kernel_size=kernel_size,
            scale_factor=3,
            gpu=gpu,
        )
    return model
