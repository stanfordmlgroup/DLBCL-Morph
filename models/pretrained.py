import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PretrainedModel(nn.Module):
    """Pretrained model, either from Cadene or TorchVision."""

    def __init__(self):
        super(PretrainedModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel ' +
                                  'must implement forward method.')

    def fine_tuning_parameters(self, boundary_layers, lrs):
        """Get a list of parameter groups that can be passed to an optimizer.

        Args:
            boundary_layers: List of names for the boundary layers.
            lrs: List of learning rates for each parameter group, from earlier
            to later layers.

        Returns:
            param_groups: List of dictionaries, one per parameter group.
        """

        def gen_params(start_layer, end_layer):
            saw_start_layer = False
            for name, param in self.named_parameters():
                if end_layer is not None and name == end_layer:
                    # Saw the last layer -> done
                    return
                if start_layer is None or name == start_layer:
                    # Saw the first layer -> Start returning layers
                    saw_start_layer = True

                if saw_start_layer:
                    yield param

        if len(lrs) != boundary_layers + 1:
            raise ValueError(f'Got {boundary_layers + 1} param groups, ' +
                             f'but {lrs} learning rates')

        # Fine-tune the network's layers from encoder.2 onwards
        boundary_layers = [None] + boundary_layers + [None]
        param_groups = []
        for i in range(len(boundary_layers) - 1):
            start, end = boundary_layers[i:i + 2]
            param_groups.append({'params': gen_params(start, end),
                                 'lr': lrs[i]})
        return param_groups


class CadeneModel(PretrainedModel):
    """Models from Cadene's GitHub page of pretrained networks:
        https://github.com/Cadene/pretrained-models.pytorch
    """

    def __init__(self, model_name, model_args=None):
        super(CadeneModel, self).__init__()

        model_class = pretrainedmodels.__dict__[model_name]
        pretrained = "imagenet" if model_args.pretrained else None
        self.model = model_class(num_classes=1000,
                                 pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)

        num_ftrs = self.model.last_linear.in_features
        self.fc = nn.Linear(num_ftrs, model_args.num_classes)

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=False)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class TorchVisionModel(PretrainedModel):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """

    def __init__(self, model_fn, model_args):
        super(TorchVisionModel, self).__init__()

        self.model = model_fn(pretrained=model_args.pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)

        num_outputs = model_args.num_classes

        if 'fc' in self.model.__dict__:
            num_ftrs = self.model.classifier.in_features
            self.model.fc = nn.Linear(num_ftrs, num_outputs)
        elif 'classifier' in self.model.__dict__:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=False)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.classifier(x)
        return x


class DenseNet121(TorchVisionModel):
    def __init__(self, model_args=None):
        super(DenseNet121, self).__init__(models.densenet121, model_args)


class DenseNet161(TorchVisionModel):
    def __init__(self, model_args=None):
        super(DenseNet161, self).__init__(models.densenet161, model_args)


class DenseNet201(TorchVisionModel):
    def __init__(self, model_args=None):
        super(DenseNet201, self).__init__(models.densenet201, model_args)


class ResNet101(TorchVisionModel):
    def __init__(self, model_args=None):
        super(ResNet101, self).__init__(models.resnet101, model_args)


class ResNet152(TorchVisionModel):
    def __init__(self, model_args=None):
        super(ResNet152, self).__init__(models.resnet152, model_args)


class Inceptionv3(TorchVisionModel):
    def __init__(self, model_args=None):
        super(Inceptionv3, self).__init__(models.inception_v3, model_args)


class Inceptionv4(CadeneModel):
    def __init__(self, model_args=None):
        super(Inceptionv4, self).__init__('inceptionv4', model_args)


class ResNet18(CadeneModel):
    def __init__(self, model_args=None):
        super(ResNet18, self).__init__('resnet18', model_args)


class ResNet34(CadeneModel):
    def __init__(self, model_args=None):
        super(ResNet34, self).__init__('resnet34', model_args)


class ResNeXt101(CadeneModel):
    def __init__(self, model_args=None):
        super(ResNeXt101, self).__init__('resnext101_64x4d', model_args)


class NASNetA(CadeneModel):
    def __init__(self, model_args=None):
        super(NASNetA, self).__init__('nasnetalarge', model_args)


class MNASNet(CadeneModel):
    def __init__(self, model_args=None):
        super(MNASNet, self).__init__('nasnetamobile', model_args)


class SENet154(CadeneModel):
    def __init__(self, model_args=None):
        super(SENet154, self).__init__('senet154', model_args)


class SEResNeXt101(CadeneModel):
    def __init__(self, model_args=None):
        super(SEResNeXt101, self).__init__('se_resnext101_32x4d', model_args)
