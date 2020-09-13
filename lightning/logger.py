import torch
import torch.nn.functional as F

from util.constants import IMAGENET_MEAN, IMAGENET_STD


class TFLogger:
    def log_images(self, images, tag, size=125):
        """
        Log images and optionally detection to tensorboard
        :param logger: [Tensorboard Logger] Tensorboard logger object.
        :param images: [tensor] batch of images indexed
                    [batch, channel, size1, size2]
        TODO: Include an argument for image labels;
            Print the labels on the images.
        """
        images = prep_images_for_logging(images,
                                         pretrained=self.args['pretrained'],
                                         size=size)
        self.logger.experiment.add_images(tag, images)


def prep_images_for_logging(images, pretrained=False,
                            size=125):
    """
    Prepare images to be logged
    :param images: [tensor] batch of images indexed
                   [channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :param size: [int] new size of the image to be rescaled
    :return: images that are reversely normalized
    """
    if pretrained:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    images = normalize_inverse(images, mean, std)
    images = F.interpolate(images, size=size,
                           mode='bilinear', align_corners=True)
    return images


def normalize_inverse(images, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse Normalization of Pytorch Tensor
    :param images: [tensor] batch of images indexed
                   [batch, channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :return: images that are reversely normalized
    """
    mean_inv = torch.FloatTensor(
        [-m/s for m, s in zip(mean, std)]).view(1, 3, 1, 1)
    std_inv = torch.FloatTensor([1/s for s in std]).view(1, 3, 1, 1)
    if torch.cuda.is_available():
        mean_inv = mean_inv.cuda()
        std_inv = std_inv.cuda()
    return (images - mean_inv) / std_inv
