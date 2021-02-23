import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# class SharedConvolutions(nn.Module):
#     """
#     Shared convolution layers to extract shared features for both
#     text detection and recognition branch of FOTS.
#     """

#     def __init__(self, back_bone: nn.Module):
#         """Constructor."""
#         super().__init__()
#         self.back_bone = back_bone

#         # As we are using backbone as feature extractor,
#         # set it to evaluation mode.
#         self.back_bone.eval()

#         # Freeze the weights of backbone (because it's feature extractor).
#         for param in self.back_bone.parameters():
#             param.required_grad = False
        
#         # Define concat layers
#         self.concat4 = ConcatLayer(2048 + 1024, 128)
#         self.concat3 = ConcatLayer(128 + 512, 64)
#         self.concat2 = ConcatLayer(64 + 256, 32)

#         self.conv = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
#         self.bn = nn.BatchNorm2d(32, momentum=0.003)
    
#     def forward(self, x):
#         """Module's forward pass."""

#         # Subtract the mean from the image.
#         x = self._mean_image_subtraction(x)

#         # First extract features using back bone network.
#         res5, res4, res3, res2 = self._extract_features(x)

#         # As per the paper, res5 is not concatenated with any
#         # high level feature map. So apply deconv directly
#         res5_deconv = self._deconv(res5)

#         # Concat deconv'ed res5 with res4
#         res4_concat = self.concat4(res5_deconv, res4)
#         res4_deconv = self._deconv(res4_concat)

#         # Concat deconv'ed res4 with res3
#         res3_concat = self.concat3(res4_deconv, res3)
#         res3_deconv = self._deconv(res3_concat)

#         # Concat deconv'ed res3 with res2
#         res2_concat = self.concat2(res3_deconv, res2)
        
#         # Pass the final output to 1 conv and bn layers
#         output = self.conv(res2_concat)
#         output = self.bn(output)
#         output = F.relu(output)

#         return output

#     def _extract_features(self, x):
#         """Extract features from given input and backbone."""
#         x = self.back_bone.conv1(x)
#         x = self.back_bone.bn1(x)
#         x = self.back_bone.relu(x)
#         x = self.back_bone.maxpool(x)
#         res2 = self.back_bone.layer1(x)
#         res3 = self.back_bone.layer2(res2)
#         res4 = self.back_bone.layer3(res3)
#         res5 = self.back_bone.layer4(res4)

#         return res5, res4, res3, res2
    
#     def _mean_image_subtraction(self, images, means=[123.68, 116.78, 103.94]):
#         """
#         Image Standardization. Subtracts the mean from the given image.
#         """
#         num_channels = images.data.shape[1]
#         if len(means) != num_channels:
#             raise ValueError('len(means) must match the number of channels')
#         for i in range(num_channels):
#             images.data[:, i, :, :] -= means[i]

#         return images

#     def _deconv(self, feature):
#         """
#         Apply deconv operation (inverse of pooling) on given feature map.
#         """
#         # Upsample the given feature.
#         # Doc: https://pytorch.org/docs/stable/nn.functional.html#interpolate
#         return F.interpolate(
#             feature,
#             mode='bilinear',
#             scale_factor=2,  # As per the paper
#             align_corners = True
#         )
    

# class ConcatLayer(nn.Module):
#     """Concatenates given feature maps."""
    
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.003)

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.003)

#     def forward(self, prev_features, curr_features):
#         """Forward pass."""
#         concated_features = torch.cat([prev_features, curr_features], dim=1)

#         output = self.conv1(concated_features)
#         output = self.bn1(output)
#         output = F.relu(output)

#         output = self.conv2(output)
#         output = self.bn2(output)
#         output = F.relu(output)

#         return output

class SharedConvolutions(nn.Module):
    """
    Shared convolution layers to extract shared features for both
    text detection and recognition branch of FOTS.
    """

    def __init__(self, back_bone):
        """Constructor."""
        super().__init__()
        self.backbone = back_bone
        self.backbone.eval()

        self.mergeLayers0 = DummyLayer()

        self.mergeLayers1 = ConcatLayer(2048 + 1024, 128)
        self.mergeLayers2 = ConcatLayer(128 + 512, 64)
        self.mergeLayers3 = ConcatLayer(64 + 256, 32)

        self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

    def forward(self, input):

        input = self._mean_image_subtraction(input)

        f = self._extract_features(input)

        g = [None] * 4
        h = [None] * 4

        h[0] = self.mergeLayers0(f[0])
        g[0] = self._deconv(h[0])

        h[1] = self.mergeLayers1(g[0], f[1])
        g[1] = self._deconv(h[1])

        h[2] = self.mergeLayers2(g[1], f[2])
        g[2] = self._deconv(h[2])

        h[3] = self.mergeLayers3(g[2], f[3])

        final = self.mergeLayers4(h[3])
        final = self.bn5(final)
        final = F.relu(final)

        return final

    def _extract_features(self, input):
        conv2 = None
        conv3 = None
        conv4 = None
        output = None # n * 7 * 7 * 2048

        for name, layer in self.backbone.named_children():
            input = layer(input)
            if name == 'layer1':
                conv2 = input
            elif name == 'layer2':
                conv3 = input
            elif name == 'layer3':
                conv4 = input
            elif name == 'layer4':
                output = input
                break

        return output, conv4, conv3, conv2

    def _deconv(self, input):
        """
        Apply deconv operation (inverse of pooling) on given feature map.
        """
        _, _, H, W = input.shape
        # Upsample the given feature.
        # Doc: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        return F.interpolate(input, mode = 'bilinear', scale_factor=2, align_corners=True)

    def _mean_image_subtraction(self, images, means = [123.68, 116.78, 103.94]):
        '''
        image normalization
        :param images: bs * w * h * channel
        :param means:
        :return:
        '''
        num_channels = images.data.shape[1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        for i in range(num_channels):
            images.data[:, i, :, :] -= means[i]

        return images


class DummyLayer(nn.Module):

    def forward(self, input_f):
        return input_f


class ConcatLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        """
        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(ConcatLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size = 1)
        self.bnOne = nn.BatchNorm2d(outputChannels, momentum=0.003)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size = 3, padding = 1)
        self.bnTwo = nn.BatchNorm2d(outputChannels, momentum=0.003)

    def forward(self, inputPrevG, inputF):
        input = torch.cat([inputPrevG, inputF], dim = 1)
        output = self.conv2dOne(input)
        output = self.bnOne(output)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output