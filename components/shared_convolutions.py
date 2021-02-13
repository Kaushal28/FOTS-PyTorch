import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class SharedConvolutions(nn.Module):
    """
    Shared convolution layers to extract shared features for both
    text detection and recognition branch of FOTS.
    """

    def __init__(self, back_bone: nn.Module):
        """Constructor."""
        super().__init__()
        self.back_bone = back_bone

        # As we are using backbone as feature extractor,
        # set it to evaluation mode.
        self.back_bone.eval()

        # Freeze the weights of backbone (because it's feature extractor).
        for param in self.back_bone.parameters():
            param.required_grad = False
        
        # Define concat layers
        self.concat4 = ConcatLayer(2048 + 1024, 128)
        self.concat3 = ConcatLayer(128 + 512, 64)
        self.concat2 = ConcatLayer(64 + 256, 32)

        self.conv = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(32, momentum=0.003)
    
    def forward(self, x):
        """Module's forward pass."""
        # First extract features using back bone network.
        res5, res4, res3, res2 = self._extract_features(x)

        # As per the paper, res5 is not concatenated with any
        # high level feature map. So apply deconv directly
        res5_deconv = self._deconv(res5)

        # Concat deconv'ed res5 with res4
        res4_concat = self.concat4(res5_deconv, res4)
        res4_deconv = self._deconv(res4_concat)

        # Concat deconv'ed res4 with res3
        res3_concat = self.concat3(res4_deconv, res3)
        res3_deconv = self._deconv(res3_concat)

        # Concat deconv'ed res3 with res2
        res2_concat = self.concat2(res3_deconv, res2)
        
        # Pass the final output to 1 conv and bn layers
        output = self.conv(res2_concat)
        output = self.bn(output)
        output = F.relu(output)

        return output

    def _extract_features(self, x):
        """Extract features from given input and backbone."""
        res5 = self.back_bone.layer4(x)
        res4 = self.back_bone.layer3(x)
        res3 = self.back_bone.layer2(x)
        res2 = self.back_bone.layer1(x)

        return res5, res4, res3, res2
    
    def _deconv(self, feature):
        """
        Apply deconv operation (inverse of pooling) on given feature map.
        """
        # Upsample the given feature.
        # Doc: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        return F.interpolate(
            feature,
            mode='bilinear',
            scale_factor=2,  # As per the paper
            align_corners = True
        )
    

class ConcatLayer(nn.Module):
    """Concatenates given feature maps."""
    
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.003)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.003)

    def forward(self, prev_features, curr_features):
        """Forward pass."""
        concated_features = torch.cat([prev_features, curr_features], dim=1)

        output = self.conv1(concated_features)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = F.relu(output)

        return output
