from torch import nn
import torch
from math import ceil

# Simple convolution class with normalization and activation

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1,
                 padding=0, groups=1, norm=True, activation=True, bias=False):
        
        super(ConvNormAct, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)
        
        if norm:
            self.batch_normalizer = nn.BatchNorm2d(out_channels)
        else:
            self.batch_normalizer = nn.Identity()
            
        if activation:
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
            
            
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normalizer(x)
        x = self.activation(x)
        return x
    
            
# Squeeze and Excitation Block (Hu et al., 2018)

class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, r_dim):
        super(SqueezeAndExcitation, self).__init__()
        
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, r_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(r_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
       
    
    def forward(self, x):
        y_hat = self.seq(x)
        
        return x * y_hat
    
        
# Stochastic Depth Block (Huang et al., 2016)

class StochasticDepth(nn.Module):
    def __init__(self, survival_proba=0.8):
        super(StochasticDepth, self).__init__()
        
        self.proba = survival_proba
        
        
    def forward(self, x):
        if not self.training:
            return x
        
        bin_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.proba
        
        return torch.div(x, self.proba) * bin_tensor
    
    
# Mobile Inverted Bottleneck (Sandler et al., 2018; Tan et al., 2019)

class MobileInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3,
                stride=1, expansion_factor=6, reduction=4,
                survival_proba=0.8):
        super(MobileInvertedBottleneck, self).__init__()
        
        self.skip_connection = stride == 1 and in_channels == out_channels
        intermediate_chans = int(in_channels * expansion_factor)
        padding = (ksize - 1) // 2
        reduced_dim = int(in_channels // reduction)

        if expansion_factor == 1:            
            self.expand = nn.Identity()
        else:
            self.expand = ConvNormAct(in_channels, intermediate_chans, ksize=1)
            
        
        self.depthwise_conv = ConvNormAct(intermediate_chans, intermediate_chans,
                                         ksize=ksize, stride=stride,
                                         padding=padding, groups=intermediate_chans)
        
        self.seq = SqueezeAndExcitation(intermediate_chans, r_dim=reduced_dim)
        
        self.pointwise_conv = ConvNormAct(intermediate_chans, out_channels,
                                         ksize=1, activation=False)
        
        self.drop_layers = StochasticDepth(survival_proba=survival_proba)
        
    
    def forward(self, x):
        residual = x
        
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.seq(x)
        x = self.pointwise_conv(x)
        
        if self.skip_connection:
            x = self.drop_layers(x)
            x += residual
            
        return x
    
    
class EfficientNet(nn.Module):
    def __init__(self, w=1, d=1, dropout=0.2, num_classes=4):
        super(EfficientNet, self).__init__()
        
        last_channel = ceil(1280 * w)
        self.features = self._feature_extractor(w, d, last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes)
        )
              
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.out_layer(x.view(x.shape[0], -1))
        
        return x
    
    
    def _feature_extractor(self, w, d, last_channel):
        channels = 4 * ceil(int(32 * w) / 4)
        layers = [ConvNormAct(1, channels, ksize=3, stride=2, padding=1)]
        in_channels = channels
        
        kernels = [3, 3, 5, 3, 5, 5, 3]
        num_channels = [16, 24, 40, 80, 112, 192, 320]
        num_layers = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        
        scaled_num_channels = [4 * ceil(int(c * w) / 4) for c in num_channels]
        scaled_num_layers = [int(depth * d) for depth in num_layers]
        
        for i in range(len(scaled_num_channels)):
            layers += [MobileInvertedBottleneck(in_channels if repeat == 0 else scaled_num_channels[i],
                                               scaled_num_channels[i],
                                               ksize = kernels[i],
                                               stride = strides[i] if repeat == 0 else 1,)
                                                for repeat in range(scaled_num_layers[i])]
            
            in_channels = scaled_num_channels[i]
            
        layers.append(ConvNormAct(in_channels, last_channel, ksize=1, stride=1, padding=0))
        
        
        return nn.Sequential(*layers)