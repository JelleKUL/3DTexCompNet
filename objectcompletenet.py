import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """
    A basic 3D residual block with two convolutional layers.
    Helps preserve features while increasing depth.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x  # Save input for skip connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)  # Add skip connection and apply activation


class ObjectCompletionNet(nn.Module):
    """
    A 3D object completion network that predicts full 3D textures (RGB + SDF)
    from partial observations.
    
    Args:
        in_channels (int): Number of input channels. Default is 4 (RGB + SDF).
        base_channels (int): Number of base feature channels. Controls network width.
    """
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()

        # Encoder: progressively downsample and extract features
        self.enc1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)  # No downsampling
        self.enc2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)  # Downsample x2
        self.enc3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)  # Downsample x4

        # Bottleneck: process high-level features
        self.resblock = ResidualBlock3D(base_channels*4)

        # Decoder: upsample and reconstruct RGB + SDF
        self.dec3 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1)  # Upsample x2
        self.dec2 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)  # Upsample x4
        self.dec1 = nn.Conv3d(base_channels, 4, kernel_size=3, padding=1)  # Output: 3 RGB channels + 1 SDF

    def forward(self, x):
        # Encoding phase
        x1 = F.relu(self.enc1(x))  # Initial conv layer
        x2 = F.relu(self.enc2(x1))  # Downsample once
        x3 = F.relu(self.enc3(x2))  # Downsample twice

        # Bottleneck with residual connections
        x3 = self.resblock(x3)

        # Decoding phase
        x = F.relu(self.dec3(x3))  # Upsample once
        x = F.relu(self.dec2(x))  # Upsample to original resolution
        x = self.dec1(x)  # Final prediction: 4 channels (RGB + SDF)

        return x


if __name__ == "__main__":
    # Example forward pass for testing
    model = ObjectCompletionNet()
    input_tensor = torch.randn(2, 4, 64, 64, 64)  # Batch size 2, 4 channels, 64^3 grid
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (2, 4, 64, 64, 64)
