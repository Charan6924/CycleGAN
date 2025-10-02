import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=4,padding=1,bias=True,padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,X):
        return self.conv(X)
    
class Discriminator(nn.Module):
    def __init__(self,in_channels = 3, features = [64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=4,
                stride = 2,
                padding = 1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels=in_channels,out_channels = feature, stride = 1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self,X):
        X = self.initial(X)
        return torch.sigmoid(self.model(X))
    
def test():
    x = torch.randn([5,3,256,256])
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()