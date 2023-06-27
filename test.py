import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfnet_src import MMFNet

img = torch.ones(2,3,224, 224)
x_struc = torch.ones(2, 5)
model = MMFNet(num_classes=2)
out = model(img, x_struc)
print(out)