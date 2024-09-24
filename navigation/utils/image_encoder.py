
import os
from torchvision.models import resnet18 as resnet18_img
import torch
import torch.nn as nn

img_encoder = resnet18_img(num_classes=512)
dim_mlp = img_encoder.fc.weight.shape[1]
img_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), img_encoder.fc)
ckpt_pth = os.path.join('pretrained_models/Img_encoder.pth.tar')
ckpt = torch.load(ckpt_pth, map_location='cpu')
state_dict = {k[len('module.encoder_q.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_q.' in k}
img_encoder.load_state_dict(state_dict)
img_encoder.eval().cuda()

def get_image_feat(obs_rgb):
    img_tensor = (obs_rgb/255.).permute(0,3,1,2)
    feat = img_encoder(img_tensor)
    vis_embedding = nn.functional.normalize(feat, dim=1)
    return vis_embedding