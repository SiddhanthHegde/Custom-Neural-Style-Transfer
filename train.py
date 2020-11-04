#imports
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_image, content_loss_calc, style_loss_calc

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import matplotlib.pyplot as plt
#%matplotlib inline

#Command line arguments. 
arg_parser = argparse.ArgumentParser(
    description="parser for fast-neural-style-training")
    
arg_parser.add_argument("--content-img", type=str, required=True,
                        help="path to content image")
arg_parser.add_argument("--style-img", type=str, required=True,
                        help="path to style image")
arg_parser.add_argument("--output-path", type=str, required=True,
                        help="path to stylized image"
                        "containing another folder with all the training images")
arg_parser.add_argument("--output-name", type=str, default="output.png",
                        help="name of stylized image")
arg_parser.add_argument("--step-amount", type=int, default=10000,
                        help="amount of painting steps (loops, default = 10000)")
arg_parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate (default = 1e-2")

args = arg_parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#checking the layers in mobilenet v2
model = models.mobilenet_v2(pretrained=True).features
print(model)

class mobNet(nn.Module):
    def __init__(self):
        super(mobNet,self).__init__()
        self.req_layers = [3,6,9,12,15] #these layers' features will be extracted
        self.model = models.mobilenet_v2(pretrained=True).features[:17]

    def forward(self,x):
        features = []

        for layer, layername in enumerate(self.model):
          x = layername(x)
          if layer in self.req_layers:
            features.append(x)

        return features

image_size = 1024 #preferred the shape of content image for better look
transform = transforms.Compose(
    [
     transforms.Resize((316,474)),# shape of used content image to retain the shape. Noise image can be rescaled to higher dimensions for better clarity
     transforms.ToTensor()
    ]
)

#load images
content_image = load_image(args.content_img)
style_image = load_image(args.style_img)

#generated image can be a noise
#generated_image = torch.randn(orignal_image.shape).to(device).requires_grad_(True)
generated_image = content_image.clone().requires_grad_(True)

model = mobNet().to(device).eval()

#hyperparameters
total_steps = args.step_amount #default = 10000
learning_rate = args.lr #default = 1e-2
alpha = 1 #content loss weight
beta = 0.01 #style loss weight
optimizer = optim.Adam([generated_image],lr = learning_rate)
steps = []
total_losses = []

for step in range(total_steps):
    generated_features = model(generated_image)
    content_features = model(content_image)
    style_features = model(style_image)

    style_loss = content_loss = 0

    for generated_feature, content_feature, style_feature in zip(
            generated_features, content_features, style_features):
        batch_size, channel, height, width = generated_feature.shape

        content_loss += content_loss_calc(generated_feature, content_feature)
        style_loss += style_loss_calc(generated_feature, style_feature, channel,
                                      width, height)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()

    total_loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f'{time.ctime()} - tensor(total_loss: {total_loss}, device={device}) steps: {step}/ {total_steps}')
        steps.append(step)
        total_losses.append(float(total_loss))

save_model_path = os.path.join(args.output_path, args.output_name)
save_image(generated_image, save_model_path)

plt.figure(figsize=(10,8))
plt.plot(steps,total_losses)
