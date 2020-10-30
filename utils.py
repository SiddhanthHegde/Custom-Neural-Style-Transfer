import torch
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose(
    [
     transforms.Resize((316,474)),# shape of used content image
     transforms.ToTensor()
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device)


def content_loss_calc(gen_image,org_image):
  squared = (gen_image - org_image) ** 2
  return torch.mean(squared)

def style_loss_calc(gen_image,style_image,channel,width,height):
  G = gen_image.view(channel, height * width).mm(
      gen_image.view(channel,height * width).t()
  )

  A = style_image.view(channel, height * width).mm(
      style_image.view(channel, height * width).t()
  )

  squared = (G - A) ** 2
  return torch.mean(squared)



