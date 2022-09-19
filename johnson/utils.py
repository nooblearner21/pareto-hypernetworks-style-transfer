import torch
import torchvision.transforms as transforms
from PIL import Image

'''
def image_loader(image_path, imsize=256, device=None):
    if device is None:
        device = torch.device("cpu")

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    image = Image.open(image_path)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
'''
def image_loader(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


'''
def save_image(image, image_name):
    unloader = transforms.ToPILImage()    

    image = image.squeeze(0)
    image = unloader(image)
    image.save(image_name, format="png")

    return image    
'''


def save_image(data, filename):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std