from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from vgg import get_content_features, get_style_features
from utils import image_loader, save_image, normalize_batch
from transformer_net import StyleTransferNet

def gram_matrix_get(tensor):
    (batch, channel, height, width) = tensor.size()
    features = tensor.view(batch, channel, height * width)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (channel * height * width)
    return gram


def content_loss_get(layers_C, layers_G):
    if (len(layers_C) != len(layers_G)):
        raise Exception("Number of vgg layers from content image is different from generated image")

    if (len(layers_C) == 0):
        raise Exception("No layers were given as input")

    n = len(layers_C)
    mse_loss = nn.MSELoss()
    loss = 0

    for i in range(n):
        loss += mse_loss(layers_C[i], layers_G[i])

    #weighted average
    return loss / n


def style_loss_get(layers_S, layers_G):
    if (len(layers_S) != len(layers_G)):
        raise Exception("Number of vgg layers from style image is different from input image")

    if (len(layers_S) == 0):
        raise Exception("No layers were given as input")

    n = len(layers_S)
    mse_loss = nn.MSELoss()
    loss = 0

    for i in range(n):
        style_gram_matrix = gram_matrix_get(layers_S[i])
        generated_gram_matrix = gram_matrix_get(layers_G[i])

        loss += mse_loss(style_gram_matrix, generated_gram_matrix)

    #weighted average
    return loss / n


def total_loss_get(vgg_model, content_images_batch, style_images_batch, generated_images_batch):
    layers_C = get_content_features(vgg_model, content_images_batch)
    layers_S = get_style_features(vgg_model, style_images_batch)
    layers_G_C = get_content_features(vgg_model, generated_images_batch)
    layers_G_S = get_style_features(vgg_model, generated_images_batch)

    content_loss = content_loss_get(layers_C, layers_G_C)
    style_loss = style_loss_get(layers_S, layers_G_S)

    return content_loss, style_loss


def stylize(content_image_path, model_path, imsize=256):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    content_image = image_loader(content_image_path, imsize, device)

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        model = StyleTransferNet()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device)

        output = model(content_image).cpu()

        #New image name
        image_name = Path(content_image_path).stem + "-stylized" + ".png"
        save_image(output[0], image_name)


def train_network(dataset, style_image_path, trained_models_output_path, batch_size, epochs, lr, content_weight, style_weight):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()

    # Handling style input image
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_image = image_loader(style_image_path, size=256)
    style_image = style_transform(style_image)
    style_image = style_image.to(device)
    style_image = normalize_batch(style_image)
    style_image = style_image.repeat(batch_size, 1, 1, 1).to(device)

    #Initialize style network
    style_network = StyleTransferNet()
    style_network.to(device)   

    #Load dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Initialize Optimizer
    optimizer = torch.optim.Adam(style_network.parameters(), lr)

    #Train network
    for epoch in range(epochs):

        loss_sum = 0
        torch.cuda.empty_cache()

        for batch_num, (content_batch, _) in enumerate(train_loader):

            content_batch = content_batch.to(device)
            content_batch = normalize_batch(content_batch)
            optimizer.zero_grad()

            #Generate stylized images
            generated_images_batch = style_network(content_batch)
            generated_images_batch = normalize_batch(generated_images_batch)

            #Calculate Loss
            content_loss, style_loss = total_loss_get(vgg_model, content_batch, style_image, generated_images_batch)
            total_loss = content_weight * content_loss + style_weight * style_loss

            loss_sum += total_loss.item()

            #Update network parameters
            total_loss.backward()
            optimizer.step()

            if(batch_num % 100 == 0):
                print(f"Batch: #{batch_num} --- Content Loss: {content_loss}, Style Loss: {style_loss}, Total Loss: {loss_sum}")
                loss_sum = 0

            if(batch_num % 100 == 0):
              model_name = f"{trained_models_output_path}/model_epoch_{epoch}_batch_{batch_num}.pth"
            
            # print("Epoch: #{} --- Current Loss:{}".format(epoch, loss_sum))
              torch.save(style_network.state_dict(), model_name)

