from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from solver import LinearScalarizationSolver
from transformer_net import StyleTransferNet, Hyper
from vgg import get_content_features, get_style_features
from utils import image_loader, save_image, normalize_batch, load_modules_weights, save_modules_weights

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
    return (loss / n)


def total_loss_get(vgg_model, content_images_batch, style_images_batch, generated_images_batch):
    layers_C = get_content_features(vgg_model, content_images_batch)
    layers_S = get_style_features(vgg_model, style_images_batch)
    layers_G_C = get_content_features(vgg_model, generated_images_batch)
    layers_G_S = get_style_features(vgg_model, generated_images_batch)

    content_loss = content_loss_get(layers_C, layers_G_C)
    style_loss = style_loss_get(layers_S, layers_G_S)

    return content_loss, style_loss


def stylize(content_image_path, model_path, num_chunks, num_hypervecs, hypervec_dim, ray, imsize=256):
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    #Load and preprocess image
    content_image = image_loader(content_image_path, imsize, device)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        model = StyleTransferNet()
        hnet = Hyper(num_chunks=num_chunks, num_hypervecs=num_hypervecs, hypervec_dim=hypervec_dim)
        load_modules_weights(model_path, model, hnet)
        hnet = hnet.to(device)
        model = model.to(device)

        ray = torch.Tensor(ray)
        weights = hnet(ray)
        output = model(content_image, weights).cpu()

        #Save generated IMAGE
        image_name = Path(content_image_path).stem + "-stylized" + ".png"
        save_image(output[0], image_name)


@torch.no_grad()
def evaluate(hypernet, targetnet, eval_image):
    if eval_image:
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        hypernet.eval()
        content_image = image_loader(eval_image, 256, device)

        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        rays = torch.Tensor([0.05,0.95],[0.5,0.5],[0.95,0.05]).to(device)
        for i,ray in enumerate(rays):
            hypernet.zero_grad()
            weights = hypernet(ray)
            g_image = targetnet(content_image, weights).cpu()

            image_name = Path(f"{eval_image}{i}-stylized.png")
            save_image(g_image[0], image_name)


def train_network(dataset, style_image_path, trained_models_output_path, batch_size, epochs, lr, content_weight, style_weight, dirichlet_alpha, hypervec_dim, num_hypervecs, chunks, eval_image):
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    #Load VGG-16
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()
    vgg =  vgg.to(device)

    #Initialize Hypernetwork
    hypernet: nn.Module = Hyper(num_chunks=chunks, num_hypervecs=num_hypervecs, hypervec_dim=hypervec_dim)
    hypernet = hypernet.to(device)

    #Initialize Style Network
    style_network = StyleTransferNet()
    style_network.to(device)

    #Initialize MOO Solver
    solver = LinearScalarizationSolver()

    #Preprocess style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style_image = image_loader(style_image_path, size=256)
    style_image = style_transform(style_image)
    style_image = style_image.to(device)
    style_image = normalize_batch(style_image)
    style_image = style_image.repeat(batch_size, 1, 1, 1).to(device)


    #Load Dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Initialize Optimizer
    optimizer = torch.optim.Adam(list(hypernet.parameters()) + list(style_network.parameters()), lr)
    style_sum = 0
    content_sum = 0
    losses_save = []

    #Train Network
    for epoch in range(epochs):
        total_loss_sum = 0
        torch.cuda.empty_cache()

        for batch_num, (content_batch, _) in enumerate(train_loader):
            optimizer.zero_grad()

            #Sample ray from dirchlet disturbution
            ray = torch.from_numpy(
                np.random.dirichlet((dirichlet_alpha, dirichlet_alpha), 1).astype(np.float32).flatten()
            ).to(device)

            #Normalize batch
            content_batch = content_batch.to(device)
            content_batch = normalize_batch(content_batch)

            #Produce weights for style transformation network
            weights = hypernet(ray)

            #Load weights and perform style transfer on current batch
            generated_images_batch = style_network(content_batch, weights)
            generated_images_batch = normalize_batch(generated_images_batch)

            #Calculate Loss
            content_loss, style_loss = total_loss_get(vgg, content_batch, style_image, generated_images_batch)
            content_loss = content_weight * content_loss
            style_loss = style_weight * style_loss

            total_loss =  content_weight * content_loss + style_weight * style_loss 
            style_sum += style_weight * style_loss
            content_sum += content_weight * content_loss
            total_loss_sum += total_loss.item()

            losses = torch.stack((content_loss, style_loss))
            ray = ray.squeeze(0)
            loss = solver(losses, ray)

            #Update network parameters
            loss.backward()
            optimizer.step()

            print(f"Total Weighted Loss: {loss.item():.6f}, Content Loss: {content_loss.item():.6f}, Style Loss: {style_loss.item():.6f}")

            if(batch_num % 100 == 0):
                print(f"Batch: #{batch_num} --- Content Loss: {content_sum}, Style Loss: {style_sum}, Total Loss: {total_loss_sum}")
                myloss = f"total loss:{total_loss_sum} style loss:{style_sum} content loss:{content_sum}"
                losses_save.append(myloss)
                total_loss_sum = 0
                style_sum = 0
                content_sum = 0

            if(batch_num % 100 == 0):
              evaluate(hypernet, style_network, eval_image)

            if(batch_num % 1000 == 0):
              model_name = f"{trained_models_output_path}/model_epoch{epoch}_batch{batch_num}.pth"
              print(f"Checkpoint - Saving Model: {model_name}")
              save_modules_weights(model_name, style_network, hypernet)