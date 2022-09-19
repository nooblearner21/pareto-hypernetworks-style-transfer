import torch
from PIL import Image
import numpy as np


def image_loader(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


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

def save_modules_weights(output_path, target_network, hyper_network):
    in1 = target_network.in1
    in2 = target_network.in2
    in3 = target_network.in3

    res1_norm1 = target_network.res1.norm1
    res1_norm2 = target_network.res1.norm2
    res2_norm1 = target_network.res2.norm1
    res2_norm2 = target_network.res2.norm2
    res3_norm1 = target_network.res3.norm1
    res3_norm2 = target_network.res3.norm2
    res4_norm1 = target_network.res4.norm1
    res4_norm2 = target_network.res4.norm2
    res5_norm1 = target_network.res5.norm1
    res5_norm2 = target_network.res5.norm2

    in4 = target_network.in4
    in5 = target_network.in5

    torch.save({
        'hyper_network': hyper_network.state_dict(),
        'in1': in1.state_dict(),
        'in2': in2.state_dict(),
        'in3': in3.state_dict(),
        'res1_norm1': res1_norm1.state_dict(),
        'res1_norm2': res1_norm2.state_dict(),
        'res2_norm1': res2_norm1.state_dict(),
        'res2_norm2': res2_norm2.state_dict(),
        'res3_norm1': res3_norm1.state_dict(),
        'res3_norm2': res3_norm2.state_dict(),
        'res4_norm1': res4_norm1.state_dict(),
        'res4_norm2': res4_norm2.state_dict(),
        'res5_norm1': res5_norm1.state_dict(),
        'res5_norm2': res5_norm2.state_dict(),
        'in4': in4.state_dict(),
        'in5': in5.state_dict()
    }, output_path)

def load_modules_weights(model_path, target_network, hyper_network):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    hyper_network.load_state_dict(checkpoint['hyper_network'])

    target_network.in1.load_state_dict(checkpoint['in1'])
    target_network.in2.load_state_dict(checkpoint['in2'])
    target_network.in3.load_state_dict(checkpoint['in3'])

    target_network.res1.norm1.load_state_dict(checkpoint['res1_norm1'])
    target_network.res1.norm2.load_state_dict(checkpoint['res1_norm2'])
    target_network.res2.norm1.load_state_dict(checkpoint['res2_norm1'])
    target_network.res2.norm2.load_state_dict(checkpoint['res2_norm2'])
    target_network.res3.norm1.load_state_dict(checkpoint['res3_norm1'])
    target_network.res3.norm2.load_state_dict(checkpoint['res3_norm2'])
    target_network.res4.norm1.load_state_dict(checkpoint['res4_norm1'])
    target_network.res4.norm2.load_state_dict(checkpoint['res4_norm2'])
    target_network.res5.norm1.load_state_dict(checkpoint['res5_norm1'])
    target_network.res5.norm2.load_state_dict(checkpoint['res5_norm2'])

    target_network.in4.load_state_dict(checkpoint['in4'])
    target_network.in5.load_state_dict(checkpoint['in5'])


def hypervolume_score(solutions_set, ref_point):

  #Convert to lists to ndarray
  if(type(solutions_set) is list):
    solutions_set = np.array(solutions_set)

  #Sort solutions by x axis (first column)
  solutions_set = solutions_set[solutions_set[:, 0].argsort()]
  solutions_set = eliminate_dominated_solutions(solutions_set)

  n = solutions_set.shape[0]

  score = 0

  for i in range(n-1):
    current_solution = solutions_set[i]
    next_solution = solutions_set[i + 1]

    height = ref_point[1] - current_solution[1]
    width = next_solution[0] - current_solution[0]

    score += (height * width)

  last_solution = solutions_set[n - 1]
  height = ref_point[1] - last_solution[1]
  width = ref_point[0] - last_solution[0]

  score += (height * width)

  return score / (ref_point[0] * ref_point[1])

def eliminate_dominated_solutions(solutions_set):
  print(solutions_set)
  n = len(solutions_set)
  if (n == 0 or n == 1):
    return solutions_set

  result = [solutions_set[0]]
  i = 0

  while(i < n - 1):
    current_solution = solutions_set[i]
    next_solution = solutions_set[i + 1]

    #Set is sorted by X axis, so if Y axis is also bigger, then it's a dominated point and should be removed (not be appended to the result set)
    if current_solution[1] > next_solution[1]:
      result.append(next_solution)

    i += 1

  return np.array(result)