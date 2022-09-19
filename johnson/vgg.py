def get_content_features(model, batch, layers=None):
    if layers is None:
        layers = {
            '8': 'relu2_2',
        }

    features = []
    x = batch

    for index, layer in model._modules.items():
        x = layer(x)
        if index in layers:
            features.append(x)

    return features

def get_style_features(model, batch, layers=None):
    if layers is None:
        layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

    features = []
    x = batch
    for index, layer in model._modules.items():
        x = layer(x)
        if index in layers:
            features.append(x)

    return features