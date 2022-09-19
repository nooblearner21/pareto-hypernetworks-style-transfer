import argparse
from torchvision import datasets, transforms
from style_transfer import train_network, stylize


LR = 0.0005
STYLE_WEIGHT = 100000
CONTENT_WEIGHT = 1
EPOCHS = 100
BATCH_SIZE = 4


def main():
    arg_parser = argparse.ArgumentParser(description="User arguments for style transfer")
    arg_parser.add_argument("--train", action="store_true", help="Train a new model")
    arg_parser.add_argument("--stylize", action="store_true", help="Use an existing model to perform style transfer on an image")
    arg_parser.add_argument("--image", type=str, required=True,
                            help="In train mode then this is the style image that the model will be trained on. In stylize mode this is the image to perform style transfer on")
    arg_parser.add_argument("--trained-models-output-path", type=str, default='./')
    arg_parser.add_argument("--stylize-model-path", type=str, help="In stylize mode this is the path of the model that performs the style transfer")
    arg_parser.add_argument("--lr", type=float, default=LR, help="Learning rate of the network")
    arg_parser.add_argument("--style-weight", type=float, default=STYLE_WEIGHT, help="Style weight - more means larger emphasis on style")
    arg_parser.add_argument("--content-weight", type=float, default=CONTENT_WEIGHT, help="Content Weight - more means larger conservation of the original content")
    arg_parser.add_argument("--epochs", type=float, default=EPOCHS, help="Number of epochs to run on the training set")
    arg_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Size of batch in each iteration during training")
    arg_parser.add_argument("--train-data-path", type=str, help="Path of training data")

    args = arg_parser.parse_args()

    if(args.train):
        if not args.train_data_path:
            raise Exception('Missing --train-data-path flag!')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        dataset = datasets.ImageFolder(args.train_data_path, transform=transform)

        train_network(
            dataset=dataset, 
            style_image_path=args.image, 
            trained_models_output_path=args.trained_models_output_path, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr, 
            content_weight=args.content_weight, 
            style_weight=args.style_weight
        )

    elif(args.stylize):
        stylize(content_image_path=args.image, model_path=args.stylize_model_path, imsize=256)

    else:
        raise Exception('Program must be given either --train or --stylize flags!')


if __name__ == "__main__":
    main()
