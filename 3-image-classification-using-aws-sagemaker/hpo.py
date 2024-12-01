#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, 
            len(test_loader.dataset), 
            100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, args.epochs + 1):
        model.train()
        logger.info("Traning ...")
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    # Freeze CNN layer
    for param in model.parameters():
        param = param.requires_grad_(False)
        
    # Custom fully Connected layers
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 133),
    )
    
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    valid_path = os.path.join(data, 'valid')

    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    train = torchvision.datasets.ImageFolder(train_path, transform=transform)
    test = torchvision.datasets.ImageFolder(valid_path, transform=transform)
    valid = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)

    return train_loader, test_loader, valid_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device infor: {device}")
    model = net()
    model.to(device)
    logger.info(
        f"Argument: batch size {args.batch_size} | epochs {args.epochs} | lr {args.lr} | model dir {args.model_dir} | data dir {args.data_dir}"
    )
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    train(model, train_loader, loss_criterion, optimizer)

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing ...")
    test(model, test_loader, loss_criterion)

    '''
    TODO: Save the trained model
    '''
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        metavar="LR", 
        help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.environ["SM_CHANNEL_DATA"]
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    args=parser.parse_args()

    main(args)
