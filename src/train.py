import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

global breeds_list
breeds_list = os.listdir('cat-breeds-dataset/images')

def model_exists():
    return os.path.exists('model.pt')

def create_model():
    model_resnet = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
    for param in model_resnet.parameters():
        param.requires_grad = False

    

    model_resnet.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 62, bias=True),
        nn.LogSoftmax(dim=1)
    )
    model_resnet = model_resnet.cuda()
    if model_exists(): model_resnet.load_state_dict(torch.load('model.pt'))

    # model_vgg = models.vgg16(weights=models.vgg.VGG16_Weights.DEFAULT)
    # for param in model_vgg.parameters():
    #     param.requires_grad = False
    
    # model_vgg.classifier[6] = nn.Sequential(
    #     nn.Linear(4096, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(512, 62),
    #     nn.LogSoftmax(dim=1)
    # )
    # model_vgg.classifier[6].requires_grad = True

    # model_vgg = model_vgg.cuda()

    return model_resnet

def create_folders():
    import os
    import shutil
    import random

    validate = 0.15
    test = 0.05

    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
    if not os.path.exists('data/validate'):
        os.makedirs('data/validate')
    if not os.path.exists('data/test'):
        os.makedirs('data/test')
    for breed in tqdm(breeds_list):
        os.mkdir(f'data/train/{breed}')
        os.mkdir(f'data/validate/{breed}')
        os.mkdir(f'data/test/{breed}')
    
        breed_data = os.listdir(f'cat-breeds-dataset/images/{breed}/')
        breed_train, \
        breed_validate, \
        breed_test = \
            np.split(np.array(breed_data), \
                    [int(len(breed_data)* (1 - validate + test)), \
                     int(len(breed_data)* (1 - test))])
        
        for img in breed_train:
            shutil.copy(f'cat-breeds-dataset/images/{breed}/{img}', f'data/train/{breed}/{img}')
        for img in breed_validate:
            shutil.copy(f'cat-breeds-dataset/images/{breed}/{img}', f'data/validate/{breed}/{img}')
        for img in breed_test:
            shutil.copy(f'cat-breeds-dataset/images/{breed}/{img}', f'data/test/{breed}/{img}')

def create_loaders():
    data_dir = 'data/'
    train_dir = data_dir + 'train/'
    validate_dir = data_dir + 'validate/'
    test_dir = data_dir + 'test/'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'validate': datasets.ImageFolder(validate_dir, transform=data_transforms['validate']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'validate': torch.utils.data.DataLoader(image_datasets['validate'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }

    return dataloaders

def train_model(model, loaders, epochs=10):
    validate_loss_min = np.Inf
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = 0.0
        validate_loss = 0.0

        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['validate']):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)

            validate_loss = validate_loss + ((1 / (batch_idx + 1)) * (loss.data - validate_loss))

        if validate_loss <= validate_loss_min:
            torch.save(model.state_dict(), 'model.pt')
            validate_loss_min = validate_loss
    return model

def test_model(model, loaders):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    class_correct = list(0. for i in range(37))
    class_total = list(0. for i in range(37))

    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        for i in range(len(target.data)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

if __name__ == '__main__':
    if not os.path.exists('data'):
        create_folders()
    loaders = create_loaders()
    model = create_model()
    model = train_model(model, loaders, epochs=30)
    test_model(model, loaders)
