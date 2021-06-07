import os
import torch
import numpy as np
from torchvision import transforms

def augmentor(data, image_size=(256, 256), normalize=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    transform1 = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            normalize
                        ])

    transform2 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(45),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform3 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(90),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform4 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(120),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform5 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(180),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform6 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(270),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform7 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(300),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform8 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(330),
                                transforms.ToTensor(),
                                normalize 
                                ])
        
    transformers = [transform1, transform2, transform3, transform4, transform5, transform6, transform7, transform8]
    
    new_imgs = []
    new_labels = []

    for X, y in data:            
        for trans in transformers:
            new_imgs.append(trans(X))
            new_labels.append(torch.tensor(y))
        
    return torch.stack(new_imgs), torch.stack(new_labels)

def plain_transform(data, image_size=(256, 256), normalize=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    transformers = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            normalize
                        ])

    new_imgs = []
    new_labels = []

    for X, y in data:
        new_imgs.append(transformers(X))
        new_labels.append(torch.tensor(y))
    
    return torch.stack(new_imgs), torch.stack(new_labels)

def augmentor_xception(data, image_size=(256, 256), normalize=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])):
    transform1 = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            normalize
                        ])

    transform2 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(45),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform3 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(90),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform4 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(120),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform5 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(180),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform6 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(270),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform7 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(300),
                                transforms.ToTensor(),
                                normalize
                                ])
    
    transform8 = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomRotation(330),
                                transforms.ToTensor(),
                                normalize 
                                ])
        
    transformers = [transform1, transform2, transform3, transform4, transform5, transform6, transform7, transform8]
    
    new_imgs = []
    new_labels = []

    for X, y in data:            
        for trans in transformers:
            new_imgs.append(trans(X))
            new_labels.append(torch.tensor(y))
        
    return torch.stack(new_imgs), torch.stack(new_labels)

def plain_transform_xception(data, image_size=(256, 256), normalize=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])):
    transformers = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            normalize
                        ])

    new_imgs = []
    new_labels = []

    for X, y in data:
        new_imgs.append(transformers(X))
        new_labels.append(torch.tensor(y))
    
    return torch.stack(new_imgs), torch.stack(new_labels)