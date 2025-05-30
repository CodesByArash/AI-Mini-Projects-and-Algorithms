from torchvision.transforms import transforms
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_transformations(img_size):
    #####################################################################################
    # Use torchvision.transforms.transforms.Compose to stack several transformations    #
    #####################################################################################
    image_transforms = None

    # my code
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.RandomVerticalFlip(p=0.1),
    ])

    return image_transforms


def get_one_hot(label, num_classes):
    #####################################################################################
    # label --  (int) Categorical labels                                                #
    # num_classes --  (int) Number of different classes that label can take             #
    #####################################################################################
    one_hot_encoded_label   = None


    # my code
    one_hot_digits          = F.one_hot(torch.arange(0, num_classes), num_classes)
    one_hot_encoded_label   = one_hot_digits[label]


    return one_hot_encoded_label


def visualize_samples(dataset, n_samples, cols=4):
    #####################################################################################
    # dataset --  (int) Categorical labels                                              #
    # Your plot must be a grid of images with title of their labels                     #
    # Note: use torch.argmax to decode the one-hot encoded labels back to integer labels#
    # Note: You may need to permute the image of shape (C, H, W) to (H, W, C)           #
    #####################################################################################
    rows = n_samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    figure.tight_layout()
    i=0
    rand=np.random.randint(1, dataset.__len__(), 16)
    for a in ax:
        for x in a:
            x.imshow(dataset[rand[i]][0].permute(1, 2, 0))
            x.set_title(dataset[rand[i]][1])
            x.grid(True)
            i+=1



def init_weights(net: nn.Module, init_type='uniform'):
    #####################################################################################
    # When you get an instance of your nn.Module model later, pass this function        #
    # to torch.nn.Module.apply. For more explanation visit:                             #
    # Note: initialize both weights and biases of the entire model                      #
    #####################################################################################
    valid_initializations = ['zero_constant', 'uniform']
    if isinstance(net , nn.Linear):
        if init_type not in valid_initializations:
            pass
        elif init_type == valid_initializations[0] :
            torch.nn.init.zeros_(net.weight)
            torch.nn.init.zeros_(net.bias)
        elif init_type == valid_initializations[1] :
            torch.nn.init.uniform_(net.weight,a=-1.0, b=1.0)
            torch.nn.init.uniform_(net.bias,a=-1.0, b=1.0)
