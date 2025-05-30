import os
import argparse
from torchvision import datasets
import matplotlib.pyplot as plt
import torch,utils
import torch.nn as nn
from dataset import SignDigitDataset
from torch.utils.data import DataLoader
from utils import *
from model import MLP
from torch.utils.tensorboard import SummaryWriter


# default `log_dir` is "runs" - we'll be more specific here
i=1
while os.path.exists(f'runs/sign_digits_experiment_{i}'):
    i+=1


os.makedirs(f'runs/sign_digits_experiment_{i}')
writer = SummaryWriter(f'runs/sign_digits_experiment_{i}')

parser = argparse.ArgumentParser()
# Hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100, required=True, help='number of epochs for training')
parser.add_argument('--print_every', type=int, default=10, help='print the loss every n epochs')
parser.add_argument('--img_size', type=int, default=64, help='image input size')
parser.add_argument('--n_classes', type=int, default=6, help='number of classes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_layers', type=int, required=True, nargs='+',
                    help='number of units per layer (except input and output layer)')
parser.add_argument('--activation', type=str, default=None, choices=['relu', 'tanh','sigmoid'], help='activation layers')
parser.add_argument('--init', type=str, default=None, choices=['zero_constant', 'uniform',], help='weight initialize')
parser.add_argument('--dropout', type=bool, default=False, choices=[True,False,], help='data regulization')

args = parser.parse_args()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

#####################################################################################
# 0. Create train/test datasets                                                     #
# 1. Create train and test data loaders with respect to some hyper-parameters       #
# 2. Get an instance of your MLP model.                                             #
# 3. Define an appropriate loss function (e.g. cross entropy loss)                  #
# 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
# 5. Implement the main loop function with n_epochs iterations which the learning   #
#    and evaluation process occurred there.                                         #
# 6. Save the model weights                                                         #
#####################################################################################


# Initializing Hyper Parameters
num_epochs      = args.n_epochs
hidden_layers   = args.hidden_layers
learning_rate   = args.learning_rate
batch_size      = args.batch_size
activation      = args.activation
n_classes       = args.n_classes
img_size        = args.img_size


# 0. creating train_dataset and test_dataset


train_dataset = SignDigitDataset(root_dir='data/',
                                 h5_name='train_signs.h5',
                                 train=True,
                                 transform=get_transformations(64))

test_dataset = SignDigitDataset(root_dir='data/',
                                h5_name='test_signs.h5',
                                train=False,
                                transform=get_transformations(64))
# 1. creating train loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset
                                           , batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset
                                           , batch_size=64, shuffle=False)




# 2. get an instance of the model

units=[]
units.append((img_size**2)*3)
for i in hidden_layers:
    units.append(i)
units.append(n_classes)

mlp = MLP(units, hidden_layer_activation=activation, init_type=args.init , dropout=args.dropout)

print(mlp)

# 3, 4. loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(),lr=learning_rate)


# 5. Train the model
n_train_batches = 0
n_test_batches = 0


losses=[]
epoches=[]
losses2=[]
epoches2=[]

for epoch in range(args.n_epochs):
    train_running_loss, test_running_loss = 0.0, 0.0

    for i ,(images,labels) in enumerate(train_loader):
        images    = images.reshape(-1, 3*(img_size**2)).to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        outputs   = mlp(images)
        loss      = criterion(outputs,labels)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train_batches = i+1

    # log test average loss
    writer.add_scalar('Train Loss', train_running_loss / n_train_batches, epoch)

    # print loss for debugging
    print("Last Train Loss of an Epoch [{}/{}]:Loss {:.4f}".format(epoch+1,num_epochs, loss.item() ))
    print("Mean Train loss of an Epoch [{}/{}]:Loss {:.4f}".format(epoch + 1, num_epochs, train_running_loss / n_train_batches))
    losses.append(loss.item())
    epoches.append(epoch)

    with torch.no_grad():
        correct=0
        total  =0
        for i,(images,labels) in enumerate(test_loader):
            images   = images.reshape(-1, 3*(img_size**2)).to(device)
            labels   = labels.to(device)
            outputs  = mlp(images)
            _, predicted= torch.max(outputs.data,1)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            total   += labels.size(0)
            correct += (predicted==labels).sum().item()
            n_test_batches = i+1

        print('Accuracy on the test set : {}'.format(100*correct/total))
        losses2.append(loss.item())
        epoches2.append(epoch)

    # print loss for debugging
    print("Last Test Loss of an Epoch [{}/{}]:Loss {:.4f}".format(epoch + 1, num_epochs, loss.item()))
    print("Mean Test loss of an Epoch [{}/{}]:Loss {:.4f}".format(epoch + 1, num_epochs, test_running_loss / n_test_batches))


    # log the running loss
    writer.add_scalar('Test Loss', test_running_loss / n_test_batches, epoch)
    writer.add_scalar('Accuracy', 100*correct/total, epoch)


    if epoch % args.print_every == 0:
        # You have to log the accuracy as well
        print('Epoch [{}/{}]:\t Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, args.n_epochs,
                                                                                train_running_loss / n_train_batches,
                                                                                test_running_loss / n_test_batches))




checkpoint_dir = 'checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# save the model weights
torch.save(mlp.state_dict(),checkpoint_dir+"model.pth")

plt.plot(epoches,losses,)
plt.plot(epoches2,losses2,)
plt.legend(['Train','Valid'])
plt.show()