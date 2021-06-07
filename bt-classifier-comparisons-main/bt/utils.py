import os
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from tqdm import tqdm
import scikitplot as skplt
from math import sqrt, ceil
from scipy.misc import imsave
import torchvision.utils as vutils
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
from torch.nn.functional import interpolate
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score

def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensores
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def savetensor(tensor, filename, ch=0, allkernels=False, nrow=8, padding=2):
    '''
    savetensor: save tensor
        @filename: file name
        @ch: visualization channel 
        @allkernels: visualization all tensores
    '''    

    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)    
    vutils.save_image(tensor, filename, nrow=nrow )

def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.title('Loss Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def plot_accuracy(train_correct, test_correct):
    plt.plot(train_correct, label='Training accuracy')
    plt.plot(test_correct, label='Validation accuracy')
    plt.title('Accuracy Metrics')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_pred, y_true, labels=['Meningioma', 'Glioma', 'Pitutary']):
    arr = confusion_matrix(y_pred.view(-1).cpu(), y_true.view(-1).cpu())
    df_cm = pd.DataFrame(arr,labels, labels)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.show()

def get_all_metrics(y_pred, y_test, print_results=True):
    cr = classification_report(y_pred.view(-1).cpu(), y_test.view(-1).cpu())
    p = precision_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    r = recall_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    f = f1_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    a = accuracy_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu())

    # display the results if selected
    print(f'\nAccuracy Score: {a:.4f}\n')
    print(f"Classification Report: \n\n{cr}\n")
    print(f'Precision Score (Class-Wise): \n{p}\nAverage Precision Score: {np.mean(p)}\n')
    print(f'Recall Score (Class-Wise): \n{r}\nAverage Recall Score: {np.mean(r)}\n')
    print(f'F1 Score (Class-Wise): \n{f}\nAverage F1: {np.mean(f)}\n')
    
    return a, p, np.mean(p), r, np.mean(r), f, np.mean(f)

def train(train_loader, dev_loader, model, epochs, batch_size, criterion, optimizer, device, model_path, model_name, n_classes=3):
    print('\nTraining the model')

    b, test_b = 0, 0

    training_losses = []
    training_accuracies = []
    test_losses = []
    test_correct = []

    start_time = time.time()

    for epoch in range(epochs):
        e_start = time.time()

        model.train()

        running_loss = 0.0
        running_accuracy = 0.0
        tst_corr = 0.0

        for b, (X_train, y_train) in enumerate(train_loader):
            
            X_train, y_train = X_train.to(device), y_train.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = model(X_train).view(-1, n_classes)

            loss = criterion(y_pred, y_train)

            predicted = torch.argmax(y_pred.data, dim=1).data
            batch_corr = (predicted == y_train).sum()
            running_accuracy += batch_corr
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if b % int(len(train_loader)/batch_size) == 0:
                print(f'Epoch: {epoch+1:2}  batch: {b+1:6} [{b+1:6}/{len(train_loader)}]  Loss: {loss.item():.6f}  Accuracy: {running_accuracy.item()*100/(batch_size * (b+1)):.6f}%')
            
        training_losses.append(loss.item())
        training_accuracies.append(running_accuracy.item()*100/(batch_size * (b+1)))

        print(f"Epoch {epoch+1} | Training Accuracy: {training_accuracies[-1]:.6f} | Training Loss: {training_losses[-1]:.6f}")

        with torch.no_grad():
            for test_b, (X_test, y_test) in enumerate(dev_loader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                
                y_val = model(X_test).view(-1, n_classes)

                predicted = torch.argmax(y_val.data, dim=1)
                tst_corr += (predicted == y_test).sum()
                
        loss = criterion(y_val, y_test)
        test_losses.append(loss.item())
        test_correct.append(tst_corr.item() * 100/(batch_size*(test_b+1)))
        
        e_end = time.time() - e_start

        print(f"Validation Accuracy: {test_correct[-1]:.6f} | Validation Loss: {test_losses[-1]:.6f}")
        print(f"Duration: {e_end/60:.4f} minutes")

    print('Finished Training')

    end_time = time.time() - start_time    

    # print training summary
    print("\nTraining Duration {:.2f} minutes".format(end_time/60))
    print("GPU memory used : {} kb".format(torch.cuda.memory_allocated()))
    print("GPU memory cached : {} kb".format(torch.cuda.memory_reserved()))

    plot_loss(training_losses, test_losses) 
    plot_accuracy(training_accuracies, test_correct) 

    torch.save(model.state_dict(), model_path + '/' + model_name + '_final.pt')

def test(test_loader, model, batch_size, criterion, device, n_classes=3, label_names=None):
    model.eval()
    b = 0

    with torch.no_grad():
        correct = 0
        test_loss = []
        test_corr = []
        labels = []
        pred = []

        new_y = 0.0

        # perform test set evaluation batch wise
        for b, (X, y) in enumerate(test_loader):
            b += 1
            # set label to use CUDA if available
            X, y = X.to(device), y.to(device)

            # append original labels            
            if y.shape[0] != batch_size:
                blank_values = [torch.Tensor([0.0]) for i in range((batch_size - y.shape[0]))]
                new_y = torch.Tensor([*y, *blank_values])
                labels.append(new_y.cpu().numpy())
            else:            
                labels.append(y.cpu().numpy())

            # perform forward pass
            y_val = model(X).view(-1, n_classes)

            # get argmax of predicted values, which is our label
            predicted = torch.argmax(y_val.data, dim=1) 

            if predicted.shape[0] != batch_size:
                blank_values = [torch.Tensor([0.0]) for i in range((batch_size - predicted.shape[0]))]
                predicted = torch.Tensor([*predicted, *blank_values])

            # append predicted label
            pred.append(predicted.cpu().numpy())

            # calculate loss
            loss = criterion(y_val, y)

            # increment correct with correcly predicted labels per batch
            if y.shape[0] != batch_size:
                correct += (predicted == new_y).sum()
            else:
                correct += (predicted == y).sum()

            # append correct samples labels and losses
            test_corr.append(correct.item()*100/(b*batch_size))
            test_loss.append(loss.item())
            
    print(f"Test Loss: {test_loss[-1]:.4f}")
    print(f'Test accuracy: {test_corr[-1]:.2f}%')

    labels = torch.Tensor(labels)
    pred = torch.Tensor(pred)

    if label_names:
        plot_confusion_matrix(pred, labels, label_names)
    else:
        plot_confusion_matrix(pred, labels)
    
    get_all_metrics(pred, labels)
