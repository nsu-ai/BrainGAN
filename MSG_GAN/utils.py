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
    # calculate classification report
    cr = classification_report(y_pred.view(-1).cpu(), y_test.view(-1).cpu())
    # calculate jaccard similarity score
    js = jaccard_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    # calculate other metrics
    p = precision_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    r = recall_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    f = f1_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu(), average=None)
    a = accuracy_score(y_pred.view(-1).cpu(), y_test.view(-1).cpu())

    # display the results if selected
    if print_results:
        print(f'\nAccuracy Score: {a:.4f}\n')
        print(f"Classification Report: \n\n{cr}\n")
        print(f"Jaccard Index (Class-Wise): \n{js}\nAverage Jaccard Index: {np.mean(js)}\n")
        # print(f'ROC-AUC Score: {roc_auc:.5f}\n')
        print(f'Precision Score (Class-Wise): \n{p}\nAverage Precision Score: {np.mean(p)}\n')
        print(f'Recall Score (Class-Wise): \n{r}\nAverage Recall Score: {np.mean(r)}\n')
        print(f'F1 Score (Class-Wise): \n{f}\nAverage F1: {np.mean(f)}\n')
    
    # return the calculated values
    else:
        return cr, js, p, r, f, a

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

        # Run the testing batches
        with torch.no_grad():
            for test_b, (X_test, y_test) in enumerate(dev_loader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                # Apply the model
                y_val = model(X_test).view(-1, n_classes)

                # Tally the number of correct predictions
                predicted = torch.argmax(y_val.data, dim=1)
                tst_corr += (predicted == y_test).sum()
                
        loss = criterion(y_val, y_test)
        test_losses.append(loss.item())
        test_correct.append(tst_corr.item() * 100/(batch_size*(test_b+1)))
        
        e_end = time.time() - e_start

        print(f"Validation Accuracy: {test_correct[-1]:.6f} | Validation Loss: {test_losses[-1]:.6f}")
        print(f"Duration: {e_end/60:.4f} minutes")

        # torch.save(model.state_dict(), model_path + '/' + model_name + f'_{epoch+1}.pt')

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

def read_loss_log(file_name, delimiter='\t'):
    """
    read and load the loss values from a loss.log file
    :param file_name: path of the loss.log file
    :param delimiter: delimiter used to delimit the two columns
    :return: loss_val => numpy array [Iterations x 2]
    """
    from numpy import genfromtxt
    losses = genfromtxt(file_name, delimiter=delimiter)
    return losses

def plot_loss_fig(*loss_vals, plot_name="Loss plot",
              fig_size=(17, 7), save_path=None,
              legends=("discriminator", "generator")):
    """
    plot the discriminator loss values and save the plot if required
    :param loss_vals: (Variable Arg) numpy array or Sequence like for plotting values
    :param plot_name: Name of the plot
    :param fig_size: size of the generated figure (column_width, row_width)
    :param save_path: path to save the figure
    :param legends: list containing labels for loss plots' legends
                    len(legends) == len(loss_vals)
    :return:
    """
    assert len(loss_vals) == len(legends), "Not enough labels for legends"

    plt.figure(figsize=fig_size).suptitle(plot_name)
    plt.grid(True, which="both")
    plt.ylabel("loss value")
    plt.xlabel("spaced iterations")

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    # plot all the provided loss values in a single plot
    plts = []
    for loss_val in loss_vals:
        plts.append(plt.plot(loss_val)[0])

    plt.legend(plts, legends, loc="upper right", fontsize=16)

    if save_path is not None:
        plt.savefig(save_path)

def generate_loss_graph(loss_file: str, plot_file: str):
    assert loss_file is not None, "Loss-Log file not specified"

    # read the loss file
    loss_vals = read_loss_log(loss_file)

    # plot the loss:
    plot_loss_fig(loss_vals[:, 1], loss_vals[:, 2], save_path=plot_file)

    print("Loss plots have been successfully generated ...")
    print("Please check: ", plot_file)

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)

def progressive_upscaling(images):
    """
    upsamples all images to the highest size ones
    :param images: list of images with progressively growing resolutions
    :return: images => images upscaled to same size
    """
    with torch.no_grad():
        for factor in range(1, len(images)):
            images[len(images) - 1 - factor] = interpolate(
                images[len(images) - 1 - factor],
                scale_factor=pow(2, factor)
            )

    return images

def get_image(gen, point):
    """
    obtain an All-resolution grid of images from the given point
    :param gen: the generator object
    :param point: random latent point for generation
    :return: img => generated image
    """
    images = list(map(lambda x: x.detach(), gen(point)))[1:]
    images = [adjust_dynamic_range(image) for image in images]
    images = progressive_upscaling(images)
    images = list(map(lambda x: x.squeeze(dim=0), images))
    image = make_grid(
        images,
        nrow=int(ceil(sqrt(len(images))))
    )
    return image.cpu().numpy().transpose(1, 2, 0)

def interpolate_latent_space(gen=None, depth=7, latent_size=512, time=30, fps=30, smoothing=2.0, device=torch.device('cuda'), generator_file='models/msggan/geenrator_final.pt', out_dir='generated_images/msggan'):
    if gen == None:
        from MSG_GAN.GAN import Generator

        # create generator object:
        print("Creating a generator object ...")
        generator = torch.nn.DataParallel(
            Generator(depth=depth,
                    latent_size=latent_size).to(device))

        # load the trained generator weights
        print("loading the trained generator weights ...")
        generator.load_state_dict(torch.load(generator_file))

    # total_frames in the video:
    total_frames = int(time * fps)

    # Let's create the animation video from the latent space interpolation
    # all latent vectors:
    all_latents = torch.randn(total_frames, latent_size).to(device)
    all_latents = gaussian_filter(all_latents.cpu(), [smoothing * fps, 0])
    all_latents = torch.from_numpy(all_latents)
    all_latents = (all_latents / all_latents.norm(dim=-1, keepdim=True)) \
                  * (sqrt(latent_size))

    # create output directory
    os.makedirs(out_dir, exist_ok=True)

    global_frame_counter = 1
    # Run the main loop for the interpolation:
    print("Generating the video frames ...")
    for latent in tqdm(all_latents):
        latent = torch.unsqueeze(latent, dim=0)

        # generate the image for this point:
        img = get_image(generator, latent)

        # save the image:
        plt.imsave(os.path.join(out_dir, str(global_frame_counter) + ".png"), img)

        # increment the counter:
        global_frame_counter += 1

    # video frames have been generated
    print("Video frames have been generated at:", out_dir)

def generate_samples(gen=None, depth=6, latent_size=512, out_depth=6, generator_file='models/msggan/geenrator_final.pt', num_samples=300, out_dir='generated_images/msggan'):
    if gen == None:
        print("Creating generator object ...")
        # create the generator object
        from MSG_GAN.GAN import Generator

        gen = torch.nn.DataParallel(Generator(
            depth=depth,
            latent_size=latent_size
        ))

        print("Loading the generator weights from:", generator_file)
        # load the weights into it
        gen.load_state_dict(
            torch.load(generator_file)
        )

    # path for saving the files:
    save_path = out_dir

    print("Generating scale synchronized images ...")
    for img_num in tqdm(range(1, num_samples + 1)):
        # generate the images:
        with torch.no_grad():
            point = torch.randn(1, latent_size)
            point = (point / point.norm()) * (latent_size ** 0.5)
            ss_images = gen(point)

        # resize the images:
        ss_images = [adjust_dynamic_range(ss_image) for ss_image in ss_images]
        ss_images = progressive_upscaling(ss_images)
        ss_image = ss_images[out_depth]

        # save the ss_image in the directory
        imsave(os.path.join(save_path, str(img_num) + ".png"),
               ss_image.squeeze(0).permute(1, 2, 0).cpu())

    print("Generated %d images at %s" % (num_samples, save_path))