"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 22, 2024
"""
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import gc
from GPUtil import showUtilization as gpu_usage

def train_segmentation_model_pt(pthDL):

    # Load variables from pickle file
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        sxy, classNames, nm = data['sxy'], data['classNames'], data['nm']

    if 'net' in data:
        raise ValueError(f"A network has already been trained for model {nm}. Choose a new model name to retrain.")
    else:
        # rest of the code here
        pass  # --delete pass when the rest of the code is placed below the else statement

    # Paths to training and validation datasets:
    classes = list(range(1, len(classNames)))

    nmim = 'im'
    nmlabel = 'label'

    pthTrain = os.path.join(pthDL, 'training')
    pthVal = os.path.join(pthDL, 'validation')

    Train_HE = os.path.join(pthTrain, nmim)
    Train_label = os.path.join(pthTrain, nmlabel)

    Validation_HE = os.path.join(pthVal, nmim)
    Validation_label = os.path.join(pthVal, nmlabel)

    # Model input data blueprint
    class CustomDataset(Dataset):
        def __init__(self, image_dir, label_dir, transform=None, label_transform=None):
            self.image_dir = image_dir
            self.label_dir = label_dir
            self.transform = transform
            self.label_transform = label_transform
            self.images = os.listdir(image_dir)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = os.path.join(self.image_dir, self.images[idx])
            label_name = os.path.join(self.label_dir, self.images[idx])

            image = Image.open(img_name).convert("RGB")  # Convert image to RGB
            label = Image.open(label_name).convert("L")  # Convert label to grayscale

            if self.transform:
                image = self.transform(image)
            if self.label_transform:
                label = self.label_transform(label)
                label = label.squeeze(0).long()  # Remove the channel dimension and convert to LongTensor

            return image, label

    # Define data transformations
    data_transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # Resize to the model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # normalization according to imageNet stats
    ])

    label_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Make datasets
    train_dataset = CustomDataset(image_dir=Train_HE, label_dir=Train_label, transform=data_transform,
                                  label_transform=label_transform)
    val_dataset = CustomDataset(image_dir=Validation_HE, label_dir=Validation_label, transform=data_transform,
                                label_transform=label_transform)

    # Create data loaders
    batch_size = 4  # datapoints per 'mini-batch' - ideally a small power of 2 (32, 64, 128, or 256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(type(train_loader))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #Fucntion to free up GPU
    def free_gpu_cache():
        print("Initial GPU Usage")
        gpu_usage()

        # Check allocated memory before clearing cache
        print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated()} bytes")

        # Delete all tensors from GPU memory
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except:
                pass

        gc.collect()  # Run garbage collection
        torch.cuda.empty_cache()  # Empty the CUDA cache
        torch.cuda.ipc_collect()  # Collect any unused memory

        # Reset max memory allocated
        torch.cuda.reset_max_memory_allocated()

        # Check allocated memory after clearing cache
        print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated()} bytes")

        print("GPU Usage after emptying the cache")
        gpu_usage()



    # _______________________Model Initialization________________________#

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True) # old
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights='DEFAULT')

    # Learning rate - controls how much the network updates its weights in each minibatch
    initial_lr = 0.0005

    # Optimizer - algorithm that updates the model's weights based on the loss
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Loss function - measures the difference between predictions and actual labels
    criterion = nn.CrossEntropyLoss()

    # ___________________________Model Training______________________________#

    # Free GPU Cache
    if torch.cuda.is_available():
        print(f'GPU device: {torch.cuda.get_device_name()}')
        free_gpu_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(type(device))
    model.to(device)

    #### Training Hyperparameters

    num_epochs = 8  # Number of epochs - maximum number of times to iterate through the entire dataset

    validation_patience = 7  # Validation patience - number of epochs with no improvement to stop training early

    lr_drop_factor = 0.75  # Learning rate drop factor - how much to reduce learning rate after each drop period

    lr_drop_period = 1  # Learning rate drop factor - how much to reduce learning rate after each drop period

    validation_frequency = 10  # Validation frequency - how often (in batches) to print validation loss

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    #### Turn on interactive mode for live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')


    print("Let's train bby!")
    import time
    start_training_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Record the start time of the epoch
        start_time_epoch = time.time()

        for i, (inputs, labels) in enumerate(train_loader):

            # move the input and model to GPU for speed if available

            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)['out']

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print training performance after each iteration
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Plot live graph and adjust learning rate
            if i % validation_frequency == validation_frequency - 1:
                print('Plotting training....')
                avg_loss = running_loss / validation_frequency
                print(f"Average Loss: {avg_loss:.4f}")
                train_losses.append(avg_loss)
                ax1.plot(train_losses, 'b-o', label='Training Loss')
                ax1.set_xticks(range(0, len(train_losses), max(1, len(train_losses) // 10)))
                ax1.set_xticklabels(range(1, len(train_losses) + 1, max(1, len(train_losses) // 10)))
                ax1.set_xlim([0, len(train_losses)])
                ax1.set_ylim([0, max(train_losses) * 1.1])
                ax1.legend()
                fig.canvas.draw()
                running_loss = 0.0

                # Validate the model
                if (i + 1) % (validation_frequency * validation_patience) == 0:
                    model.eval()
                    correct = 0
                    total = 0
                    val_loss = 0.0
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                            val_outputs = model(val_inputs)['out']
                            val_loss += criterion(val_outputs, val_labels).item()

                            # Calculate accuracy
                            _, predicted = torch.max(val_outputs, 1)
                            total += val_labels.numel()
                            correct += (predicted == val_labels).sum().item()

                    val_loss /= len(val_loader)
                    val_accuracy = 100 * correct / total
                    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    ax1.plot(val_losses, 'g-o', label='Validation Loss')
                    ax1.legend()
                    ax2.plot(val_accuracies, 'r-o', label='Validation Accuracy')
                    ax2.set_xticks(range(0, len(val_accuracies), max(1, len(val_accuracies) // 10)))
                    ax2.set_xticklabels(range(1, len(val_accuracies) + 1, max(1, len(val_accuracies) // 10)))
                    ax2.set_xlim([0, len(val_accuracies)])
                    ax2.set_ylim([0, 100])
                    ax2.legend()
                    fig.canvas.draw()

                    model.train()

                    # Adjust learning rate
                    scheduler.step()
                    print(f"Learning rate adjusted to: {scheduler.get_last_lr()[0]}")

        # Calculate and print the time taken for the epoch
        epoch_time = time.time() - start_time_epoch
        hours_epoch, rem_epoch = divmod(epoch_time, 3600)
        minutes_epoch, seconds_epoch = divmod(rem_epoch, 60)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {hours_epoch}h {minutes_epoch}m {seconds_epoch}s.")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        train_losses.append(epoch_loss)

        x_axis = list(range(1, len(train_losses) + 1))
        ax1.plot(x_axis, train_losses, 'bo-')
        ax1.set_xticks(x_axis)
        ax1.set_xlim([1, len(train_losses)])
        ax1.set_ylim([0, max(train_losses + val_losses) * 1.1])
        fig.canvas.draw()

    plt.ioff()
    plt.show()

    end_training_time = time.time() - start_training_time
    hours, rem = divmod(end_training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print('Finished Training')
    print(f"Training time: {hours}h {minutes}m {seconds}s")



if __name__ == '__main__':
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\model test tiles'
    train_segmentation_model_pt(pthDL)
