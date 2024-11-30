import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from New_network import GeneratorUNet, Discriminator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(G, D, dataloader, optimizer_G, optimizer_D, criterion_GAN, criterion_recon, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G.train()
    D.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer_D.zero_grad()

        # Forward pass
        fake_y = G(image_semantic)
        real = D(image_semantic, image_rgb)
        fake = D(image_semantic, fake_y.detach())
        real_loss = criterion_GAN(real, torch.ones_like(real))
        fake_loss = criterion_GAN(fake, torch.zeros_like(fake))
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        valid = D(image_semantic, fake_y)
        gan_loss = criterion_GAN(valid, torch.ones_like(valid))
        recon_loss = criterion_recon(fake_y, image_rgb)
        loss_G = gan_loss + 100 * recon_loss
        loss_G.backward()
        optimizer_G.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_semantic, image_rgb, fake_y, 'train_results', epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(dataloader)}] "
            f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

def validate(G, dataloader, criterion_GAN, criterion_recon, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = G(image_semantic)

            # Compute the loss
            loss = criterion_recon(outputs, image_rgb)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_semantic, image_rgb, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    G = GeneratorUNet(in_channels=3, out_channels=3).to(device)
    D = Discriminator(in_channels=3).to(device)
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_recon = nn.L1Loss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.2)
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(G, D, train_loader, optimizer_G, optimizer_D, criterion_GAN, criterion_recon, device, epoch, num_epochs)
        validate(G, val_loader, criterion_GAN, criterion_recon, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(G.state_dict(), f'checkpoints/G_epoch_{epoch + 1}.pth')
            torch.save(D.state_dict(), f'checkpoints/D_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
