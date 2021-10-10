import os
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from detecto import core, utils


WORKING_DIRECTORY = 'ShelfImages'

os.chdir(WORKING_DIRECTORY)

# Specify a list of transformations for our dataset to apply on our images
transform_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(800),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

print('Setting up dataloader...')
dataset = core.Dataset('train.csv', 'images/', transform=transform_img)

# Loading single element to check everything works
image, target = dataset[0]

# Create our validation dataset
val_dataset = core.Dataset('val.csv', 'images/')

# Create the loader for our training dataset
loader = core.DataLoader(dataset, batch_size=2, shuffle=True)

print('Training...')
model = core.Model(['Product'])

losses = model.fit(loader, val_dataset, epochs=10, verbose=True)

# Plot the accuracy over time
plt.plot(losses)
plt.savefig('../loss.png')

model.save('../saved_models/model_weights.pth')
print('Training complete!\nModel weights in saved_models/ folder')