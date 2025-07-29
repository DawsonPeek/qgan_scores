import math
import random
import numpy as np
import pennylane as qml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision import models
from scipy.linalg import sqrtm
from torchvision.utils import make_grid

import lpips
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

######################################################################
# Custom FID Implementation
######################################################################

# Load the pre-trained InceptionV3 model for feature extraction
inception = models.inception_v3(pretrained=True, transform_input=False)
inception.fc = torch.nn.Identity()  # Remove the final classification layer
inception.eval()

# Transform to match InceptionV3 input size and normalization
transform_fid = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to extract features using InceptionV3
def extract_features(images, model):
    with torch.no_grad():
        features = model(images)
    return features


# Function to compute mean and covariance of features
def calculate_mean_cov(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# Function to calculate the FID score
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


# Function to compute FID between real and generated images
def compute_fid(real_images, fake_images, n_features):
    real_images_reshaped = real_images.view(-1, 1, int(np.sqrt(n_features)), int(np.sqrt(n_features)))
    fake_images_reshaped = fake_images.view(-1, 1, int(np.sqrt(n_features)), int(np.sqrt(n_features)))

    real_images_reshaped = torch.stack(
        [(image - image.min()) / (image.max() - image.min()) for image in real_images_reshaped])
    fake_images_reshaped = torch.stack(
        [(image - image.min()) / (image.max() - image.min()) for image in fake_images_reshaped])

    real_images_rgb = real_images_reshaped.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
    fake_images_rgb = fake_images_reshaped.repeat(1, 3, 1, 1)

    real_images_fid = torch.stack([transform_fid(image) for image in real_images_rgb])
    fake_images_fid = torch.stack([transform_fid(image) for image in fake_images_rgb])

    real_features = extract_features(real_images_fid, inception)
    fake_features = extract_features(fake_images_fid, inception)

    mu_real, sigma_real = calculate_mean_cov(real_features.cpu().numpy())
    mu_fake, sigma_fake = calculate_mean_cov(fake_features.cpu().numpy())

    fid_value = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value


# Function to compute LPIPS score
def compute_lpips(real_images, fake_images, n_features, target_size=64):
    # Reshape the images to their original size
    real_images_reshaped = real_images.view(-1, 1, int(np.sqrt(n_features)), int(np.sqrt(n_features)))
    fake_images_reshaped = fake_images.view(-1, 1, int(np.sqrt(n_features)), int(np.sqrt(n_features)))

    # Resize images to the target size (e.g., 64x64)
    real_images_resized = F.interpolate(real_images_reshaped, size=(target_size, target_size), mode='bilinear')
    fake_images_resized = F.interpolate(fake_images_reshaped, size=(target_size, target_size), mode='bilinear')

    # Convert grayscale (1 channel) to RGB (3 channels)
    real_images_rgb = real_images_resized.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
    fake_images_rgb = fake_images_resized.repeat(1, 3, 1, 1)

    # Compute LPIPS score
    lpips_score = lpips_metric(real_images_rgb, fake_images_rgb)

    # Return the mean LPIPS score for the batch
    return lpips_score.mean().item()


######################################################################
# Image Preparation Function (Modified)
######################################################################

# def image_prep(images, target_size=32, lpips=False):
#     # Pytorch format
#     images = images.view(-1, 1, image_size, image_size)
#
#     # RGB
#     images = images.repeat(1, 3, 1, 1)
#
#     if lpips:
#         images = torch.nn.functional.interpolate(images, size=(target_size, target_size), mode='bilinear',
#                                                  align_corners=False)
#         images = images.clamp(0, 1).float()
#     else:
#         images = (images * 255).clamp(0, 255).byte()
#
#     return images


######################################################################
# Data (MNIST dataset filtered for zeros)
######################################################################

# Variables and dataloader
image_size = 16
batch_size = 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((16, 16)), ])

full_dataset = MNIST(root='/hpc/archive/G_QSLAB/emanuele.maffezzoli/data/', train=True, download=True, transform=transform)
indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
dataset = torch.utils.data.Subset(full_dataset, indices)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


######################################################################
# Discriminator
######################################################################

class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


######################################################################
# Quantum Generator
######################################################################

# Quantum variables
n_qubits = 7  # Total number of qubits
n_a_qubits = 1  # Number of ancillary qubits
q_depth = 6  # Depth of the parameterised quantum circuit
n_generators = 4  # Number of subgenerators for the patch method

# Quantum simulator
dev = qml.device("default.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@qml.qnode(dev, diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))


def partial_measure(noise, weights):
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        patch_size = 2 ** (n_qubits - n_a_qubits)
        images = torch.Tensor(x.size(0), 0).to(device)

        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            images = torch.cat((images, patches), 1)

        return images


######################################################################
# Training
######################################################################

lrG = 0.3  # Learning rate for the generator
lrD = 0.01  # Learning rate for the discriminator
num_iter = 500  # Number of training iterations

discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(n_generators).to(device)

lpips_metric = lpips.LPIPS(net='alex')  # Initialize LPIPS metric

# TensorBoard writer
writer = SummaryWriter('/hpc/archive/G_QSLAB/emanuele.maffezzoli/tensorboard_logs/')

# Binary cross entropy
criterion = nn.BCELoss()

optimizer_type = 'SGD'

if optimizer_type == 'Adam':
    optD = optim.Adam(discriminator.parameters(), lr=lrD, betas=(0.5, 0.999))
    optG = optim.Adam(generator.parameters(), lr=lrG, betas=(0.5, 0.999))
elif optimizer_type == 'SGD':
    optD = optim.SGD(discriminator.parameters(), lr=lrD)
    optG = optim.SGD(generator.parameters(), lr=lrG)
else:
    raise ValueError("optimizer_type must be either 'SGD' or 'Adam'")

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

# Collect real images for metrics calculation
real_images = []
for i, (data, _) in enumerate(dataloader):
    if i >= 5000:
        break
    real_images.append(data)

real_images = torch.cat(real_images, dim=0)

counter = 0
results = []

while True:
    for i, (data, _) in enumerate(dataloader):

        # Data for training the discriminator
        data = data.reshape(-1, image_size * image_size)
        real_data = data.to(device)

        # Noise following a uniform distribution in range [0,pi/2)
        noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
        fake_data = generator(noise)

        # Training the discriminator
        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        optD.step()

        # Training the generator
        generator.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = criterion(outD_fake, real_labels)
        errG.backward()
        optG.step()

        counter += 1

        # Show values
        if counter % 10 == 0:
            print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

            writer.add_scalar('Loss/Discriminator', errD.item(), counter)
            writer.add_scalar('Loss/Generator', errG.item(), counter)

            test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()

            if counter % 50 == 0:
                results.append(test_images)

                grid = make_grid(test_images, nrow=4, normalize=True, scale_each=True)
                writer.add_image('Generated Images', grid, counter)

                # Generate fake images for metrics
                fake_images = []
                for _ in range(5000):
                    noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
                    fake_img = generator(noise).view(1, 1, image_size, image_size)
                    fake_images.append(fake_img.detach())

                fake_images = torch.cat(fake_images, dim=0)

                # Calculate FID
                fid_score = compute_fid(real_images, fake_images, image_size * image_size)

                # Calculate LPIPS
                avg_lpips = compute_lpips(real_images, fake_images, image_size * image_size)

                # Log metrics to TensorBoard
                writer.add_scalar('Metrics/FID_Score', fid_score, counter)
                writer.add_scalar('Metrics/LPIPS_Score', avg_lpips, counter)

                print(f'Iteration: {counter}, FID Score: {fid_score:.3f}, LPIPS Score: {avg_lpips:.3f}')

        if counter == num_iter:
            break
    if counter == num_iter:
        break

writer.close()