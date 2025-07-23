import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def image_prep(images, target_size=32, lpips=False):
    # Pytorch format
    images = images.view(-1, 1, image_size, image_size)

    # RGB
    images = images.repeat(1, 3, 1, 1)

    if lpips:
        images = torch.nn.functional.interpolate(images, size=(target_size, target_size), mode='bilinear',
                                                 align_corners=False)
        images = images.clamp(0, 1).float()
    else:
        images = (images * 255).clamp(0, 255).byte()

    return images


######################################################################
# Data (MNIST dataset filtered for zeros)
######################################################################

# Variables and dataloader
image_size = 16
batch_size = 1

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((16, 16)),])

full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
dataset = torch.utils.data.Subset(full_dataset, indices)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# # Let's visualize some of the data.
#
# plt.figure(figsize=(8,2))
#
# for i in range(8):
#     image = dataset[i][0].reshape(image_size,image_size)
#     plt.subplot(1,8,i+1)
#     plt.axis('off')
#     plt.imshow(image.numpy(), cmap='gray')
#
# plt.show()


######################################################################
#Discriminator
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
dev = qml.device("lightning.qubit", wires=n_qubits)
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

# Metrics
fid_metric = FrechetInceptionDistance(feature=192, normalize=True).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

#Binary cross entropy
criterion = nn.BCELoss()

optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

# Collect 1000 real images for metrics calculation
real_images = []
for i, (data, _) in enumerate(dataloader):
    if i >= 1000:
        break
    real_images.append(data)

real_images = torch.cat(real_images, dim=0)
real_images_prepared = image_prep(real_images).to(device)

# Update FID
fid_metric.update(real_images_prepared, real=True)

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
            test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()

            if counter % 50 == 0:
                results.append(test_images)

                # Generate fake images for metrics
                fake_images = []
                for _ in range(1000):
                    noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
                    fake_img = generator(noise).view(1, 1, image_size, image_size)
                    fake_images.append(fake_img.detach())

                fake_images = torch.cat(fake_images, dim=0)
                fake_images_fid = image_prep(fake_images).to(device)
                fake_images_lpips = image_prep(fake_images[:200], lpips=True).to(device)

                # Update FID
                fid_metric.update(fake_images_fid, real=False)
                fid_score = fid_metric.compute()

                # Update LPIPS
                half = len(fake_images_lpips) // 2
                lpips_scores = []
                for j in range(half):
                    img1 = fake_images_lpips[j].unsqueeze(0)
                    img2 = fake_images_lpips[j + half].unsqueeze(0)
                    lpips_score = lpips_metric(img1, img2)
                    lpips_scores.append(lpips_score.item())

                avg_lpips = np.mean(lpips_scores)

                print(f'Iteration: {counter}, FID Score: {fid_score:.3f}, LPIPS Score: {avg_lpips:.3f}')

                # Reset FID
                fid_metric.reset()
                fid_metric.update(real_images_prepared, real=True)

        if counter == num_iter:
            break
    if counter == num_iter:
        break

######################################################################
# Plot results
######################################################################

fig = plt.figure(figsize=(10, 10))
outer = gridspec.GridSpec(10, 2, wspace=0.2)

for i, images in enumerate(results):
    inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
                    subplot_spec=outer[i])

    images = torch.squeeze(images, dim=1)
    for j, im in enumerate(images):

        ax = plt.Subplot(fig, inner[j])
        ax.imshow(im.numpy(), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_title(f'Iteration {50 + i * 50}', loc='left')
        fig.add_subplot(ax)

plt.show()