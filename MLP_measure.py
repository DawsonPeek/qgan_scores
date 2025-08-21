import math
import random
import numpy as np
import pennylane as qml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Parameters
image_size = 16
batch_size = 1


def get_args():
    parser = argparse.ArgumentParser(description='Quantum GAN Training')
    parser.add_argument('--num_iter', type=int, default=500, help='number of training iterations')
    parser.add_argument('--lrG', type=float, default=0.3, help='learning rate for generator')
    parser.add_argument('--lrD', type=float, default=0.01, help='learning rate for discriminator')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer type')
    parser.add_argument('--n_generators', type=int, default=4, help='number of sub-generators')
    parser.add_argument('--n_a_qubits', type=int, default=1, help='number of ancillary qubits')
    parser.add_argument('--n_layers', type=int, default=1, help='number of hidden layers')
    return parser.parse_args()


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

def create_quantum_circuit(n_qubits, n_a_qubits, q_depth):
    # Quantum simulator
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits, 2)

        # Initialise latent vectors
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
            qml.RX(noise[i + n_qubits], wires=i)

        # Repeated layer
        for i in range(q_depth):
            # Parameterised layer
            for y in range(n_qubits):
                qml.RX(weights[i][y][0], wires=y)
                qml.RY(weights[i][y][1], wires=y)

            # Control Z gates
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(n_qubits)))

    return quantum_circuit


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, n_a_qubits, n_layers=1, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            n_a_qubits (int): Number of ancillary qubits.
            n_layers (int): Number of hidden layers.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        total_pixels = image_size * image_size
        if total_pixels % n_generators != 0:
            raise ValueError(f"n_generators ({n_generators}) must divide total_pixels")

        pixels_per_generator = total_pixels // n_generators
        n_qubits = int(math.log2(pixels_per_generator)) + n_a_qubits

        self.n_generators = n_generators
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.q_depth = 6
        self.patch_size = 2 ** (self.n_qubits - self.n_a_qubits)

        self.quantum_circuit_fn = create_quantum_circuit(n_qubits, n_a_qubits, self.q_depth)

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(self.q_depth * n_qubits * 2), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

        # Classical neural network
        layers = []

        layers.append(nn.Linear(image_size * image_size, 500))
        layers.append(nn.ReLU())

        for _ in range(n_layers):
            layers.append(nn.Linear(500, 500))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(500, image_size * image_size))

        self.classical = nn.Sequential(*layers)

    def forward(self, x):
        images_list = []
        for params in self.q_params:
            patch_list = []
            for elem in x:
                q_out = self.quantum_circuit_fn(elem, params).float().unsqueeze(0)
                patch_list.append(q_out)
            patches = torch.cat(patch_list, dim=0)
            images_list.append(patches)

        output = torch.cat(images_list, dim=1)

        image = self.classical(output)
        image = torch.sigmoid(image)

        return image


######################################################################
# Training
######################################################################

def main(args):
    print(f"Starting training with arguments: {args}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((16, 16))])

    full_dataset = MNIST(root='/hpc/archive/G_QSLAB/emanuele.maffezzoli/data/', train=True, download=True,
                         transform=transform)
    indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
    dataset = torch.utils.data.Subset(full_dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Initialize models
    discriminator = Discriminator().to(device)
    generator = PatchQuantumGenerator(args.n_generators, args.n_a_qubits, args.n_layers).to(device)

    noise_dim = generator.n_qubits * 2

    # Metrics
    fid = FrechetInceptionDistance(feature=192, normalize=True).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    writer = SummaryWriter('/hpc/archive/G_QSLAB/emanuele.maffezzoli/tensorboard_logs/')

    # Binary cross entropy
    criterion = nn.BCELoss()

    if args.opt == 'Adam':
        optD = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(0.5, 0.999))
        optG = optim.Adam(generator.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    elif args.opt == 'SGD':
        optD = optim.SGD(discriminator.parameters(), lr=args.lrD)
        optG = optim.SGD(generator.parameters(), lr=args.lrG)
    else:
        raise ValueError("optimizer must be either 'SGD' or 'Adam'")

    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

    fixed_noise = torch.rand(8, noise_dim, device=device) * math.pi / 2

    # Collect real images for metrics calculation
    real_images = []
    for i, (data, _) in enumerate(dataloader):
        if i >= 5000:
            break
        real_images.append(data)

    real_images = torch.cat(real_images, dim=0)
    real_images_prepared = image_prep(real_images).to(device)

    # Update FID
    fid.update(real_images_prepared, real=True)

    counter = 0
    results = []

    while True:
        for i, (data, _) in enumerate(dataloader):

            # Data for training the discriminator
            data = data.reshape(-1, image_size * image_size)
            real_data = data.to(device)

            # Noise following a uniform distribution in range [0,pi/2)
            noise = torch.rand(batch_size, noise_dim, device=device) * math.pi / 2
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
                        noise = torch.rand(batch_size, noise_dim, device=device) * math.pi / 2
                        fake_img = generator(noise).view(1, 1, image_size, image_size)
                        fake_images.append(fake_img.detach())

                    fake_images = torch.cat(fake_images, dim=0)
                    fake_images_fid = image_prep(fake_images).to(device)
                    fake_images_lpips = image_prep(fake_images[:1000], lpips=True).to(device)

                    # Update FID
                    fid.update(fake_images_fid, real=False)
                    fid_score = fid.compute()

                    # Update LPIPS - batch calculation
                    half = len(fake_images_lpips) // 2
                    batch1 = fake_images_lpips[:half]  # Prima metà
                    batch2 = fake_images_lpips[half:half * 2]  # Seconda metà
                    avg_lpips = lpips(batch1, batch2).item()

                    # Log metrics to TensorBoard
                    writer.add_scalar('Metrics/FID_Score', fid_score, counter)
                    writer.add_scalar('Metrics/LPIPS_Score', avg_lpips, counter)

                    print(f'Iteration: {counter}, FID Score: {fid_score:.3f}, LPIPS Score: {avg_lpips:.3f}')

                    # Reset FID
                    fid.reset()
                    fid.update(real_images_prepared, real=True)

            if counter == args.num_iter:
                break
        if counter == args.num_iter:
            break

    writer.close()


if __name__ == "__main__":
    args = get_args()
    print(f"Configuration: {args}")
    main(args)