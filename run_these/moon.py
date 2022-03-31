import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedUMNNAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedUMNNAutoregressiveTransform(features=2,
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 2000
for i in range(num_iter):
    # moons
    x, _ = datasets.make_moons(128, noise=.1)

    # multimodal normal dist.
    x1, _ = datasets.make_gaussian_quantiles(mean=[1,2], n_samples=50)
    x2, _ = datasets.make_gaussian_quantiles(mean=[-1,-2], n_samples=100)
    x = np.concatenate([x1, x2], 0)

    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()

    # at the final iteration
    if (i + 1) % 2000 == 0:
        xline = torch.linspace(-3, 5, 100)
        yline = torch.linspace(-4, 4, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        # concatenate
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        # we don't need a gradient for this, so nullify the gradient info
        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()