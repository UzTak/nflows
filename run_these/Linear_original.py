"""
Normalizing Flow Algorithm
Consider the flow z = f(x), where f(x) = fN @ ... @ f1 @ f0(x0)
i.e., x = distribution to be sampled at every iteration
      z = target distribution = base distribution
"""


import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.linear import NaiveLinear
from nflows.transforms.permutations import ReversePermutation

import global_variable as g
g.alpha = 1


num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    # do we need a permulation layer for the linear flow?
    # transforms.append(ReversePermutation(features=2))
    transforms.append(NaiveLinear(features=2))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 2000
for i in range(num_iter):

    """ Generating Initial Distribution (x)...  """
    # moons
    # x, y = datasets.make_moons(128, noise=.1)

    # normal dist.
    x, _ = datasets.make_gaussian_quantiles(mean=[3,4], cov=[2,2], n_samples=128, n_features=2, n_classes=1)
    # x = data[:,0]
    # y = data[:,1]
    # plt.scatter(x,y)

    # multimodal normal dist.
    # x1, y1 = datasets.make_gaussian_quantiles(mean=[1,2], n_samples=50)
    # x2, y2 = datasets.make_gaussian_quantiles(mean=[-1,-2], n_samples=100)
    # x = np.concatenate([x1, x2], 0)


    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    # print(loss)
    loss.backward()
    optimizer.step()

    # at the final iteration
    # flow from Normal (base) to the sampling dist.
    if (i + 1) % 2000 == 0:
        xline = torch.linspace(-3, 5, 100)
        yline = torch.linspace(-4, 6, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        # concatenate
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        # we don't need a gradient for this, so nullify the gradient info
        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_aspect('equal', adjustable='box')
        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()

        # printing the parameters
        for name, param in flow.parameters():
            print (name, param.data)

    # flow from the sampling dist. to the base dist. (normal)
    if (i + 1) % 2000 == 0:
        x, _ = datasets.make_gaussian_quantiles(mean=[3, 4], cov=[2, 2], n_samples=10000, n_features=2, n_classes=1)
        with torch.no_grad():
            x = torch.tensor(x)
        noise_out = flow.transform_to_noise(x.float())
        noise_out = noise_out.detach().numpy()
        plt.scatter(noise_out[:,0], noise_out[:,1])
        plt.show()