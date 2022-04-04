"""
Normalizing Flow Algorithm
Consider the flow z = f(x), where f(x) = fN @ ... @ f1 @ f0(x0)
i.e., x = distribution to be sampled at every iteration
      z = target distribution = base distribution

Dynamics: x_ = Ax + Bu, where u = KX + V (linear feedback law)
A, B: Dynamics coefficients (fixed)
K, V: Optimized Variable/parameters
"""


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as tf
import sklearn.datasets as datasets
import numpy as np
import copy
import random

import torch
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.utils.plotting import confidence_ellipse, confidence_ellipse2

from planer_trans_wContr import NaivePlaner
from nflows.transforms.permutations import ReversePermutation

import global_variable as g

num_layers = 32
# MC sampling number
M = 100

g.num_iter = 30000
g.idx = 0

base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    # do we need a permulation layer for the linear flow?
    # transforms.append(ReversePermutation(features=2))
    transforms.append(NaivePlaner(features=2))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters(), lr=0.002)  # default lr = 0.01
# print(list(flow.parameters()))

for i in range(g.num_iter):
    g.idx = i

    """ Generating Initial Distribution (x)...  """
    # moons
    # x, y = datasets.make_moons(128, noise=.1)

    # normal dist.
    x, _ = datasets.make_gaussian_quantiles(mean=[0,0], cov=[5,5], n_samples=128, n_features=2, n_classes=1)
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
    if (i + 1) % (g.num_iter/3) == 0:   # g.num_iter
        xline = torch.linspace(-3, 10, 100)
        yline = torch.linspace(-4, 10, 100)
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

# End of optimization (training): save the parameters.
path = r"C:\Users\yujit\OneDrive\ドキュメント\github\nflows\run_these\trained_models\planer.pt"
torch.save(flow.state_dict(), path)

# obtain the traind model
transforms = []
for _ in range(num_layers):
    # do we need a permulation layer for the linear flow?
    # transforms.append(ReversePermutation(features=2))
    transforms.append(NaivePlaner(features=2, using_cache=True))
transform = CompositeTransform(transforms)

trained_flow = Flow(transform, base_dist)
trained_flow.load_state_dict(torch.load(path))
trained_flow.eval()

x, _ = datasets.make_gaussian_quantiles(mean=[0,0], cov=[5, 5], n_samples=100, n_features=2, n_classes=1)
with torch.no_grad():
    x = torch.tensor(x)
noise_out = flow.transform_to_noise(x.float())
noise_out = noise_out.detach().numpy()
plt.scatter(noise_out[:,0], noise_out[:,1])
plt.axis('equal')
plt.show()




############### Until here.

A = np.matrix([[1, 0.3], [0.5, 1]])
B = np.matrix([[0.7, 0.3], [0.1, 0.8]])


"""
MC simulation
x_MC: propagation of x_{k+1} = A*x_{k} + B*K*u_{k} 
y_MC: propagation of x_{k+1} = A*x_{k} + B*K'*u_{0}
z_MC: utilization of Script A & B. x_{k+1} = ScrA * x0 + ScrB * ScrK * x0

"""


fig2 = plt.figure()
ax21 = fig2.add_subplot(111)
ax21.axis('equal')
ax21.grid()
ax21.set_title('Trajectory')
ax21.set_xlabel('x')
ax21.set_ylabel('y')


fig5 = plt.figure()
ax22 = fig5.add_subplot(211)
ax22.grid()
ax22.set_title('x vs time')
ax22.set_xlabel('layer')
ax22.set_ylabel('x')

ax23 = fig5.add_subplot(212)
ax23.grid()
ax23.set_title('y vs time')
ax23.set_xlabel('layer')
ax23.set_ylabel('y')

# ax24 = fig2.add_subplot(224)
# ax24.grid()
# ax24.set_title('u vs time')
# ax24.set_xlabel('layer')
# ax24.set_ylabel('u')

# fig6 = plt.figure()
# ax61 = fig6.add_subplot(111)
# ax61.grid()
# ax61.set_title('control vs time')
# ax61.set_xlabel('step')
# ax61.set_ylabel('control')


output_MC = []
x_log = np.zeros((M,num_layers+1))
y_log = np.zeros((M,num_layers+1))
unorm_log = np.zeros((M,1))

time = range(num_layers+1)
for i in range(M):
    mu0 = np.matrix([[0], [0]])
    Sigma0 = np.matrix([[5, 0], [0,5]])
    # x0_MC = mu0 + np.sqrt(Sigma0) * np.random.uniform(low=-1,high=1,size=(2,1))
    # x0_MC = [[random.gauss(0, 5)], [random.gauss(0, 5)]]
    x0_MC = np.matrix(x[i]).T
    x_MC = np.matrix(np.zeros((2, num_layers+1)))
    u_MC = np.zeros((2,num_layers))
    unorm_MC = np.zeros(num_layers)

    y_MC = copy.deepcopy(x_MC)
    # Uvec = np.matrix(np.zeros((num_layers*2,1)))
    
    for t in range(num_layers+1):
        if t == 0:
            x_MC[:, t] = x0_MC
        else:
            U = K_list_[t-1] * x_MC[:,t-1]
            u_MC[:,t-1] = U.ravel()
            unorm_MC[t-1] = np.linalg.norm(U.ravel())
            Uy = K_list2[t-1] * x0_MC
            x_MC[:,t] = (A + B*K_list_[t-1])*x_MC[:,t-1] + np.matrix(V_list[t-1]).T
            y_MC[:,t] = A*y_MC[:,t-1] + B*Uy #+ np.matrix(V_list[t]).T

        # Uvec[2*t,0] = Uy[0]
        # Uvec[2*t+1,0] = Uy[1]
    x_MC = np.asarray(x_MC)

    # ax21 plots
    # initial point
    ax21.scatter(x_MC[0][0], x_MC[1][0], color='r', s=2, zorder=2)
    # trajectory
    ax21.plot(x_MC[0], x_MC[1], '-', color='lightgrey', linewidth=1, zorder=0)
    # terminal: These three return the same points
    ax21.scatter(x_MC[0][num_layers], x_MC[1][num_layers], color='k', s=2, zorder=2)
    # plt.scatter(y_MC[0,-1], y_MC[1,-1], color='g', s=3)

    # ax22 plots: x vs time
    ax22.plot(time, x_MC[0], '-', color='lightgrey', linewidth=1, zorder=1)
    x_log[i,:] = x_MC[0]

    # ax23 plots: y vs time
    ax23.plot(time, x_MC[1], '-', color='lightgrey', linewidth=1, zorder=1)
    y_log[i,:] = x_MC[1]

    unorm_log[i,0] = sum(unorm_MC)

    # ax61.plot(time[0:-1], u_MC, '-', color='lightgrey', linewidth=1, zorder=1)

unorm_avg = np.mean(unorm_log)
print("avg. of U_norm = %s" %unorm_avg)

output_MC = np.array(output_MC)
cov_x = np.zeros(num_layers+1)
cov_y = np.zeros(num_layers+1)


# ellipse
for k in range(num_layers+1):
    cov_x[k] = np.var(x_log[:,k])
    cov_y[k] = np.var(y_log[:,k])
    if k == 0 or k == num_layers:
        confidence_ellipse(x_log[:, k], y_log[:, k], ax21, edgecolor='blue')
    # else:
    #     confidence_ellipse(x_log[:,k], y_log[:,k], ax21, edgecolor='blue')

confidence_ellipse2(np.array([[1,0],[0,1]]), [0,0], ax21, edgecolor='red')
confidence_ellipse2(np.array([[5,0],[0,5]]), [0,0], ax21, edgecolor='red')

ax22.plot(time, cov_x, '-', color='r', linewidth=2, zorder=2)
plt.tight_layout()
ax23.plot(time, cov_y, '-', color='r', linewidth=2, zorder=2)
plt.tight_layout()

plt.show()



