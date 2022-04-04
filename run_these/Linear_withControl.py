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
from linear_trans_wContr import NaiveLinear
from nflows.transforms.permutations import ReversePermutation

import global_variable as g

num_layers = 5
# MC sampling number
M = 100

g.num_iter = 15000
g.idx = 0
g.alpha = 1

base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    # do we need a permulation layer for the linear flow?
    # transforms.append(ReversePermutation(features=2))
    transforms.append(NaiveLinear(features=2))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())
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
    if (i + 1) % g.num_iter == 0:
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
        K_list = []
        V_list = []
        # print("iteration", i+1)
        para = list(flow.parameters())
        # para = list(flow.state_dict())

        j = 0
        # print(para)
        for a, b in para:
            # print (a, b)
            # K_list.append([a.data.tolist(), b.data.tolist()])

            if np.mod(j,2) == 0:
                V_list.append([a.data.tolist(), b.data.tolist()])
            else:
                K_list.append([a.data.tolist(), b.data.tolist()])
            j += 1


        # flow from the sampling dist. to the base dist. (normal)
        x, _ = datasets.make_gaussian_quantiles(mean=[0,0], cov=[5, 5], n_samples=M, n_features=2, n_classes=1)
        with torch.no_grad():
            x = torch.tensor(x)
        noise_out = flow.transform_to_noise(x.float())
        noise_out = noise_out.detach().numpy()

        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(111)
        # ax3.axis('equal')
        # plt.scatter(noise_out[:,0], noise_out[:,1])

A = np.matrix([[1, 0.3], [0.5, 1]])
B = np.matrix([[0.7, 0.3], [0.1, 0.8]])


K_list2 = [np.matrix(K_list[0])]
K_list_ = [np.matrix(K_list[0])]
coeff = 1
for m in range(len(K_list)-1):
    coeff = coeff * (A+B*(np.matrix(K_list[m])))
    K_list2.append(K_list[m+1] * coeff)
    K_list_.append(np.matrix(K_list[m+1]))

# print(K_list_)

"""
MC simulation
x_MC: propagation of x_{k+1} = A*x_{k} + B*K*u_{k} 
y_MC: propagation of x_{k+1} = A*x_{k} + B*K'*u_{0}
z_MC: utilization of Script A & B. x_{k+1} = ScrA * x0 + ScrB * ScrK * x0

"""

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = tf.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def confidence_ellipse2(cov, avg, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Take the covariance matrix, and return the ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = avg[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = avg[1]

    transf = tf.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


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


# Why not? 3d plot
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection='3d')
# for i in range(M):
#     mu0 = np.matrix([[0], [0]])
#     Sigma0 = np.matrix([[5, 0], [0, 5]])
#     # x0_MC = mu0 + np.sqrt(Sigma0) * np.random.uniform(low=-1,high=1,size=(2,1))
#     # x0_MC = [[random.gauss(0, 5)], [random.gauss(0, 5)]]
#     x0_MC = np.matrix(x[i]).T
#     x_MC = np.matrix(np.zeros((2, num_layers + 1)))
#
#     y_MC = copy.deepcopy(x_MC)
#     # Uvec = np.matrix(np.zeros((num_layers*2,1)))
#
#     for t in range(num_layers + 1):
#         if t == 0:
#             x_MC[:, t] = x0_MC
#         else:
#             U = K_list_[t - 1] * x_MC[:, t - 1]
#             Uy = K_list2[t - 1] * x0_MC
#             x_MC[:, t] = (A + B * K_list_[t - 1]) * x_MC[:, t - 1] + np.matrix(V_list[t - 1]).T
#             y_MC[:, t] = A * y_MC[:, t - 1] + B * Uy  # + np.matrix(V_list[t]).T
#
#         # Uvec[2*t,0] = Uy[0]
#         # Uvec[2*t+1,0] = Uy[1]
#     x_MC = np.asarray(x_MC)
#
#     # initial point
#     ax4.scatter3D(x_MC[0][0], 0, x_MC[1][0], color='k', s=2, zorder=2)
#     # trajectory
#     ax4.plot3D(x_MC[0], time, x_MC[1], 'gray', linewidth=0.3)
#     # terminal: These three return the same points
#     ax4.scatter3D(x_MC[0][num_layers], num_layers, x_MC[1][num_layers], color='r', s=2, zorder=2)
#     # plt.scatter(y_MC[0,-1], y_MC[1,-1], color='g', s=3)
#
#     ax4.scatter3D(x_MC[0][1], 1, x_MC[1][1], color='b', s=1, zorder=2)
#     ax4.scatter3D(x_MC[0][2], 2, x_MC[1][2], color='b', s=1, zorder=2)
#     ax4.scatter3D(x_MC[0][3], 3, x_MC[1][3], color='b', s=1, zorder=2)
#     ax4.scatter3D(x_MC[0][4], 4, x_MC[1][4], color='b', s=1, zorder=2)

plt.show()



