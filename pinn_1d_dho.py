
"""
    ------------------------------
    - Harmonic Oscilator - PINNs -
    ------------------------------

    A _harmonic oscillator_ is a fundamental concept in physics that 
    describes a system where an object or a physical quantity oscillates 
    back and forth around an equilibrium position. The differential 
    equation (ODE) for a simple oscillator is given by:

    (1)                    mẍ + μẋ + kx = 0.

    In this code, we implement a neural network architecture based in
    constrained learning (physics-informed) to fit the soultion of a 
    under-damped harmonic oscillator.  

    author: @idiegoalvarado
    repo:   github.com/idiegoalvarado/pinn-dho

"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fcnn


def under_damped_oscillator(t, d, w0, X0):

    """
        The Damped Harmonic Oscillator (DHO) has exact analytical
        solution when the condition δ < ω₀. Where δ and ω₀ are defied
        as follows: 

        (2)             δ = µ/2m ; ω₀ = sqrt(k/m).

        This state is known as 'under-damped'. Given the initial
        conditions {x(0) = x₀; ẋ(0) = v₀} the solution to the ODE can
        be written as:

        (3)               2 A e^(-δt) cos(φ + ωt)). 
        
    """
    
    assert d < w0
    
    x0, v0 = X0
    w   = np.sqrt(w0 ** 2 - d ** 2)
    phi = np.arctan(- d / w - v0 / (x0 * w))
    A   = 1 / (2 * np.cos(phi))
    
    x   = torch.exp(- d * t) * (2 * A * torch.cos(phi + w * t))
    
    return x


# coefficients
m, mu, k = 1., 4., 225.
d  = mu / (2 * m)
w0 = np.sqrt(k / m)

x0, v0 = 1., 15.
X0 = torch.tensor([x0, v0])

# time domain
N = 500
t = torch.linspace(0, 1, N).view(-1,1)

# analytical solution
x_an = under_damped_oscillator(t, d, w0, X0).view(-1,1)

# training data
t_sample = int(len(t) * 2/5)
n_train  = 10

t_train =    t[0:t_sample:int(t_sample/n_train)]
x_train = x_an[0:t_sample:int(t_sample/n_train)]


"""
    Standard Fully Conected Neural Netork.

    It uses a the Mean Squared Error (MSE) loss function. The architecture
    of the neural network is fully-connected.

    (4)                 Loss = MSE(x,t,u)
    
"""

torch.manual_seed(123)
N_ts       = 5000
plt_offset = 10

model_snn     = fcnn.FCNN(1,1,32,3)
optimizer_snn = torch.optim.Adam(model_snn.parameters(), lr=1e-3)

x_snn = []

for i in range(N_ts):
    
    optimizer_snn.zero_grad()
    
    xn = model_snn(t_train)
    loss = torch.mean((xn - x_train) ** 2)
    
    loss.backward()
    optimizer_snn.step()
    
    if (i+1) % plt_offset == 0: 
        
        xn = model_snn(t).detach()
        x_snn.append(xn)


"""
    Physics-Informed Neural Network (PINN).

    The NN architecture is the same as for the standard NN arquitecture.
    In this case. the loss fuction is different. It combines physical
    constrait in the loss (physical loss):

    (5)          Loss = MSE(x,t,u) + ODE_loss(x,t,u)
    
"""

n_phy = 50
t_phy = torch.linspace(0,1,n_phy).view(-1,1).requires_grad_(True)

model_pinn     = fcnn.FCNN(1,1,32,3)
optimizer_pinn = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)

x_pinn = []

for i in range(N_ts):
    
    optimizer_pinn.zero_grad()
    
    xn = model_pinn(t_train)
    loss1 = torch.mean((xn - x_train) ** 2)
    
    # compute the "physics loss"
    xnp = model_pinn(t_phy)
    dx  = torch.autograd.grad(xnp, t_phy, torch.ones_like(xnp), create_graph=True)[0]
    dx2 = torch.autograd.grad(dx,  t_phy, torch.ones_like(dx),  create_graph=True)[0]
    phy = m * dx2 + mu * dx + k * xnp
    loss2 = (1e-4) * torch.mean(phy ** 2)
    
    # backpropagate joint loss
    loss_pinn = loss1 + loss2
    loss_pinn.backward()
    optimizer_pinn.step()
    
    if (i+1) % plt_offset == 0: 
        
        xnp = model_pinn(t).detach()
        x_pinn.append(xnp)


fig, ax = plt.subplots(2, 2, figsize=(11, 9))
fig.tight_layout(pad=4.0, rect=[0, 0, 1, 0.95])
fig.suptitle('Standard NN vs. Physics-Informed NN', fontsize=16, fontweight='bold')

ax[0,0].set_title('Standard NN Fit')
ax[0,0].set_ylim([- 1.05 * max(x_an), 1.05 * max(x_an)])
ax[0,0].set_xlabel('$t$ [s]')
ax[0,0].set_ylabel('$x$ [m]')
ax[0,0].plot(t, x_an, linestyle='--', color='Grey', label='Analytical solution', zorder=3)
ax[0,0].scatter(t_train, x_train, color='tab:orange', label='Trainig data')

ax[1,0].set_title('Physics-Informed NN Fit')
ax[1,0].set_ylim([- 1.05 * max(x_an), 1.05 * max(x_an)])
ax[1,0].set_xlabel('$t$ [s]')
ax[1,0].set_ylabel('$x$ [m]')
ax[1,0].plot(t, x_an, linestyle='--', color='Grey', label='Analytical solution', zorder=3)
ax[1,0].scatter(t_train, x_train, color='tab:orange', label='Trainig data')

ax[0,1].set_title('Approximation error: Standard NN')
ax[0,1].set_ylim([- 1.05 * max(x_an), 1.05 * max(x_an)])
ax[0,1].set_xlabel('$t$ [s]')
ax[0,1].set_ylabel('$x$ [m]')
ax[0,1].plot(t, x_an*0, linestyle='--', color='Grey', label='Analytical solution', zorder=3)

ax[1,1].set_title('Aproximation error: Physics-Informed NN')
ax[1,1].set_ylim([- 1.05 * max(x_an), 1.05 * max(x_an)])
ax[1,1].set_xlabel('$t$ [s]')
ax[1,1].set_ylabel('$x$ [m]')
ax[1,1].plot(t, x_an*0, linestyle='--', color='Grey', label='Analytical solution', zorder=3)

line_00, = ax[0,0].plot(t, x_snn[0]      , lw=2, label='NN approximation')
line_10, = ax[1,0].plot(t, x_pinn[0]     , lw=2, label='NN approximation')
line_01, = ax[0,1].plot(t, x_an-x_snn[0] , lw=2, label='NN approximation', color='tab:green')
line_11, = ax[1,1].plot(t, x_an-x_pinn[0], lw=2, label='NN approximation', color='tab:green')

label_text = 'Iteration Step: '
label_fig = fig.text(0.50,0.92, label_text, fontsize=12, 
                     horizontalalignment='center')

def animate(n):
    
    line_00.set_ydata(x_snn[n])
    line_10.set_ydata(x_pinn[n])
    line_01.set_ydata(x_snn[n] - x_an)
    line_11.set_ydata(x_pinn[n] - x_an)
    
    label_fig.set_text(label_text + str(n * plt_offset))
    
    return line_00, line_10, line_01, line_11,

ani = animation.FuncAnimation(fig, animate, frames=len(x_pinn), 
                                interval=20, blit=True)

save_anim = False
if save_anim:
    ani.save('pinn_vs_snn.mp4', fps=30, dpi=400)

plt.show()
