# TODO add regularizatoin prob L2
# TODO add dropout
# TODO add batch normilizatoin 

import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#======= SEED =======
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

'''
PINN try to learn the wave function for a particle using schrodingers and other defined physical boundaries 
as a loss function
'''

#for mac use mps for nvidia use cuda         
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}")

# Define a simple feedforward neural network or machinen learning perceptron (MLP)
class MLP(nn.Module):
    #depth is number of hidden layers we self define input and output 1 neuron layers
    #setting width sets number of neurons per layer like the number of weights we can have per layer so that one input is going to 64 places in next line for input layer
    def __init__(self, width=512, depth=8):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()] # defines input layer only taking 1 feature input right now
        # 2 hidden layers
        for _ in range(depth -1):
            layers += [nn.Linear(width, width), nn.Tanh()] #nn.Linear pushes data through handilng matrix multiplicaiotns
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers) # links layers together in sequence  you can call self.net(x) instead of x = layer1(x) x = layer2(x) etc...
        
    def forward(self, x):
        return self.net(x)
    
    # can improve nn by using normilizatoin regularizatoin etc. 
    
def d_dx(psi, x):
    """ 
    Compute the derivative of psi with respect to x using autograd 
    need this for looking at psi as we go on we need to minimize loss function but we need to 
    investigte schrodingers and other behavior that depend on psi should most likely be using a better 
    funciotn to return derivatives idk about autograd
    """
    return torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    '''
    by default returns touple need to used [0] to get tensor
    need create graph to allow higher order derivatives
    torch.one_like(psi) creates tensor of ones same shape as psi with all entries 1's to use as grad_outputs
    uses chain rule to fnid derivatives i think
    '''

"""
nn.linear(in_features, out_features) == Linear Peceptron layer
uses a linear hyperplane to seperate the data has
Applies a linear transformation to the incoming data: y = W x + b
we combine the weight vecotrs for each perceptron to form a weight matrix W
with absorbed bias vector b into it for each perceptron, we save time and ememory doing a weight matrix instead
of multiple weight vectors and with their own bias     
"""

# E_param = torch.nn.Parameter(torch.tensor([5.0], device=device)) #make energy a parameter to learn
# optimizer = optim.Adam(list(model.parameters()) + [E_param], lr=1e-3) #optimizer to use for training should have varrying learning rate see deepseek



#not sure how we know how to update E or if we should set it to be constant for some system


'''
model.paramaters() is an iterator over all learnable paramaters / tensors in the models 
for p in model.parameters():
    print(p.shape)
        : (64, 1)

            (64,)

            (64, 64)

            (1, 64)

            (1,)
            
    optim.adam(which paramaters to optimize wrt , learning rate the alpha)
        
        
'''

def psi_force_BC(x):
    """
    Enforce boundary conditions by construction
    psi(0) = 0 and psi(L) = 0
    call isntead of model(x) to get wavefunction that satisfies BCs
    not sure if this is better yet
    """
    return x * (L-x) * model(x) 

L = 1.0  
X_NORM = torch.linspace(0.0, L, 4096, device=device).view(-1, 1)

def psi_force_normalize(x, eps=1e-12):
    # force normalization of wavefunction at points x using batch estimate
    #Returns:
    #   psi_norm(x), norm_raw  (norm_raw is the integral of psi_bc^2 over [0,L])
    
    psi_grid = psi_force_BC(X_NORM)
    psi2 = (psi_grid.view(-1)**2)
    x1d = X_NORM.view(-1)
    norm_raw = torch.trapz(psi2, x1d)  # approximate integral over [0,L] using trap
    
    psi_norm=psi_force_BC(x) / torch.sqrt(norm_raw + eps)
    return psi_norm, norm_raw
    
def norm_check(eps=1e-12):
    with torch.no_grad():
        psi_norm_grid, _ = psi_force_normalize(X_NORM, eps=eps)
        psi2 = (psi_norm_grid.view(-1)**2)
        x1d = X_NORM.view(-1)
        return torch.trapz(psi2, x1d)  # approximate integral over [0,L] using trap

# returns schrodingers residual at points x assuming boundary conditions are enforced by construction
def Res_TISE_with_BC(x):
    # RMS loss of time-independent schrodingers equation with boundary conditions enforced by construction
    x.requires_grad_(True)
    psi, norm_est = psi_force_normalize(x)
    psi_x = d_dx(psi, x)
    psi_xx = d_dx(psi_x, x)
    # Schrödinger residual: -psi'' - E*psi = 0
    # use this in loss funcciotn should alwasy be close to zero for physicaly valid solution 
    #return -psi_xx - E_param * psi
    return -psi_xx - E_const * psi, norm_est

# online says to enforce boundary conditons before loss function not sure why...
# can enforce boundary contisoin by constructoin instead of adding to loss: (not used right now)

def with_bc_loss():
    #not used anymore
    # gives RMS loss of boundary conditions being incorrect
    x0 = torch.zeros(1, 1, device=device)
    xL = torch.full((1, 1), L, device=device)
    psi0 = psi_force_BC(x0)
    psiL = psi_force_BC(xL)
    return (psi0**2).mean() + (psiL**2).mean()

def norm_loss(Nn=256):
    #not used anymore
    # normilizatoin condition RMS loss
    x_n = torch.rand(Nn, 1, device=device) * L
    psi_n = psi_force_BC(x_n)
    norm_est = L * torch.mean(psi_n**2)
    return (norm_est - 1.0)**2, norm_est

# uses boundaries and schrodingers to define loss function what we are trying to minimize error wrt weights in NN
def loss_fn(Nf=256, Nn=256, lam_bc=1.0, lam_norm=1.0): #nf is number of randomly selected ponits were we want schrodingers to be valid randomly selected in the well
    # pick batch of interior points (0, L) to investigate
    x_f = torch.rand(Nf, 1, device=device) * L
    r, _ = Res_TISE_with_BC(x_f)
    loss_pde = torch.mean(r**2)

    #with Explicit BC Loss
    #loss_bc = with_bc_loss()
    
    #normilizatoin loss
    #loss_norm, norm_est = norm_loss(Nn)
    
    # total loss
    #total_loss = loss_pde + lam_bc * loss_bc + lam_norm * loss_norm
    total_loss = loss_pde  #+ lam_norm * loss_norm

    return total_loss, loss_pde.detach(), _.detach()

    #detach() doesnt save gradients for these losses we just want to monitor them not use them for backprop or anything that we need to store them with
    
# ==== Constants ======
L = 1.0  # Length of the well
n = 25    # Quantum number can do 1
LAM_BC = 1.0  # Weight for boundary condition loss
NUM_F = 1600  # Number of collocation points for PDE loss
NUM_N = 1600  # Number of points for normalization loss
LAM_NORM = 1.5  # Weight for normalization loss
LEARNING_RATE = 1e-2    # if getting stuck in local minima try to raise learnring rate should idealy use some varying learnring rate
TRAINING_STEPS = 22000
PLOT = True
TRAIN = True
LAYERS = 6
WIDTH = 128
# =====================

model = MLP(width=WIDTH, depth=LAYERS).to(device) #assigns what device to use for it uses random initial weights i think
E_const = ((n * math.pi) / L)**2 #analytical energy levels for infinite square well
A = torch.nn.Parameter(torch.tensor(10.0, device=device)) # learnable paramater that starts large to stop convergence towards trivial solution
optimizer = optim.Adam(list(model.parameters()) + [A], lr=LEARNING_RATE) #Adam!!

def train(optimizer=optimizer, training_steps=TRAINING_STEPS, Nf=NUM_F, Nn=NUM_N, lam_bc=LAM_BC, lam_norm=LAM_NORM, loss_fn=loss_fn):
    for step in range(1, training_steps + 1):
        optimizer.zero_grad()
        loss, loss_pde, _ = loss_fn(Nf=512, Nn=256)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            nc = norm_check()
            print(
                f"Step {step:4d}  "
                f"Loss={loss.item():.3e}  "
                f"PDELoss={loss_pde.item():.3e}  "
                #f"BCLoss={loss_bc.item():.3e}  "
                #f"NormLoss={loss_norm.item():.3e}  "
                #f"Norm={norm_est.item():.3e}  "
                f"NormCheck={nc.item():.6f}  "
                #f"E={E_param.item():.6f}"
            )
            
def test_psi(num=2000):
    model.eval()

    # Sample interior points
    x = torch.rand(num, 1, device=device) * L

    # PDE residual
    r, _ = Res_TISE_with_BC(x)
    norm_est = norm_check()
    r_rms = torch.sqrt(torch.mean(r**2)).item()

    with torch.no_grad():
        # Boundary values
        x0 = torch.zeros(1,1, device=device)
        xL = torch.full((1,1), L, device=device)
        psi0 = psi_force_BC(x0).abs().item()
        psiL = psi_force_BC(xL).abs().item()

        # Norm estimate via Monte Carlo
        #psi = psi_with_BC(x)
        #psi = psi_force_normalize(x)
        #norm = (L * torch.mean(psi**2)).item()

    print("\n====== Physical report ======")
    print(f"E_const        = {E_const:.6f}")
    print(f"Residual RMS  = {r_rms:.3e}")
    print(f"|psi(0)|      = {psi0:.3e}")
    print(f"|psi(L)|      = {psiL:.3e}")
    print(f"Norm          = {norm_est:.6f}")

#training loop
if TRAIN:
    train()
    
test_psi()          


# Plot learned wavefunction and probability density
if PLOT:
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(0.0, L, 400, device=device).view(-1, 1)
        psi, _ = psi_force_normalize(x_plot)
        psi = psi.view(-1)
        
        x_plot_np = x_plot.view(-1).cpu().numpy()
        psi_np = psi.cpu().numpy()
        prob_np = psi_np**2

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # --- Wavefunction ---
    axes[0].plot(x_plot_np, psi_np, color="blue")
    axes[0].axhline(0, linewidth=2, color="green", ls="--")
    axes[0].axvline(0, linewidth=2, color="green", ls="--")
    axes[0].axvline(L, linewidth=2, color="green", ls="--")
    axes[0].set_ylabel(r"$\psi(x)$")
    axes[0].set_title("Learned wavefunction")

    # --- Probability density ---
    axes[1].plot(x_plot_np, prob_np, color="magenta")
    axes[1].axhline(0, linewidth=2, color="green", ls="--")
    axes[1].axvline(0, linewidth=2, color="green", ls="--")
    axes[1].axvline(L, linewidth=2, color="green", ls="--")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$|\psi(x)|^2$")
    axes[1].set_title("Probability density")

    plt.tight_layout()
    
    if device == "mps" or device == "cpu":
        plt.show()
    else:
        plt.savefig("pinn_square_well.png", dpi=200, bbox_inches="tight")
