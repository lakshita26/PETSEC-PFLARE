import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

from AI4PDEs_utils import get_weights_linear_2D

# -----------------------------
# PARAMETERS
# -----------------------------
dt = 0.05
dx = 1.0
dy = 1.0
Re = 0.0
# Re = 0.5
ub = 1.0
nx = 256
ny = 256
ntime = 100

tolerance = 1e-6
max_iter = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CREATE STENCIL WEIGHTS
# -----------------------------
w1, w2, w3, wA, w_res, diag = get_weights_linear_2D(dx)

w1 = w1.to(device)
w2 = w2.to(device)
w3 = w3.to(device)

bias_initializer = torch.tensor([0.0], device=device)

# -----------------------------
# SOLVER CLASS
# -----------------------------
class AI4CFD(nn.Module):

    def __init__(self):
        super(AI4CFD, self).__init__()

        self.xadv = nn.Conv2d(1,1,3,1,0)
        self.yadv = nn.Conv2d(1,1,3,1,0)
        self.diff = nn.Conv2d(1,1,3,1,0)

        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1

        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer

    def L_operator(self, u):

        u_pad = F.pad(u, (1,1,1,1), mode='constant', value=0)

        ADx = self.xadv(u_pad)
        ADy = self.yadv(u_pad)
        AD2 = self.diff(u_pad)

        return Re * AD2 - ub * ADx - ub * ADy

    def apply_A(self, u):
        return u - dt * self.L_operator(u)

    def implicit_step(self, u_old):

        u_new = u_old.clone()

        for k in range(max_iter):

            Au = self.apply_A(u_new)
            residual = Au - u_old

            res_norm = torch.norm(residual)

            if res_norm < tolerance:
                final_res = res_norm.item()
                break


            # Proper Jacobi scaling (diagonal ≈ 1 here)
            u_new = u_new - residual

        return u_new, final_res

# -----------------------------
# INITIAL CONDITION
# -----------------------------
x = torch.arange(nx, device=device) * dx
y = torch.arange(ny, device=device) * dx
Y, X = torch.meshgrid(y, x, indexing="ij")

def UNIT_TEST(c):
    mass = torch.sum(c)*dx*dy
    if mass < 1e-5: return 0.0, 0.0, 0.0, 0.0
    x_com = torch.sum(c*X)*dx*dy/mass
    y_com = torch.sum(c*Y)*dx*dy/mass
    var = torch.sum(c*((X-x_com)**2+(Y-y_com)**2))*dx*dy/mass
    return mass.item(), x_com.item(), y_com.item(), var.item()

values_u = torch.zeros((1,1,ny,nx), device=device)

x0 = nx*dx/2
y0 = ny*dx/2
a  = 25.0

values_u[0,0,
         (torch.abs(X-x0)<=a) &
         (torch.abs(Y-y0)<=a)] = 1.0

# -----------------------------
# RUN SIMULATION
# -----------------------------
model = AI4CFD().to(device)

print(f"{'Step':<5} | {'Residual':<10} | {'Step Time':<10} | {'Mass':<10} | {'Max_Val':<10}")
print("-" * 60)

mass, _, _, _ = UNIT_TEST(values_u)
print(f"{0:<5} | {'---':<10} | {'---':<10} | {mass:<10.2f} | {values_u.max():<10.2f}")

total_start = time.time()

with torch.no_grad():

    for itime in range(1, ntime+1):

        step_start = time.time()

        values_u, step_res = model.implicit_step(values_u)

        step_elapsed = time.time() - step_start

        mass, _, _, _ = UNIT_TEST(values_u)

        print(f"{itime:<5} | {step_res:<10.2e} | {step_elapsed:<10.4f} | {mass:<10.2f} | {values_u.max():<10.2f}")


total_end = time.time()
print("-" * 60)
print(f"Total Elapsed Time: {total_end-total_start:.4f} seconds")
print("-" * 60)

# ---- BUILD EXPLICIT A MATRIX ----
nx_small = 3
ny_small = 3
N = nx_small * ny_small

A = torch.zeros((N, N), device=device)

def flatten(i, j):
    return i * nx_small + j

for i in range(ny_small):
    for j in range(nx_small):

        u = torch.zeros((1,1,ny_small,nx_small), device=device)
        u[0,0,i,j] = 1.0

        Au = model.apply_A(u)

        col = flatten(i,j)
        A[:, col] = Au.view(-1)

import numpy as np

A_np = A.detach().cpu().numpy()

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=150
)

print("\nExplicit A matrix:\n")
print(A_np)


# -----------------------------
# PLOT RESULT
# -----------------------------
plt.figure(figsize=(6,5))
plt.imshow(values_u[0,0].cpu())
plt.colorbar()
plt.title("Matrix-Free Implicit Solution")
if Re>0.0:
    plt.savefig('sol_Re.png')
else:
    plt.savefig('sol_pure.png')

