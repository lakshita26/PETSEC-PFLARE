import sys
import numpy as np
import matplotlib.pyplot as plt
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import os

# ==========================
# PUBLICATION PLOT STYLING
# ==========================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
})

# ==========================
# PAPER REQUIREMENTS
# ==========================
Re = 100.0  
Pr = 0.7    
nu = 1.0 / Re
nu_art = 0.0015       
alpha_art = 0.0025    

D_cells = 20                  
dx = 1.0 / D_cells            
dt = 0.005                    

beta = 0.25                   
H_cells = int(D_cells / beta) 
Xu_cells = int(45 * D_cells)  
Xd_cells = int(120 * D_cells) 

nx = Xu_cells + Xd_cells      
ny = H_cells                  
ntime = 60000

# Cylinder positioning
cor_x = Xu_cells             
cor_y = int(ny / 2)          
radius = int(D_cells / 2)    

os.makedirs("results", exist_ok=True)

# ==========================
# NUMPY DOMAIN INITIALIZATION
# ==========================
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))
T = np.zeros((nx, ny))

# Add initial perturbation to trigger vortex shedding
np.random.seed(42)
u += 1e-4 * np.random.randn(nx, ny)

# Generate cylinder mask (1 inside, 0 outside)
mask = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        if (i - cor_x)**2 + (j - cor_y)**2 <= radius**2 and i <= cor_x:
            mask[i, j] = 1.0

# ==========================
# PETSc POISSON SOLVER SETUP
# ==========================
# 1-DoF DMDA specifically optimized for the Pressure matrix
da = PETSc.DMDA().create([nx, ny], dof=1, stencil_width=1, stencil_type=PETSc.DMDA.StencilType.STAR)
A = da.createMatrix()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()

# Assemble Sparse Matrix with STRICT Boundary Enforcement
for i in range(nx):
    for j in range(ny):
        row.index = (i, j); row.field = 0
        
        if i == nx - 1:
            # Outlet: Dirichlet (p = 0)
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
        elif i == 0:
            # Inlet: Neumann (dp/dx = 0 -> p_0 - p_1 = 0)
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
            col.index = (i+1, j); col.field = 0; A.setValueStencil(row, col, -1.0)
        elif j == 0:
            # Bottom Wall: Neumann (dp/dy = 0 -> p_0 - p_1 = 0)
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
            col.index = (i, j+1); col.field = 0; A.setValueStencil(row, col, -1.0)
        elif j == ny - 1:
            # Top Wall: Neumann (dp/dy = 0 -> p_{ny-1} - p_{ny-2} = 0)
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
            col.index = (i, j-1); col.field = 0; A.setValueStencil(row, col, -1.0)
        else:
            # Interior: Standard 5-point Laplacian
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, -4.0 / dx**2)
            col.index = (i-1, j); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)
            col.index = (i+1, j); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)
            col.index = (i, j-1); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)
            col.index = (i, j+1); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)

A.assemble()

# Configure Algebraic Multigrid (GAMG) Preconditioner for extreme speed
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.GAMG) 
ksp.setTolerances(rtol=1e-5)

P_vec = da.createGlobalVec()
RHS_vec = da.createGlobalVec()

# ==========================
# NUMPY MATH UTILITIES
# ==========================
def apply_boundaries(u_pad, v_pad, T_pad):
    y_idx = np.arange(ny)
    y_val = y_idx / (beta * (ny - 1))
    u_prof = 1.5 * (1.0 - np.abs(1.0 - 2.0 * beta * y_val)**2)
    
    u_pad[0, 1:-1] = u_prof
    v_pad[0, 1:-1] = 0.0
    T_pad[0, 1:-1] = 0.0
    
    u_pad[-1, 1:-1] = u_pad[-2, 1:-1]
    v_pad[-1, 1:-1] = v_pad[-2, 1:-1]
    T_pad[-1, 1:-1] = T_pad[-2, 1:-1]
    
    u_pad[1:-1, 0] = 0.0
    v_pad[1:-1, 0] = 0.0
    T_pad[1:-1, 0] = T_pad[1:-1, 1]
    
    u_pad[1:-1, -1] = 0.0
    v_pad[1:-1, -1] = 0.0
    T_pad[1:-1, -1] = T_pad[1:-1, -2]
    return u_pad, v_pad, T_pad

def calc_derivs(f):
    dfdx = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
    dfdy = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx)
    lapf = (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4*f[1:-1, 1:-1]) / dx**2
    return dfdx, dfdy, lapf

def upwind_advection(f_pad, u_vel, v_vel):
    f_inner = f_pad[1:-1, 1:-1]
    f_left  = f_pad[:-2, 1:-1]
    f_right = f_pad[2:, 1:-1]
    f_down  = f_pad[1:-1, :-2]
    f_up    = f_pad[1:-1, 2:]
    
    dfdx = np.where(u_vel > 0, (f_inner - f_left)/dx, (f_right - f_inner)/dx)
    dfdy = np.where(v_vel > 0, (f_inner - f_down)/dx, (f_up - f_inner)/dx)
    return u_vel * dfdx + v_vel * dfdy

def calc_grad_p(p_field):
    p_pad = np.pad(p_field, 1, mode='edge')
    p_pad[-1, :] = 0.0 # Outlet Dirichlet
    px = (p_pad[2:, 1:-1] - p_pad[:-2, 1:-1]) / (2 * dx)
    py = (p_pad[1:-1, 2:] - p_pad[1:-1, :-2]) / (2 * dx)
    return px, py

# ==========================
# MAIN TIME LOOP
# ==========================
PETSc.Sys.Print(f"Starting NumPy-PETSc Hybrid CFD Simulation for Re={Re}...")

probe_x, probe_y = cor_x + int(1.5 * D_cells), cor_y + 3
phase_state, t_start, T_period_steps, saved_phases, prev_v_probe = 0, 0, 0, 0, 0.0
target_steps = []

for t in range(ntime):
    
    # --- 1. PREDICTOR STEP ---
    u_pad = np.pad(u, 1, mode='constant')
    v_pad = np.pad(v, 1, mode='constant')
    T_pad = np.pad(T, 1, mode='constant')
    apply_boundaries(u_pad, v_pad, T_pad)
    
    adv_u = upwind_advection(u_pad, u, v)
    adv_v = upwind_advection(v_pad, u, v)
    adv_T = upwind_advection(T_pad, u, v)
    
    _, _, lap_u = calc_derivs(u_pad)
    _, _, lap_v = calc_derivs(v_pad)
    _, _, lap_T = calc_derivs(T_pad)
    
    u_pred = u + 0.5 * dt * (-adv_u + (nu + nu_art) * lap_u)
    v_pred = v + 0.5 * dt * (-adv_v + (nu + nu_art) * lap_v)
    T_pred = T + 0.5 * dt * (-adv_T + (1.0 / (Re * Pr) + alpha_art) * lap_T)
    
    u_pred = u_pred * (1 - mask)
    v_pred = v_pred * (1 - mask)
    T_pred = T_pred * (1 - mask) + mask
    
    # --- 2. CORRECTOR STEP ---
    u_pred_pad = np.pad(u_pred, 1, mode='constant')
    v_pred_pad = np.pad(v_pred, 1, mode='constant')
    T_pred_pad = np.pad(T_pred, 1, mode='constant')
    apply_boundaries(u_pred_pad, v_pred_pad, T_pred_pad)
    
    adv_u_star = upwind_advection(u_pred_pad, u_pred, v_pred)
    adv_v_star = upwind_advection(v_pred_pad, u_pred, v_pred)
    adv_T_star = upwind_advection(T_pred_pad, u_pred, v_pred)
    
    _, _, lap_u_star = calc_derivs(u_pred_pad)
    _, _, lap_v_star = calc_derivs(v_pred_pad)
    _, _, lap_T_star = calc_derivs(T_pred_pad)
    
    u_star = u + dt * (-adv_u_star + (nu + nu_art) * lap_u_star)
    v_star = v + dt * (-adv_v_star + (nu + nu_art) * lap_v_star)
    T_new  = T + dt * (-adv_T_star + (1.0 / (Re * Pr) + alpha_art) * lap_T_star)
    
    u_star = u_star * (1 - mask)
    v_star = v_star * (1 - mask)
    T_new = T_new * (1 - mask) + mask
    
    # --- 3. PRESSURE POISSON SOLVE (PETSc) ---
    u_star_pad = np.pad(u_star, 1, mode='constant')
    v_star_pad = np.pad(v_star, 1, mode='constant')
    apply_boundaries(u_star_pad, v_star_pad, T_pred_pad)
    
    us_x, _, _ = calc_derivs(u_star_pad)
    _, vs_y, _ = calc_derivs(v_star_pad)
    div_star = us_x + vs_y
    
    # Enforce zeros on the boundaries of the RHS to match the matrix rows
    div_star[0, :] = 0.0
    div_star[-1, :] = 0.0
    div_star[:, 0] = 0.0
    div_star[:, -1] = 0.0
    
    rhs_arr = da.getVecArray(RHS_vec)
    rhs_arr[:] = div_star / dt 
    
    ksp.solve(RHS_vec, P_vec)
    
    # --- 4. VELOCITY CORRECTION ---
    p_arr = da.getVecArray(P_vec)[:] 
    px, py = calc_grad_p(p_arr)
    
    u = (u_star - dt * px) * (1 - mask)
    v = (v_star - dt * py) * (1 - mask)
    T = T_new

    # ==========================
    # LOGGING & PHASE DETECTION
    # ==========================
    if t % 500 == 0:
        PETSc.Sys.Print(f"Step {t} | PETSc KSP Iters: {ksp.getIterationNumber()}")

    v_probe = v[probe_x, probe_y]

    if t > 20000:
        if phase_state == 0:
            if prev_v_probe < 0 and v_probe >= 0:
                t_start = t
                phase_state = 1
        elif phase_state == 1:
            if prev_v_probe < 0 and v_probe >= 0:
                T_period_steps = t - t_start
                PETSc.Sys.Print(f"\n[PHASE LOCK] Vortex Shedding Period Detected: {T_period_steps} steps.")
                target_steps = [
                    t,                             # t = Tp
                    t + T_period_steps // 4,       # t = Tp/4
                    t + 2 * T_period_steps // 4,   # t = 2Tp/4
                    t + 3 * T_period_steps // 4    # t = 3Tp/4
                ]
                phase_state = 2
        elif phase_state == 2:
            if t in target_steps:
                idx = target_steps.index(t)
                phase_names = ["Tp", "Tp_4", "2Tp_4", "3Tp_4"]
                phase = phase_names[idx]
                
                # --- PLOT EXTRACTION ---
                plot_start_x = max(0, cor_x - int(1.0 * D_cells))
                plot_end_x = min(nx, cor_x + int(2.5 * D_cells))

                U_plot = u[plot_start_x:plot_end_x, :].T
                T_plot = T[plot_start_x:plot_end_x, :].T
                mask_plot = mask[plot_start_x:plot_end_x, :].T

                x_dim = np.linspace(44.0, 47.5, U_plot.shape[1])
                y_dim = np.linspace(0, 4, ny)
                X, Y = np.meshgrid(x_dim, y_dim)

                psi = np.zeros_like(U_plot)
                for j in range(1, ny):
                    psi[j, :] = psi[j-1, :] + 0.5 * (U_plot[j, :] + U_plot[j-1, :]) * (1.0/D_cells)

                p_map = {"Tp": "(a)", "Tp_4": "(b)", "2Tp_4": "(c)", "3Tp_4": "(d)"}
                letter = p_map.get(phase, "")
                t_txt = f"$t=T_p/4$" if phase == "Tp_4" else f"$t={phase.replace('_', '/')}$"
                if phase == "Tp": t_txt = "$t=T_p$"

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                levels_stream = sorted(list(np.linspace(1.8, 2.2, 21)) + [0.2, 0.6, 1.0, 1.4, 2.6, 3.0, 3.4, 3.8])
                axes[0].contour(X, Y, psi, levels=levels_stream, colors='black', linewidths=0.9)
                axes[0].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
                axes[0].set_xlim([44, 47.5]); axes[0].set_ylim([0, 4])
                axes[0].text(44.15, 3.75, letter, fontsize=20, fontweight='bold', va='top')
                axes[0].text(47.4, 3.75, t_txt, fontsize=18, va='top', ha='right', style='italic')

                axes[1].contour(X, Y, T_plot, levels=np.linspace(0.05, 0.95, 15), colors='red', linewidths=1.0)
                axes[1].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
                axes[1].set_xlim([44, 47.5]); axes[1].set_ylim([0, 4])
                axes[1].text(44.15, 3.75, letter, fontsize=20, fontweight='bold', va='top')
                axes[1].text(47.4, 3.75, t_txt, fontsize=18, va='top', ha='right', style='italic')

                plt.tight_layout()
                plt.savefig(f"results/Validation_{phase}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                PETSc.Sys.Print(f"--> Saved validation plot for phase: {phase}")
                saved_phases += 1
                
                if saved_phases == 4:
                    PETSc.Sys.Print("\nValidation Complete! All 4 phases extracted.")
                    sys.exit()

    prev_v_probe = v_probe