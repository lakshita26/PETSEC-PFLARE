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

D_cells = 20                  
dx = 1.0 / D_cells            
dt = 0.0025  # Extremely safe RK2 stability limit             

beta = 0.25                   
H_cells = int(D_cells / beta) # ny = 80
Xu_cells = int(45 * D_cells)  # Upstream = 45D
Xd_cells = int(15 * D_cells)  # Downstream = 15D

nx = Xu_cells + Xd_cells      # 1200 cells
ny = H_cells                  # 80 cells
ntime = 30000                 

cor_x = Xu_cells             
cor_y = int(ny / 2)          
radius = int(D_cells / 2)    

os.makedirs("results", exist_ok=True)

# ==========================
# DOMAIN INITIALIZATION
# ==========================
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))
T = np.zeros((nx, ny))

np.random.seed(42)
u += 1e-3 * np.random.randn(nx, ny)

# Brinkman Penalization Body Mask
mask = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        if (i - cor_x)**2 + (j - cor_y)**2 <= radius**2 and i <= cor_x:
            mask[i, j] = 1.0

penalize_denom = 1.0 + dt * mask * 1e5

# ==========================
# STRICT PETSc MATRIX SETUP
# ==========================
da = PETSc.DMDA().create([nx, ny], dof=1, stencil_width=1, stencil_type=PETSc.DMDA.StencilType.STAR)
A = da.createMatrix()

row = PETSc.Mat.Stencil()
col = PETSc.Mat.Stencil()
(xs, xe), (ys, ye) = da.getRanges()

# Safe Matrix Assembly
for i in range(xs, xe):
    for j in range(ys, ye):
        row.index = (i, j); row.field = 0
        
        if i == nx - 1:
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
        elif i == 0:
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
            col.index = (1, j); col.field = 0; A.setValueStencil(row, col, -1.0)
        elif j == 0:
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
            col.index = (i, 1); col.field = 0; A.setValueStencil(row, col, -1.0)
        elif j == ny - 1:
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, 1.0)
            col.index = (i, ny - 2); col.field = 0; A.setValueStencil(row, col, -1.0)
        else:
            col.index = (i, j); col.field = 0; A.setValueStencil(row, col, -4.0 / dx**2)
            col.index = (i-1, j); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)
            col.index = (i+1, j); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)
            col.index = (i, j-1); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)
            col.index = (i, j+1); col.field = 0; A.setValueStencil(row, col, 1.0 / dx**2)

A.assemble()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.GAMG) 
ksp.setTolerances(rtol=1e-5)

P_vec = da.createGlobalVec()
RHS_vec = da.createGlobalVec()

# ==========================
# BULLETPROOF MATH UTILITIES
# ==========================
def apply_bcs(u_field, v_field, T_field, step):
    ramp = min(1.0, step / 1000.0) 
    y_idx = np.arange(ny)
    u_prof = 1.5 * ramp * (1.0 - (1.0 - 2.0 * y_idx / (ny - 1))**2)
    
    u_field[0, :] = u_prof
    v_field[0, :] = 0.0
    T_field[0, :] = 0.0
    
    u_field[:, 0] = 0.0; v_field[:, 0] = 0.0; T_field[:, 0] = T_field[:, 1]
    u_field[:, -1] = 0.0; v_field[:, -1] = 0.0; T_field[:, -1] = T_field[:, -2]
    
    u_field[-1, :] = u_field[-2, :]
    v_field[-1, :] = v_field[-2, :]
    T_field[-1, :] = T_field[-2, :]
    return u_field, v_field, T_field

def get_cen_diff(f):
    f_pad = np.pad(f, 1, mode='edge')
    fx = (f_pad[2:, 1:-1] - f_pad[:-2, 1:-1]) / (2 * dx)
    fy = (f_pad[1:-1, 2:] - f_pad[1:-1, :-2]) / (2 * dx)
    return fx, fy

def calc_laplacian(f):
    f_pad = np.pad(f, 1, mode='edge')
    return (f_pad[2:, 1:-1] + f_pad[:-2, 1:-1] + f_pad[1:-1, 2:] + f_pad[1:-1, :-2] - 4*f) / dx**2

# ==========================
# MAIN TIME LOOP
# ==========================
print(f"Starting Fully Stable PETSc CFD | Domain: {nx}x{ny} | Re={Re}...")

probe_x, probe_y = cor_x + int(1.5 * D_cells), cor_y + 3
phase_state, t_start, T_period_steps, saved_phases, prev_v_probe = 0, 0, 0, 0, 0.0
target_steps = []

u, v, T = apply_bcs(u, v, T, 0)

for t in range(ntime):
    
    # --- 1. RK2 PREDICTOR ---
    ux, uy = get_cen_diff(u); vx, vy = get_cen_diff(v); Tx, Ty = get_cen_diff(T)
    lap_u = calc_laplacian(u); lap_v = calc_laplacian(v); lap_T = calc_laplacian(T)
    
    u_p = u + 0.5 * dt * (-u*ux - v*uy + nu * lap_u)
    v_p = v + 0.5 * dt * (-u*vx - v*vy + nu * lap_v)
    T_p = T + 0.5 * dt * (-u*Tx - v*Ty + (1.0 / (Re * Pr)) * lap_T)
    
    u_p, v_p, T_p = apply_bcs(u_p, v_p, T_p, t)
    u_p /= penalize_denom; v_p /= penalize_denom; T_p[mask == 1] = 1.0
    
    # --- 2. RK2 CORRECTOR ---
    ux_p, uy_p = get_cen_diff(u_p); vx_p, vy_p = get_cen_diff(v_p); Tx_p, Ty_p = get_cen_diff(T_p)
    lap_u_p = calc_laplacian(u_p); lap_v_p = calc_laplacian(v_p); lap_T_p = calc_laplacian(T_p)
    
    u_star = u + dt * (-u_p*ux_p - v_p*uy_p + nu * lap_u_p)
    v_star = v + dt * (-u_p*vx_p - v_p*vy_p + nu * lap_v_p)
    T_new  = T + dt * (-u_p*Tx_p - v_p*Ty_p + (1.0 / (Re * Pr)) * lap_T_p)
    
    u_star, v_star, T_new = apply_bcs(u_star, v_star, T_new, t)
    u_star /= penalize_denom; v_star /= penalize_denom; T_new[mask == 1] = 1.0
    
    # --- 3. DIVERGENCE & POISSON ---
    us_x, _ = get_cen_diff(u_star)
    _, vs_y = get_cen_diff(v_star)
    div_star = us_x + vs_y
    
    div_star[0, :] = 0.0; div_star[-1, :] = 0.0
    div_star[:, 0] = 0.0; div_star[:, -1] = 0.0
    div_star[mask == 1] = 0.0 
    
    # Safely load into PETSc Vector
    rhs_arr = da.getVecArray(RHS_vec)
    rhs_arr[:, :] = div_star / dt
    
    ksp.solve(RHS_vec, P_vec)
    
    p_arr = da.getVecArray(P_vec)
    p = np.array(p_arr[:, :])
    
    # --- 4. CORRECTION ---
    px, py = get_cen_diff(p)
    px[-1, :] = px[-2, :] # Stabilize outlet gradient
    
    u_new = u_star - dt * px
    v_new = v_star - dt * py
    
    u_new, v_new, T_new = apply_bcs(u_new, v_new, T_new, t)
    u_new /= penalize_denom
    v_new /= penalize_denom
    
    u, v, T = u_new, v_new, T_new

    # ==========================
    # LOGGING & PHASE DETECTION
    # ==========================
    if t % 500 == 0:
        print(f"Step {t} | PETSc Iters: {ksp.getIterationNumber()} | Max V Velocity: {v.max():.4f}")

    v_probe = v[probe_x, probe_y]

    if t > 8000:
        if phase_state == 0:
            if prev_v_probe < 0 and v_probe >= 0:
                t_start = t
                phase_state = 1
        elif phase_state == 1:
            if prev_v_probe < 0 and v_probe >= 0:
                T_period_steps = t - t_start
                print(f"\n[PHASE LOCK] Vortex Shedding Period Detected: {T_period_steps} steps.")
                target_steps = [
                    t,                             
                    t + T_period_steps // 4,       
                    t + 2 * T_period_steps // 4,   
                    t + 3 * T_period_steps // 4    
                ]
                phase_state = 2
        elif phase_state == 2:
            if t in target_steps:
                idx = target_steps.index(t)
                phase_names = ["Tp", "Tp_4", "2Tp_4", "3Tp_4"]
                phase = phase_names[idx]
                
                print(f"--> Extracting validation graphs for {phase}...")
                plot_start_x = max(0, cor_x - int(1.0 * D_cells))
                plot_end_x = min(nx, cor_x + int(4.0 * D_cells))

                U_plot = u[plot_start_x:plot_end_x, :].T
                V_plot = v[plot_start_x:plot_end_x, :].T
                T_plot = T[plot_start_x:plot_end_x, :].T
                mask_plot = mask[plot_start_x:plot_end_x, :].T

                x_dim = np.linspace(44.0, 49.0, U_plot.shape[1])
                y_dim = np.linspace(0, 4, ny)
                X, Y = np.meshgrid(x_dim, y_dim)

                dy_val = 1.0 / D_cells
                U_masked = U_plot * (1 - mask_plot)
                psi = np.zeros_like(U_masked)
                for j in range(1, ny):
                    psi[j, :] = psi[j-1, :] + 0.5 * (U_masked[j, :] + U_masked[j-1, :]) * dy_val

                p_map = {"Tp": "(a)", "Tp_4": "(b)", "2Tp_4": "(c)", "3Tp_4": "(d)"}
                letter = p_map.get(phase, "")
                t_txt = f"$t=T_p/4$" if phase == "Tp_4" else f"$t={phase.replace('_', '/')}$"
                if phase == "Tp": t_txt = "$t=T_p$"

                # 1. Streamlines & Isotherms
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                levels_stream = sorted(list(np.linspace(1.8, 2.2, 21)) + [0.2, 0.6, 1.0, 1.4, 2.6, 3.0, 3.4, 3.8])
                axes[0].contour(X, Y, psi, levels=levels_stream, colors='black', linewidths=0.9)
                axes[0].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
                axes[0].set_xlim([44, 49]); axes[0].set_ylim([0, 4])
                axes[0].text(44.15, 3.75, letter, fontsize=20, fontweight='bold', va='top')
                axes[0].text(48.8, 3.75, t_txt, fontsize=18, va='top', ha='right', style='italic')
                axes[0].set_title("Streamlines")

                axes[1].contour(X, Y, T_plot, levels=np.linspace(0.05, 0.95, 15), colors='red', linewidths=1.0)
                axes[1].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
                axes[1].set_xlim([44, 49]); axes[1].set_ylim([0, 4])
                axes[1].text(44.15, 3.75, letter, fontsize=20, fontweight='bold', va='top')
                axes[1].text(48.8, 3.75, t_txt, fontsize=18, va='top', ha='right', style='italic')
                axes[1].set_title("Isotherms")

                plt.tight_layout()
                plt.savefig(f"results/Validation_{phase}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Contours
                Mag = np.sqrt(U_plot**2 + V_plot**2)
                fig, axes = plt.subplots(2, 1, figsize=(8, 8))
                
                cf1 = axes[0].contourf(X, Y, Mag, levels=40, cmap='viridis')
                axes[0].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
                fig.colorbar(cf1, ax=axes[0], label='Velocity Magnitude')
                axes[0].set_title(f"Velocity Contour ({t_txt})")
                
                cf2 = axes[1].contourf(X, Y, T_plot, levels=40, cmap='hot')
                axes[1].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
                fig.colorbar(cf2, ax=axes[1], label='Temperature')
                axes[1].set_title(f"Temperature Contour ({t_txt})")

                plt.tight_layout()
                plt.savefig(f"results/Contours_{phase}.png", dpi=200, bbox_inches='tight')
                plt.close()
                
                saved_phases += 1
                if saved_phases == 4:
                    print("\nSuccess: Validation Complete! All phases extracted perfectly.")
                    sys.exit()

    prev_v_probe = v_probe