"""
For problem 2b
Use leap frog time stepping scheme w/2nd order spatial center differencing to solve given-velocity 1d advection-diffusion
    Q_t = k*Q_xx - u*Q_x, k and u given
    First time step is handled with Forward Euler (by using IC value for first step "previous" value)
    Grid edges are handled with forward (backward) differencing at the left (right) boundaries
Gaussian initial condition centered in grid
Units: Q (concentration) in g/kg, k (diffusivity) in m^2/s, u (velocity) in m/s, m (Gaussian IC mean) in m, v (Gaussian IC variance) in m^2
"""

import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation

_Q0 = 6.
_v = 200.
_k = 20.
_u = 12.
_domain_size = 200.
_m = _domain_size / 2.
_t0 = 0.
_tf = 4.

_FPS = 60
_SPEED = 0.2

output_path = pathlib.Path(__file__).parent / "results"
os.makedirs(output_path, exist_ok=True)

def gaussian_initial_condition(Q0: float=_Q0, m: float=_m, v: float=_v) -> Callable:
    def func(x: float) -> float:
        return Q0 * np.exp(-((x-m)**2)/v)
    return func

@dataclass
class Solution:
    Q: npt.ArrayLike
    x: npt.ArrayLike
    t: npt.ArrayLike

    def __post_init__(self):
        self.dx = self.x[1]-self.x[0]
        self.dt = self.t[1]-self.t[0]
        self.Q_tot = np.sum(self.Q, axis=0) * self.dx

    def animate_solution(self, saveto: str|None=None, fps: int=_FPS, speed: float=_SPEED):
        # Logic to maintain given framerate with real-time factor
        interval = max(int(1000/fps), 10)
        total_real_time = (self.t.max() - self.t.min()) / speed
        frames = int(total_real_time * 1000 / interval)
        frames = max(frames, 1)

        def get_idx(frame_number):
            # Not actually showing every timestep - decimating (staggered intervals)
            return int((frame_number / (frames - 1)) * (len(self.t) - 1)) if frames > 1 else 0

        def get_text(idx):
            return f"t={self.t[idx]:.4g} s\nQ_max={self.Q[:, idx].max():.5g}\nQ_tot={self.Q_tot[idx]:.11g}"
        
        fig, ax = plt.subplots()
        y0 = self.Q[:, 0]
        line = ax.plot(self.x, y0)[0]
        label = ax.text(1, 1, get_text(0), horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)
        ylim0_bot, ylim0_top = y0.min(), y0.max()
        ax.set(xlim=(self.x.min(), self.x.max()), ylim=(ylim0_bot, ylim0_top), xlabel="x (m)", ylabel="Q (g/kg)")
        fig.suptitle(f"Solution: dx = {self.dx:.2g} m, dt = {self.dt:.2g} s")

        def update(frame_number):
            idx = get_idx(frame_number)
            y = self.Q[:, idx]
            ax.set_ylim(min(ylim0_bot, y.min()), max(ylim0_top, y.max()))
            line.set_ydata(y)
            label.set(text=get_text(idx))
            return (line, label)
        
        anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval)
        
        if saveto:
            anim.save(output_path / saveto)
        else:
            plt.show()

    def plot_initial_condition(self, saveto: str|None=None):
        fig, ax = plt.subplots()
        y = self.Q[:, 0]
        ax.plot(self.x, y)
        ax.set(xlim=(self.x.min(), self.x.max()), ylim=(y.min(), y.max()), xlabel="x (m)", ylabel="Q (g/kg)")
        fig.suptitle(f"Initial conditions: dx = {self.dx:.2g} m\nQ_max={y.max():.5g}, Q_tot={self.Q_tot[-1]:.11g}")

        if saveto:
            plt.savefig(output_path / saveto)
        else:
            plt.show()

    def plot_final_solution(self, saveto: str|None=None, use_initial_limits: bool=True):
        fig, ax = plt.subplots()
        y = self.Q[:, -1]
        ylims = (y.min(), y.max())
        if use_initial_limits:
            y0 = self.Q[:, 0]
            ylims = (y0.min(), y0.max())
        ax.plot(self.x, y)
        ax.set(xlim=(self.x.min(), self.x.max()), ylim=ylims, xlabel="x (m)", ylabel="Q (g/kg)")
        fig.suptitle(f"Final solution (t = {self.t.max()} s): dx = {self.dx:.2g} m, dt = {self.dt:.2g} s\nQ_max={y.max():.5g}, Q_tot={self.Q_tot[-1]:.11g}")

        if saveto:
            plt.savefig(output_path / saveto)
        else:
            plt.show()

def dx_cd_2o(Q, i, dx):
    # First derivative, 2nd order central difference; 2nd order forward/backward difference at left/right edge
    if i == 0: # left edge -> FD
        return (-3*Q[i] + 4*Q[i+1] - Q[i+2])/(2*dx)
    if i == len(Q)-1: # right edge -> BD
        return (3*Q[i] - 4*Q[i-1] + Q[i-2])/(2*dx)
    return (Q[i+1] - Q[i-1])/(2*dx)

def dx2_cd_2o(Q, i, dx):
    # Second derivative, 2nd order central difference; 1st order forward/backward difference at left/right edge
    if i == 0: # left edge -> FD
        return (Q[i+2] - 2*Q[i+1] + Q[i])/(dx*dx)
    if i == len(Q)-1: # right edge -> BD
        return (Q[i] - 2*Q[i-1] + Q[i-2])/(dx*dx)
    return (Q[i+1] - 2*Q[i] + Q[i-1])/(dx*dx)

class AdvectionDiffusionSystem:
    def __init__(self, initial_condition: Callable, domain_size: float=_domain_size, k: float=_k, u: float=_u):
        self.initial_condition = initial_condition
        self.domain_size = domain_size
        self.k = k
        self.u = u

    def solve(self, dx: float, dt: float, t0: float=_t0, tf: float=_tf) -> Solution:
        x = np.arange(0., self.domain_size+dx, dx)
        t = np.arange(t0, tf+dt/2, dt) # end is exclusive, so tf+epsilon (epsilon<=dt) gives us the whole range; using epsilon=dt/2 to avoid floating-point issue

        Nx = x.shape[0]
        Nt = t.shape[0]

        Q = np.zeros((Nx, Nt))

        Q[:, 0] = self.initial_condition(x)

        for n in range(Nt-1):
            for i in range(Nx):
                Q_prev, dt_factor = (Q[i, n-1], 2*dt) if n > 0 else (Q[i, n], dt) # handle first time step correctly
                Q[i, n+1] = Q_prev + dt_factor * (
                    self.k * dx2_cd_2o(Q[:, n], i, dx) # diffusion term
                    - self.u * dx_cd_2o(Q[:, n], i, dx) # advection term
                )

        return Solution(Q=Q, x=x, t=t)

if __name__ == "__main__":
    initial_condition = gaussian_initial_condition()
    system = AdvectionDiffusionSystem(initial_condition=initial_condition)
    
    # dx = 4 m, dt = 0.01 s
    sol1 = system.solve(dx=4., dt=0.01)
    sol1.plot_initial_condition("sol1initial.png")
    sol1.plot_final_solution("sol1final.png")
    sol1.animate_solution("sol1.mp4")

    # dx = 6 m, dt = 0.01 s
    sol2 = system.solve(dx=6., dt=0.01)
    sol1.plot_initial_condition("sol2initial.png")
    sol2.plot_final_solution("sol2final.png")
    sol2.animate_solution("sol2.mp4")

    # dx = 2 m, dt = 0.01 s
    sol3 = system.solve(dx=2., dt=0.01)
    sol1.plot_initial_condition("sol3initial.png")
    sol3.plot_final_solution("sol3final.png")
    sol3.animate_solution("sol3.mp4")

    # try other combinations
    sol4 = system.solve(dx=2., dt=0.001)
    sol4.plot_final_solution("sol4final.png")
    sol4.animate_solution("sol4.mp4")
