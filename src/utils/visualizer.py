import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import imageio.v2 as imageio
import io
from IPython.display import Image, display

OUTPUT_FIGURE_DIR = Path(r"C:\Users\Eren\Desktop\vs code\nature_inspired_optimization_algorithms_comperative\outputs\figures")
OUTPUT_ANIMATION_DIR = Path(r"C:\Users\Eren\Desktop\vs code\nature_inspired_optimization_algorithms_comperative\outputs\animations")

def plot_3d_function(func, bounds, title, filename, dpi=300, figsize=(14, 10), cmap='viridis'):
    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9, antialiased=True)
    
    cbar = fig.colorbar(surf, shrink=0.6, aspect=30, pad=0.05)
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
    ax.set_xlabel('Dimension 1 (X1)', fontsize=12, labelpad=15)
    ax.set_ylabel('Dimension 2 (X2)', fontsize=12, labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Fitness Cost (Z)', fontsize=12, labelpad=20, rotation=90)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect(aspect=None, zoom=0.85)

    filepath = OUTPUT_FIGURE_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close()

def plot_2d_heatmap(func, bounds, title, filename, dpi=300, figsize=(10, 8), cmap='viridis'):
    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig, ax = plt.subplots(figsize=figsize)
    
    heatmap = ax.contourf(X, Y, Z, levels=100, cmap=cmap)
    cbar = fig.colorbar(heatmap, ax=ax, shrink=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Fitness Cost (Z)', fontsize=12, labelpad=15, rotation=270)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Dimension 1 (X1)', fontsize=12, labelpad=10)
    ax.set_ylabel('Dimension 2 (X2)', fontsize=12, labelpad=10)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_aspect('equal', adjustable='box')

    filepath = OUTPUT_FIGURE_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

# --- PSO ---
class PSOVisualizer:
    def __init__(self, func, bounds, pop_size=12, max_gens=100, threshold=1e-4, w=0.4, c1=0.6, c2=0.6):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.threshold = threshold
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.pop_pos = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        self.pop_vel = np.zeros((pop_size, 2))
        self.pbest_pos = self.pop_pos.copy()
        self.pbest_fit = np.array([func(p[0], p[1]) for p in self.pop_pos])
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_fit)].copy()
        self.gbest_fit = np.min(self.pbest_fit)

        self.pos_history = []
        self.vel_history = []
        self.fit_history = []

    def solve(self):
        success = False
        final_gen = self.max_gens

        for g in range(self.max_gens):
            self.pos_history.append(self.pop_pos.copy())
            self.vel_history.append(self.pop_vel.copy())
            self.fit_history.append(self.gbest_fit)

            if self.gbest_fit <= self.threshold:
                success = True
                final_gen = g + 1
                break

            r1, r2 = np.random.rand(self.pop_size, 1), np.random.rand(self.pop_size, 1)
            self.pop_vel = (self.w * self.pop_vel +
                            self.c1 * r1 * (self.pbest_pos - self.pop_pos) +
                            self.c2 * r2 * (self.gbest_pos - self.pop_pos))
            self.pop_pos += self.pop_vel
            self.pop_pos = np.clip(self.pop_pos, self.bounds[0], self.bounds[1])

            current_fit = np.array([self.func(p[0], p[1]) for p in self.pop_pos])
            improved = current_fit < self.pbest_fit
            self.pbest_pos[improved] = self.pop_pos[improved]
            self.pbest_fit[improved] = current_fit[improved]

            if np.min(current_fit) < self.gbest_fit:
                self.gbest_fit = np.min(current_fit)
                self.gbest_pos = self.pop_pos[np.argmin(current_fit)].copy()

        return self.pos_history, self.vel_history, self.fit_history, success, final_gen

def create_pso_dashboard(func, bounds, title, filename, subtitle=None, max_gens=100, threshold=1e-3, 
                              pop_size=12, w=0.4, c1=0.6, c2=0.6, fps=8, dpi=200, display_gif=True):
    solver = PSOVisualizer(func, bounds, pop_size=pop_size, max_gens=max_gens, threshold=threshold, w=w, c1=c1, c2=c2)
    pos_h, vel_h, fit_h, success, total_gen = solver.solve()

    x = np.linspace(bounds[0], bounds[1], 300)
    y = np.linspace(bounds[0], bounds[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    frames = []
    status_text = "SUCCESS" if success else "FAILED"
    print(f"Creating Dashboard GIF: {filename} | Status: {status_text} at Gen {total_gen}")

    for gen in range(len(pos_h)):
        fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(18, 8))

        cont = ax_map.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax_map.quiver(pos_h[gen][:, 0], pos_h[gen][:, 1],
                      vel_h[gen][:, 0], vel_h[gen][:, 1],
                      color='white', alpha=0.5, angles='xy', scale_units='xy', scale=1)
        ax_map.scatter(pos_h[gen][:, 0], pos_h[gen][:, 1],
                       c='magenta', edgecolors='white', s=50, label='Particles')
        ax_map.scatter(0, 0, c='red', marker='*', s=250, edgecolors='black', label='Target')

        ax_map.set_title(f"Generation: {gen+1}/{total_gen}", fontsize=14, fontweight='bold')
        ax_map.set_xlim(bounds[0], bounds[1])
        ax_map.set_ylim(bounds[0], bounds[1])
        ax_map.legend(loc='upper right')
        fig.colorbar(cont, ax=ax_map, shrink=0.7, label='Fitness Cost')

        ax_fit.plot(range(1, gen+2), fit_h[:gen+1], color='blue', linewidth=2, marker='o', markersize=4)
        ax_fit.set_title(f"Best Fitness: {fit_h[gen]:.6f}", fontsize=14, fontweight='bold')
        ax_fit.set_xlabel("Generation")
        ax_fit.set_ylabel("Global Best Fitness")
        ax_fit.set_yscale('log')
        ax_fit.grid(True, which="both", ls="-", alpha=0.3)
        ax_fit.set_xlim(0, max_gens)
        ax_fit.set_ylim(threshold * 0.1, max(fit_h) * 10)

        main_title = title
        if subtitle:
            main_title += f" - {subtitle}"
            
        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()

    filepath = OUTPUT_ANIMATION_DIR / filename
    imageio.mimsave(filepath, frames, fps=fps, loop=0)
    
    if display_gif:
        display(Image(filename=filepath))

# --- GWO ---
class GWOVisualizer:
    def __init__(self, func, bounds, pop_size=15, max_gens=100, threshold=1e-5):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.threshold = threshold

        self.pop_pos = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        self.pop_fit = np.array([func(p[0], p[1]) for p in self.pop_pos])

        self.alpha_pos, self.beta_pos, self.delta_pos = np.zeros(2), np.zeros(2), np.zeros(2)
        self.alpha_fit, self.beta_fit, self.delta_fit = float('inf'), float('inf'), float('inf')

        self.pos_history = []
        self.fit_history = []
        self.hierarchy_history = []

    def solve(self):
        success = False
        final_gen = self.max_gens

        for g in range(self.max_gens):
            for i in range(self.pop_size):
                self.pop_pos[i] = np.clip(self.pop_pos[i], self.bounds[0], self.bounds[1])
                fit = self.func(self.pop_pos[i, 0], self.pop_pos[i, 1])
                self.pop_fit[i] = fit

                if fit < self.alpha_fit:
                    self.delta_fit, self.delta_pos = self.beta_fit, self.beta_pos.copy()
                    self.beta_fit, self.beta_pos = self.alpha_fit, self.alpha_pos.copy()
                    self.alpha_fit, self.alpha_pos = fit, self.pop_pos[i].copy()
                elif fit < self.beta_fit and fit != self.alpha_fit:
                    self.delta_fit, self.delta_pos = self.beta_fit, self.beta_pos.copy()
                    self.beta_fit, self.beta_pos = fit, self.pop_pos[i].copy()
                elif fit < self.delta_fit and fit != self.alpha_fit and fit != self.beta_fit:
                    self.delta_fit, self.delta_pos = fit, self.pop_pos[i].copy()

            self.pos_history.append(self.pop_pos.copy())
            self.fit_history.append(self.alpha_fit)
            self.hierarchy_history.append((self.alpha_pos.copy(), self.beta_pos.copy(), self.delta_pos.copy()))

            if self.alpha_fit <= self.threshold:
                success = True
                final_gen = g + 1
                break

            a = 2 - g * (2 / self.max_gens)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2), np.random.rand(2)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * self.alpha_pos - self.pop_pos[i])
                X1 = self.alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(2), np.random.rand(2)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * self.beta_pos - self.pop_pos[i])
                X2 = self.beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(2), np.random.rand(2)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * self.delta_pos - self.pop_pos[i])
                X3 = self.delta_pos - A3 * D_delta

                self.pop_pos[i] = (X1 + X2 + X3) / 3

        return self.pos_history, self.fit_history, self.hierarchy_history, success, final_gen

def create_gwo_dashboard(func, bounds, title, filename, subtitle=None, max_gens=100, threshold=1e-5,
                                      pop_size=15, fps=8, dpi=200, display_gif=True):
    solver = GWOVisualizer(func, bounds, pop_size=pop_size, max_gens=max_gens, threshold=threshold)
    pos_h, fit_h, hier_h, success, total_gen = solver.solve()

    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    frames = []
    status_text = "SUCCESS" if success else "FAILED"
    print(f"Creating GWO Dashboard GIF: {filename} | Status: {status_text} at Gen {total_gen}")

    for gen in range(len(pos_h)):
        # Standardized 1x2 layout matching PSO
        fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(18, 8))

        # Standardized to 'viridis' colormap matching PSO
        cont = ax_map.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        alpha_p, beta_p, delta_p = hier_h[gen]

        ax_map.scatter(pos_h[gen][:, 0], pos_h[gen][:, 1], c='gray', s=30, alpha=0.6, label='Omega Wolves')
        ax_map.scatter(delta_p[0], delta_p[1], c='brown', s=150, marker='^', edgecolors='black', label='Delta (3rd)')
        ax_map.scatter(beta_p[0], beta_p[1], c='orange', s=200, marker='s', edgecolors='black', label='Beta (2nd)')
        ax_map.scatter(alpha_p[0], alpha_p[1], c='gold', s=300, marker='*', edgecolors='black', label='Alpha (Leader)')

        ax_map.set_title(f"Generation: {gen+1}/{total_gen}", fontsize=14, fontweight='bold')
        ax_map.set_xlim(bounds[0], bounds[1])
        ax_map.set_ylim(bounds[0], bounds[1])
        ax_map.legend(loc='upper right')
        
        # Added colorbar matching PSO
        fig.colorbar(cont, ax=ax_map, shrink=0.7, label='Fitness Cost')

        # Fitness progression plot
        ax_fit.plot(range(1, gen+2), fit_h[:gen+1], color='darkgoldenrod', linewidth=2, marker='o', markersize=4)
        ax_fit.set_title(f"Alpha Best Fitness: {fit_h[gen]:.6f}", fontsize=14, fontweight='bold')
        ax_fit.set_xlabel("Generation")
        ax_fit.set_ylabel("Global Best Fitness")
        ax_fit.set_yscale('log')
        ax_fit.grid(True, which="both", ls="-", alpha=0.3)
        ax_fit.set_xlim(0, max_gens)
        ax_fit.set_ylim(threshold * 0.1, max(fit_h) * 10)

        main_title = title
        if subtitle:
            main_title += f" - {subtitle}"
            
        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()

    filepath = OUTPUT_ANIMATION_DIR / filename
    imageio.mimsave(filepath, frames, fps=fps, loop=0)
    
    if display_gif:
        display(Image(filename=filepath))

# --- ABC ---
class ABCVisualizer:
    def __init__(self, func, bounds, food_number=15, max_gens=100, threshold=1e-5, limit=20):
        self.func = func
        self.bounds = bounds
        self.food_number = food_number
        self.max_gens = max_gens
        self.threshold = threshold
        self.limit = limit

        self.foods = np.random.uniform(bounds[0], bounds[1], (food_number, 2))
        self.fitness = np.array([func(f[0], f[1]) for f in self.foods])
        self.trial = np.zeros(food_number)

        self.best_food = self.foods[np.argmin(self.fitness)].copy()
        self.best_fit = np.min(self.fitness)

        self.pos_history = []
        self.fit_history = []
        self.phase_history = []

    def calculate_prob(self):
        fit_norm = 1.0 / (1.0 + self.fitness)
        return fit_norm / np.sum(fit_norm)

    def solve(self):
        success = False
        final_gen = self.max_gens

        for g in range(self.max_gens):
            # YENİ: Sadece başarılı pozisyonları değil, (kaynak_indeksi, aday_pozisyon) çiftlerini saklıyoruz.
            active_employed_connections = [] 
            active_onlooker_connections = []
            # Scout için: (kaynak_indeksi, eski_pozisyon, yeni_pozisyon)
            active_scout_connections = []

            # 1. Görevli Arı Fazı
            for i in range(self.food_number):
                partner = np.random.choice([p for p in range(self.food_number) if p != i])
                phi = np.random.uniform(-1, 1, 2)
                v = self.foods[i] + phi * (self.foods[i] - self.foods[partner])
                v = np.clip(v, self.bounds[0], self.bounds[1])

                fit_v = self.func(v[0], v[1])
                if fit_v < self.fitness[i]:
                    self.foods[i] = v
                    self.fitness[i] = fit_v
                    self.trial[i] = 0
                    # YENİ: Bağlantıyı sakla (hangi kaynak 'i'den türedi, aday pozisyon 'v')
                    active_employed_connections.append((i, v)) 
                else:
                    self.trial[i] += 1

            # 2. Gözcü Arı Fazı
            probs = self.calculate_prob()
            for _ in range(self.food_number):
                i = np.random.choice(range(self.food_number), p=probs)
                partner = np.random.choice([p for p in range(self.food_number) if p != i])
                phi = np.random.uniform(-1, 1, 2)
                v = self.foods[i] + phi * (self.foods[i] - self.foods[partner])
                v = np.clip(v, self.bounds[0], self.bounds[1])

                fit_v = self.func(v[0], v[1])
                if fit_v < self.fitness[i]:
                    self.foods[i] = v
                    self.fitness[i] = fit_v
                    self.trial[i] = 0
                    # YENİ: Bağlantıyı sakla (seçilen kaynak 'i'den türedi, aday pozisyon 'v')
                    active_onlooker_connections.append((i, v))
                else:
                    self.trial[i] += 1

            # 3. Kaşif Arı Fazı
            for i in range(self.food_number):
                if self.trial[i] > self.limit:
                    old_source_pos = self.foods[i].copy() # Eskiyi sakla
                    self.foods[i] = np.random.uniform(self.bounds[0], self.bounds[1], 2)
                    self.fitness[i] = self.func(self.foods[i, 0], self.foods[i, 1])
                    self.trial[i] = 0
                    # YENİ: Kaşif bağlantısını sakla (hangi kaynak 'i' yenilendi, eski ve yeni pozisyon)
                    active_scout_connections.append((i, old_source_pos, self.foods[i].copy()))

            # En iyiyi güncelle
            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_fit:
                self.best_fit = self.fitness[min_idx]
                self.best_food = self.foods[min_idx].copy()

            # Geçmişi kaydet
            self.pos_history.append(self.foods.copy())
            self.fit_history.append(self.best_fit)
            # YENİ: Veri yapılarını güncelledik
            self.phase_history.append({
                'employed': active_employed_connections,
                'onlooker': active_onlooker_connections,
                'scout': active_scout_connections
            })

            if self.best_fit <= self.threshold:
                success = True
                final_gen = g + 1
                break

        return self.pos_history, self.fit_history, self.phase_history, success, final_gen

def create_abc_dashboard(func, bounds, title, filename, subtitle=None, max_gens=100, threshold=1e-5,
                                  food_number=15, limit=20, fps=8, dpi=200, display_gif=True):
    solver = ABCVisualizer(func, bounds, food_number=food_number, max_gens=max_gens, threshold=threshold, limit=limit)
    pos_h, fit_h, phase_h, success, total_gen = solver.solve()

    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    frames = []
    status_text = "SUCCESS" if success else "FAILED"
    print(f"Creating ABC Dashboard GIF: {filename} | Status: {status_text} at Gen {total_gen}")

    for gen in range(len(pos_h)):
        fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(18, 8))

        cont = ax_map.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        ax_map.scatter([], [], c='white', s=40, alpha=0.4, edgecolors='black', label='Food Sources')
        ax_map.scatter([], [], c='cyan', s=120, marker='o', edgecolors='black', linewidths=1.5, label='Employed Bees')
        ax_map.scatter([], [], c='magenta', s=180, marker='*', edgecolors='black', linewidths=1.5, label='Onlooker Bees')
        ax_map.scatter([], [], c='red', s=250, marker='X', edgecolors='white', linewidths=1.5, label='Scout Bees')
        
        sources_pos = pos_h[gen]
        ax_map.scatter(sources_pos[:, 0], sources_pos[:, 1], c='white', s=40, alpha=0.4, edgecolors='black')

        phases = phase_h[gen]

        for source_idx, candidate_pos in phases['employed']:
            ax_map.plot([sources_pos[source_idx, 0], candidate_pos[0]], 
                        [sources_pos[source_idx, 1], candidate_pos[1]], 
                        color='cyan', linestyle='-', linewidth=0.8, alpha=0.6)
            ax_map.scatter(candidate_pos[0], candidate_pos[1], 
                           c='cyan', s=120, marker='o', edgecolors='black', linewidths=1.5)

        for source_idx, candidate_pos in phases['onlooker']:
            ax_map.plot([sources_pos[source_idx, 0], candidate_pos[0]], 
                        [sources_pos[source_idx, 1], candidate_pos[1]], 
                        color='magenta', linestyle='-', linewidth=0.8, alpha=0.6)
            ax_map.scatter(candidate_pos[0], candidate_pos[1], 
                           c='magenta', s=180, marker='*', edgecolors='black', linewidths=1.5)

        for source_idx, old_pos, new_pos in phases['scout']:
            ax_map.plot([old_pos[0], new_pos[0]], 
                        [old_pos[1], new_pos[1]], 
                        color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            ax_map.scatter(old_pos[0], old_pos[1], c='red', s=80, marker='X', alpha=0.3, edgecolors='black', linewidths=1)
            ax_map.scatter(new_pos[0], new_pos[1], c='red', s=250, marker='X', edgecolors='white', linewidths=1.5)

        ax_map.set_title(f"Nectar Foraging Map | Gen: {gen+1}/{total_gen}", fontsize=14, fontweight='bold')
        ax_map.set_xlim(bounds[0], bounds[1])
        ax_map.set_ylim(bounds[0], bounds[1])
        ax_map.legend(loc='upper right', framealpha=0.9)
        fig.colorbar(cont, ax=ax_map, shrink=0.7, label='Fitness Cost')

        ax_fit.plot(range(1, gen+2), fit_h[:gen+1], color='darkred', linewidth=2, marker='o', markersize=4)
        ax_fit.set_title(f"Best Nectar Source (Cost): {fit_h[gen]:.6f}", fontsize=14, fontweight='bold')
        ax_fit.set_xlabel("Generation")
        ax_fit.set_ylabel("Global Best Fitness")
        ax_fit.set_yscale('log')
        ax_fit.grid(True, which="both", ls="-", alpha=0.3)
        ax_fit.set_xlim(0, max_gens)
        ax_fit.set_ylim(threshold * 0.1, max(fit_h) * 10)

        main_title = title
        if subtitle:
            main_title += f" - {subtitle}"
            
        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()

    filepath = OUTPUT_ANIMATION_DIR / filename
    imageio.mimsave(filepath, frames, fps=fps, loop=0)
    
    if display_gif:
        display(Image(filename=filepath))

# --- DE ---
class DEVisualizer:
    def __init__(self, func, bounds, pop_size=20, max_gens=100, threshold=1e-5, F=0.8, CR=0.9):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.threshold = threshold
        self.F = F
        self.CR = CR

        self.pop = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        self.fitness = np.array([func(p[0], p[1]) for p in self.pop])
        self.best_idx = np.argmin(self.fitness)
        self.best_pos = self.pop[self.best_idx].copy()
        self.best_fit = self.fitness[self.best_idx]

        self.pop_history = []
        self.fit_history = []
        self.mutation_history = []

    def solve(self):
        success = False
        final_gen = self.max_gens

        for g in range(self.max_gens):
            self.pop_history.append(self.pop.copy())
            self.fit_history.append(self.best_fit)
            
            successful_mutations = []
            failed_mutations = []

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]

                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(2) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, 2)] = True

                trial = np.where(cross_points, mutant, self.pop[i])
                trial_fit = self.func(trial[0], trial[1])

                if trial_fit < self.fitness[i]:
                    successful_mutations.append((self.pop[i].copy(), trial.copy()))
                    self.pop[i] = trial
                    self.fitness[i] = trial_fit

                    if trial_fit < self.best_fit:
                        self.best_fit = trial_fit
                        self.best_pos = trial.copy()
                else:
                    failed_mutations.append((self.pop[i].copy(), trial.copy()))

            self.mutation_history.append({
                'success': successful_mutations,
                'fail': failed_mutations
            })

            if self.best_fit <= self.threshold:
                success = True
                final_gen = g + 1
                break

        return self.pop_history, self.fit_history, self.mutation_history, success, final_gen

def create_de_dashboard(func, bounds, title, filename, subtitle=None, max_gens=100, threshold=1e-5,
                                  pop_size=20, F=0.8, CR=0.9, fps=8, dpi=200, display_gif=True):
    solver = DEVisualizer(func, bounds, pop_size=pop_size, max_gens=max_gens, threshold=threshold, F=F, CR=CR)
    pop_h, fit_h, mut_h, success, total_gen = solver.solve()

    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    frames = []
    status_text = "SUCCESS" if success else "FAILED"
    print(f"Creating DE Dashboard GIF: {filename} | Status: {status_text} at Gen {total_gen}")

    for gen in range(len(pop_h)):
        fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(18, 8))

        cont = ax_map.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        ax_map.scatter([], [], c='cyan', s=60, edgecolors='black', label='Target Vectors (Pop)')
        ax_map.scatter([], [], c='lime', marker='*', s=120, edgecolors='black', label='Successful Trial')
        ax_map.scatter([], [], c='red', marker='X', s=70, alpha=0.7, label='Failed Trial')
        
        ax_map.scatter(pop_h[gen][:, 0], pop_h[gen][:, 1], c='cyan', s=60, edgecolors='black', zorder=3)

        mutations = mut_h[gen]
        
        for old_pos, new_pos in mutations['fail']:
            ax_map.plot([old_pos[0], new_pos[0]], [old_pos[1], new_pos[1]], 
                        color='red', linestyle=':', linewidth=1, alpha=0.4)
            ax_map.scatter(new_pos[0], new_pos[1], c='red', s=70, marker='X', alpha=0.6)

        for old_pos, new_pos in mutations['success']:
            ax_map.annotate('', xy=new_pos, xytext=old_pos,
                            arrowprops=dict(arrowstyle="->", color="white", lw=1.5, alpha=0.9))
            ax_map.scatter(new_pos[0], new_pos[1], c='lime', s=120, marker='*', edgecolors='black', zorder=4)

        ax_map.set_title(f"Mutation/Crossover Space | Gen: {gen+1}/{total_gen}", fontsize=14, fontweight='bold')
        ax_map.set_xlim(bounds[0], bounds[1])
        ax_map.set_ylim(bounds[0], bounds[1])
        ax_map.legend(loc='upper right', framealpha=0.9)
        fig.colorbar(cont, ax=ax_map, shrink=0.7, label='Fitness Cost')

        ax_fit.plot(range(1, gen+2), fit_h[:gen+1], color='indigo', linewidth=2, marker='o', markersize=4)
        ax_fit.set_title(f"Global Best Fitness: {fit_h[gen]:.6f}", fontsize=14, fontweight='bold')
        ax_fit.set_xlabel("Generation")
        ax_fit.set_ylabel("Global Best Fitness")
        ax_fit.set_yscale('log')
        ax_fit.grid(True, which="both", ls="-", alpha=0.3)
        ax_fit.set_xlim(0, max_gens)
        ax_fit.set_ylim(threshold * 0.1, max(fit_h) * 10)

        main_title = title
        if subtitle:
            main_title += f" - {subtitle}"
        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()

    filepath = OUTPUT_ANIMATION_DIR / filename
    imageio.mimsave(filepath, frames, fps=fps, loop=0)
    
    if display_gif:
        display(Image(filename=filepath))

# --- ES ---
import io
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

class ESVisualizer:
    def __init__(self, func, bounds, lambd=30, max_gens=100, threshold=1e-5, sigma_ratio=0.1, k=5, update_interval=5):
        self.func = func
        self.bounds = bounds
        self.lambd = lambd
        self.max_gens = max_gens
        self.threshold = threshold
        self.k = k
        self.update_interval = update_interval

        self.parent = np.random.uniform(bounds[0], bounds[1], 2)
        self.parent_fit = func(self.parent[0], self.parent[1])

        self.sigma = (bounds[1] - bounds[0]) * sigma_ratio
        self.c = 0.817
        self.successful_mutations = 0

        self.parent_history = []
        self.offspring_history = []
        self.fit_history = []
        self.sigma_history = []
        self.mask_history = []

    def solve(self):
        success = False
        final_gen = self.max_gens

        for g in range(self.max_gens):
            self.parent_history.append(self.parent.copy())
            self.fit_history.append(self.parent_fit)
            self.sigma_history.append(self.sigma)

            if self.parent_fit <= self.threshold:
                success = True
                final_gen = g + 1
                break

            offsprings = []
            fitnesses = []

            for _ in range(self.lambd):
                z = np.random.randn(2)
                offspring = self.parent + self.sigma * z
                offspring = np.clip(offspring, self.bounds[0], self.bounds[1])

                offsprings.append(offspring)
                fitnesses.append(self.func(offspring[0], offspring[1]))

            offsprings = np.array(offsprings)
            fitnesses = np.array(fitnesses)
            self.offspring_history.append(offsprings.copy())

            best_idx = np.argmin(fitnesses)
            success_mask = fitnesses < self.parent_fit
            self.mask_history.append(success_mask)

            if fitnesses[best_idx] < self.parent_fit:
                self.parent = offsprings[best_idx].copy()
                self.parent_fit = fitnesses[best_idx]
                self.successful_mutations += 1

            if (g + 1) % self.update_interval == 0:
                ps = self.successful_mutations / (self.update_interval * self.lambd)
                if ps > 0.2:
                    self.sigma /= self.c
                elif ps < 0.2:
                    self.sigma *= self.c
                self.successful_mutations = 0

        return self.parent_history, self.offspring_history, self.fit_history, self.sigma_history, self.mask_history, success, final_gen

def create_es_dashboard(func, bounds, title, filename, subtitle=None, max_gens=100, threshold=1e-5,
                                 lambd=30, sigma_ratio=0.1, k=5, update_interval=5, fps=5, dpi=200, display_gif=True):
    solver = ESVisualizer(func, bounds, lambd=lambd, max_gens=max_gens, threshold=threshold, 
                          sigma_ratio=sigma_ratio, k=k, update_interval=update_interval)
    parent_h, off_h, fit_h, sig_h, mask_h, success, total_gen = solver.solve()

    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[0], bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    frames = []
    status_text = "SUCCESS" if success else "FAILED"
    print(f"Creating ES Dashboard GIF: {filename} | Status: {status_text} at Gen {total_gen}")

    for gen in range(len(parent_h) - 1):
        fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(18, 8))

        cont = ax_map.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        parent = parent_h[gen]
        sigma = sig_h[gen]
        offsprings = off_h[gen]
        mask = mask_h[gen]

        ax_map.scatter([], [], c='cyan', marker='P', s=200, edgecolors='black', label='Parent')
        ax_map.plot([], [], color='white', linestyle='--', linewidth=2, alpha=0.8, label='1-Sigma Range')
        ax_map.scatter([], [], c='lime', marker='*', s=120, edgecolors='black', label='Better Offspring')
        ax_map.scatter([], [], c='red', marker='X', s=50, alpha=0.6, label='Failed Offspring')
        
        circle = plt.Circle((parent[0], parent[1]), sigma, color='white', fill=False, linestyle='--', linewidth=2, alpha=0.8)
        ax_map.add_patch(circle)

        ax_map.scatter(offsprings[~mask, 0], offsprings[~mask, 1], c='red', marker='X', s=50, alpha=0.6)
        ax_map.scatter(offsprings[mask, 0], offsprings[mask, 1], c='lime', marker='*', s=120, edgecolors='black', zorder=4)
        
        ax_map.scatter(parent[0], parent[1], c='cyan', marker='P', s=250, edgecolors='black', zorder=5)

        sr = np.sum(mask) / len(mask) * 100 if len(mask) > 0 else 0
        ax_map.set_title(f"Gen: {gen+1}/{total_gen} | \u03c3: {sigma:.4f} | SR: {sr:.0f}%", fontsize=14, fontweight='bold')
        ax_map.set_xlim(bounds[0], bounds[1])
        ax_map.set_ylim(bounds[0], bounds[1])
        ax_map.legend(loc='upper right', framealpha=0.9)
        fig.colorbar(cont, ax=ax_map, shrink=0.7, label='Fitness Cost')

        ax_fit.plot(range(1, gen+2), fit_h[:gen+1], color='darkred', linewidth=2, marker='o', markersize=4)
        ax_fit.set_title(f"Global Best Fitness: {fit_h[gen]:.6f}", fontsize=14, fontweight='bold')
        ax_fit.set_xlabel("Generation")
        ax_fit.set_ylabel("Global Best Fitness")
        ax_fit.set_yscale('log')
        ax_fit.grid(True, which="both", ls="-", alpha=0.3)
        ax_fit.set_xlim(0, max_gens)
        ax_fit.set_ylim(threshold * 0.1, max(fit_h) * 10)

        main_title = title
        if subtitle:
            main_title += f" - {subtitle}"
            
        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()

    filepath = OUTPUT_ANIMATION_DIR / filename
    imageio.mimsave(filepath, frames, fps=fps, loop=0)
    
    if display_gif:
        display(Image(filename=filepath))