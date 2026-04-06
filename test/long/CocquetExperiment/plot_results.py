# test/long/CocquetExperiment/plot_results.py
import pyvista as pv
import matplotlib.pyplot as plt
import os

def plot_cocquet():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    vtu_file = os.path.join(results_dir, 'cocquet_flow.vtu')
    
    if not os.path.exists(vtu_file):
        print(f"VTU file not found: {vtu_file}")
        return

    mesh = pv.read(vtu_file)
    
    # Extract line across the channel center at x = 1.0
    line = mesh.sample_over_line((1.0, 0, 0), (1.0, 1.0, 0), resolution=100)
    
    y = line.points[:, 1]
    u = line['u'][:, 0]
    
    plt.figure(figsize=(6, 8))
    plt.plot(u, y, 'b-', lw=2)
    plt.xlabel('Velocity u_x')
    plt.ylabel('y')
    plt.title('Velocity Profile at x=1.0')
    plt.grid(True)
    
    plot_file = os.path.join(results_dir, 'velocity_profile.png')
    plt.savefig(plot_file)
    print(f"Velocity profile plotted and saved to {plot_file}")

if __name__ == "__main__":
    plot_cocquet()
