import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sim import simulate_step
 # Import the compiled Cython module

# Define some constants
NUM_PARTICLES = 500
WIDTH, HEIGHT = 5, 5
DT = 0.02  # time step size

# Create initial particles
particles = sim.init_particles(NUM_PARTICLES, WIDTH, HEIGHT)


# Initialize particle positions and velocities randomly
np.random.seed(42)
particles['x'] = np.random.uniform(0, WIDTH, size=NUM_PARTICLES)
particles['y'] = np.random.uniform(0, HEIGHT, size=NUM_PARTICLES)
particles['vx'] = np.random.uniform(-1, 1, size=NUM_PARTICLES)
particles['vy'] = np.random.uniform(-1, 1, size=NUM_PARTICLES)

fig, ax = plt.subplots()
scat = ax.scatter(particles['x'], particles['y'], s=10)

def update(frame):
    particle_sim.simulate_step(particles)
    scat.set_offsets(np.c_[particles['x'], particles['y']])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=range(200), blit=True, repeat=True)
plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)
plt.show()
