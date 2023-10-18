import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from numba import jit

# Constants
WIDTH, HEIGHT = 800, 600
GRAVITY = np.array([0, 0.1])
N_PARTICLES = 50
RADIUS = 2
SMOOTHING_RADIUS = 15
PRESSURE_CONSTANT = 0.1  
VISCOSITY_CONSTANT = 0.01 
TIME_STEP = 2


grid = {}

particle_dtype = np.dtype([
    ('position', float, 2),
    ('velocity', float, 2),
    ('color', float, 3),
    ('density', float),
    ('pressure', float)
])

particles = np.zeros(N_PARTICLES, dtype=particle_dtype)
for p in particles:
    p['position'] = np.array([np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT)])
    p['velocity'] = np.array([np.random.uniform(-50, 50), np.random.uniform(-50, 50)])
    p['color'] = [0, 0, 1]
    p['density'] = 1.0
    p['pressure'] = 0.0

# Rest of your code...


def bin_particles(particles):
    global grid
    grid = {}
    for particle in particles:
        x, y = particle['position']
        bin_x = int(x / SMOOTHING_RADIUS)
        bin_y = int(y / SMOOTHING_RADIUS)
        
        if (bin_x, bin_y) not in grid:
            grid[(bin_x, bin_y)] = []
        grid[(bin_x, bin_y)].append(particle)



def render(particles):
    glBegin(GL_POINTS)
    for particle in particles:
        glColor3fv(particle['color'])
        glVertex2fv(particle['position'])
    glEnd()


def neighbors(particle):
    x, y = particle['position']
    bin_x = int(x / SMOOTHING_RADIUS)
    bin_y = int(y / SMOOTHING_RADIUS)

    neighbor_particles = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            bin = grid.get((bin_x + i, bin_y + j), [])
            neighbor_particles.extend(bin)
    
    return neighbor_particles


@jit(nopython=True)
def gradient_poly6_kernel(r, h):
    return np.array([0.0, 0.0])  # Placeholder


@jit(nopython=True)
def poly6_kernel(r, h):
    if 0 <= r <= h:
        return 315 / (64 * np.pi * h**9) * (h**2 - r**2)**3
    else:
        return 0

def compute_density_pressure(particles):
    bin_particles(particles)
    for particle in particles:
        density = 0
        for other in neighbors(particle):
            if other is not particle:
                r = particle['position'] - other['position']  # <-- Change here
                distance = np.linalg.norm(r)
                if 0 < distance < SMOOTHING_RADIUS:
                    density += other['density'] * poly6_kernel(distance, SMOOTHING_RADIUS)  # <-- Change here
        density = max(density, 1e-4)
        density0 = 1.0
        particle['pressure'] = PRESSURE_CONSTANT * (density - density0)  # <-- Change here



def compute_forces(particles):
    BOUNDARY_DAMPING = 0.5
    WALL_MIN = np.array([0.0, 0.0]) # Assuming the bottom-left corner of your simulation space is (0, 0)
    WALL_MAX = np.array([1.0, 1.0]) # Assuming the top-right corner of your simulation space is (1, 1)
    
    for i in range(len(particles)):
        pressure_force = np.array([0.0, 0.0])
        viscosity_force = np.array([0.0, 0.0])

        for j in range(len(particles)):
            if i != j:
                r = particles['position'][i] - particles['position'][j]
                distance = np.linalg.norm(r)
                if 0 < distance < SMOOTHING_RADIUS:
                    pressure_gradient = gradient_poly6_kernel(r, SMOOTHING_RADIUS)
                    pressure_force += pressure_gradient * (particles['pressure'][i] + particles['pressure'][j]) / 2
                    viscosity_force += VISCOSITY_CONSTANT * (particles['velocity'][j] - particles['velocity'][i])
        
        net_force = pressure_force + viscosity_force + GRAVITY
        particles['velocity'][i] += net_force * TIME_STEP / max(particles['density'][i], 1e-4)

        # Handle wall reflection
        next_position = particles['position'][i] + particles['velocity'][i] * TIME_STEP
        for dim in range(2):
            if next_position[dim] < WALL_MIN[dim] or next_position[dim] > WALL_MAX[dim]:
                particles['velocity'][i][dim] = -particles['velocity'][i][dim] * BOUNDARY_DAMPING
        
        MAX_VELOCITY = 1.0
        particles['velocity'][i] = np.clip(particles['velocity'][i], -MAX_VELOCITY, MAX_VELOCITY)
        particles['position'][i] += particles['velocity'][i] * TIME_STEP





# Pygame and PyOpenGL Initialization
pygame.init()
display = (WIDTH, HEIGHT)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
glViewport(0, 0, WIDTH, HEIGHT)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, WIDTH, HEIGHT, 0, -1, 1) # Orthographic projection
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()



while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPointSize(RADIUS)  # Set the point size for drawing

    compute_density_pressure(particles)
    compute_forces(particles)

    # Update particle positions
    for particle in particles:
        particle['position'] += particle['velocity'] * TIME_STEP

    render(particles)

    pygame.display.flip()
    pygame.time.wait(10)





# Particle Class Probably not needed bruh
class Particle:
    def __init__(self):
        self.position = np.array([np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT)], dtype=float)  # Random initial position
        self.velocity = np.array([np.random.uniform(-50, 50), np.random.uniform(-50, 50)], dtype=float)

        self.color = [0, 0, 1]
        self.density = 1.0
        self.pressure = 0.0


    def update(self):
        self.bounce_off_walls()
        self.update_color()

    def bounce_off_walls(self):
        if self.position[0] - RADIUS < 0 or self.position[0] + RADIUS > WIDTH:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] - RADIUS < 0 or self.position[1] + RADIUS > HEIGHT:
            self.velocity[1] = -self.velocity[1]


    def update_color(self):
        speed = np.linalg.norm(self.velocity)
        normalized_speed = min(speed / 10.0, 1)

        if normalized_speed < 0.25:
            t = normalized_speed / 0.25
            self.color = [(1 - t) * 0, t * 1, 1]
        elif normalized_speed < 0.5:
            t = (normalized_speed - 0.25) / 0.25
            self.color = [0, 1, (1 - t) * 1]
        elif normalized_speed < 0.75:
            t = (normalized_speed - 0.5) / 0.25
            self.color = [t * 1, (1 - t) * 1, 0]
        else:
            t = (normalized_speed - 0.75) / 0.25
            self.color = [1, t * 1, 0]



