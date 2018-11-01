# Set up the whole space.
# Get an initial state.
# Pick a random point in the space
# Find the closes state to that point
# Load that state
# Take a random action
# Find new state
import random
from scipy.spatial import distance

max_iterations = 1000

snapshot_buffer = []
add_current_snapshot()

def euclidean_distance():


def pick_random_state():
    x = random.random(-1.2, 0.6)
    y = random.random(-0.07, 0.076)
    return (x,y)

for iteration in range(max_iterations):
    rand_state = pick_random_state
    closest_state = min(snapshot_buffer, key=lambda i: distance.sqeuclidean(i, rand_state))
    load_snapshot(closest_state)
    random_action = env.action_space.sample()
    env.step(random_action)
    add_current_snapshot()
