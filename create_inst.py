from init import *
from utils import *
import time

rows, cols, A, numTasks, k, psi, centralized, visualizer, \
wall_prob, seed, collisions, exp_strat, _, _, _, _, _, _ = getParameters()
assert rows == cols
size = rows

out = init_valid_grid(rows, cols, A, numTasks, wall_prob=wall_prob, seed=seed, colis=collisions)

offlineTrainResult = offlineTrainCent(out["verts"], out["adjList"])

## save out 
save_instance(out, size, seed, offlineTrainResult)

# inst, offlineTrainResult = load_instance(rows, seed, centralized)

