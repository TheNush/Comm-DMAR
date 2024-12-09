from tkinter import *
from PIL import Image, ImageTk
import threading

from environment import UMVRP
from utils import load_instance, prepare_nx_grid_graph
from visualizer import GridWorld

size = 10
seed = 1234

k = 3
psi = 2
N = 32
beta = 10

visualizer = True
padding = 20

def main():
	# while len(env.task_vertices) > 0:
	env.clustering()
	env.share_map_information()
	env.grow_moac()
	env.compute_controls()
	# print(f"Done!!! Movement Cost={env.movement_cost}; Communication Cost={env.communication_cost}")

def edit_instance(out):
	out['agnt_verts'].remove((1,0))
	out['agnt_verts'].remove((4,4))
	out['agnt_verts'].remove((5,5))

	out['agnt_verts'].append((3,1))
	out['agnt_verts'].append((1,3))

	out['task_verts'].append((5,9))
	out['task_verts'].append((2,6))

	out['task_verts'].remove((5,1))
	out['task_verts'].remove((7,9))
	out['task_verts'].remove((8,9))
	out['task_verts'].remove((4,3))

	# out['agnt_verts'].append((7,7))

	# out['task_verts'].append((3,3))	
	# out['agnt_verts'] = [(3,5),(1,5),(3,7),(1,2)]
	# out['task_verts'] = [(1,4),(4,4),(4,7),(0,6)]

	return out

if __name__ == "__main__":
	out, _ = load_instance(size, seed)

	out = edit_instance(out)

	env = UMVRP(10, prepare_nx_grid_graph(out['gridGraph'],
										out['adjList'],
										out['verts']), 
				out['agnt_verts'],
				out['task_verts'],
				k, psi, beta, N)
	if visualizer:
		world = GridWorld(size, env, main)
		env.visualizer = world
		world.run_visualizer()
	else:
		main()
