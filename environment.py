import sys
import random
import time
import math
import networkx as nx
from copy import deepcopy
from tkinter import Label

from utils import load_instance, prepare_nx_grid_graph
from agent import Agent

random.seed(12345)

class UMVRP:
	def __init__(self, size, grid_graph, agent_vertices, task_vertices, k, psi, beta, N):
		self.size = size

		self.grid_graph = grid_graph
		self.agent_vertices = agent_vertices

		self.agents = [Agent(i, position, self) for i, position in enumerate(self.agent_vertices)]
		self.task_vertices = task_vertices
		self.view_radius = k
		self.N = N
		self.psi = psi
		self.beta = beta

		self.movement_cost = 0
		self.exploration_cost = 0
		self.wait_cost = 0
		self.communication_cost = 0

		self.visualizer = None

		print(self.task_vertices)
		# for agent in self.agents: 
		# 	self.sense(agent, k)
		# 	print(agent.ID, agent.position(), agent.view['view_tasks'])

		self.wait_time = 1.0

	def sense(self, agent, k):
		# print(agent.ID)
		view_vertices = self.get_view_vertices(agent, k)

		global_subgraph = self.grid_graph.subgraph(view_vertices)
		view_subgraph = nx.relabel_nodes(global_subgraph, lambda x: (x[0]-agent.pos_x,
																	 x[1]-agent.pos_y))
		view_agents, view_tasks = [], []
		for a in self.agents:
			if (a.pos_x, a.pos_y) in view_vertices:
				# print("\t", a.ID, a.pos_x, a.pos_y)
				# view_agents.append((a.ID, [a.pos_x-agent.pos_x,a.pos_y-agent.pos_y], a.messages))
				view_agents.append(a)
		for task in self.task_vertices:
			# if nx.has_path(global_subgraph, source=agent.position(), target=task):
			# 	print(nx.shortest_path(global_subgraph, source=agent.position(), target=task))
			if task in view_vertices:
				# print(agent.position(), task, nx.has_path(global_subgraph, source=agent.position(), target=task))
				if nx.has_path(global_subgraph, source=agent.position(), target=task):
					view_tasks.append((task[0]-agent.pos_x, task[1]-agent.pos_y))

		agent.view = {'view_graph': view_subgraph, 
					'view_agents': view_agents,
					'view_tasks': view_tasks}

	def generate_offset_view(self, agent, offset):
		offset_x, offset_y = offset
		subcluster_graph = nx.relabel_nodes(agent.view['view_graph'], lambda x: 
												(x[0]-offset_x,x[1]-offset_y))
		subcluster_tasks = []
		for task in agent.view['view_tasks']:
			subcluster_tasks.append((task[0]-offset_x, task[1]-offset_y))

		subcluster_view = {'view_graph': subcluster_graph, 
						   'view_tasks': subcluster_tasks}
		return subcluster_view

	def get_legal_moves_from(self, position):
	    valid_states = nx.neighbors(self.grid_graph, position)
	    moves = []
	    for state in valid_states:
	    	if state == (position[0], position[1]):
	    		moves.append('q')
	    	elif state == (position[0]-1, position[1]):
	    		moves.append('u')
	    	elif state == (position[0]+1, position[1]):
	    		moves.append('d')
	    	elif state == (position[0], position[1]-1):
	    		moves.append('l')
	    	elif state == (position[0], position[1]+1):
	    		moves.append('r')
	    return moves

	def get_nearest_task(self, graph, source, target_set):
		assert source in graph.nodes()
		min_dist = float('inf')
		min_path = None
		ind = -1
		for i, vertex in enumerate(target_set):
			assert vertex in graph.nodes()
			path = nx.shortest_path(graph, source=source, target=vertex)
			if len(path) < min_dist:
				min_dist = len(path)
				min_path = path
				ind = i
		return min_path

	def get_greedy_trajectory(self, agent):
		moves = []
		min_path = self.get_nearest_task(agent.view['view_graph'], 
											(0,0), agent.view['view_tasks'])
		min_path.pop(0)
		temp_pos = (0,0)
		while len(min_path) != 0:
			next_state = min_path.pop(0)
			if next_state == (temp_pos[0]-1, temp_pos[1]):
				moves.append('u')
			elif next_state == (temp_pos[0]+1, temp_pos[1]):
				moves.append('d')
			elif next_state == (temp_pos[0], temp_pos[1]-1):
				moves.append('l')
			elif next_state == (temp_pos[0], temp_pos[1]+1):
				moves.append('r')
			temp_pos = next_state

		return moves

	def get_exploration_move(self, agent):
		legal = self.get_legal_moves_from(agent.position())
		if len(legal) > 1:
			legal.remove('q')
		return random.choice(legal)

	def multiagent_rollout(self, graph, agents, tasks, agent_ID):
		current_pos = deepcopy(agents)
		current_tasks = tasks.copy()

		min_cost = float('inf')
		best_move = None
		q_factors = []
		prev_cost = 0

		agent_pos = agents[agent_ID]

		assert agent_pos in graph.nodes

		# print(agent_pos, list(graph.neighbors(agent_pos)))
		for next_state in graph.neighbors(agent_pos):
			# print("Considering move... ", next_state)
			temp_current_tasks = current_tasks.copy()
			temp_current_pos = deepcopy(current_pos)
			# print("Current pos... ", temp_current_pos)

			if next_state != agent_pos:
				cost = prev_cost + 1
			else:
				cost = prev_cost + 0.1

			temp_current_pos[agent_ID] = next_state
			if next_state in temp_current_tasks:
				temp_current_tasks.remove(next_state)
			flag = False
			for a_ID in temp_current_pos:
				if len(temp_current_tasks) == 0:
					break
				if flag == True:
					path = self.get_nearest_task(graph, temp_current_pos[a_ID], temp_current_tasks)
					dist = len(path)

					temp_current_pos[a_ID] = path[1]
					cost += 1

					if temp_current_pos[a_ID] in temp_current_tasks:
						temp_current_tasks.remove(temp_current_pos[a_ID])

				else:
					flag = True if (a_ID == agent_ID) else False
			# print(agent_pos, next_state)
			# for a_ID in temp_current_pos:
			# 	print(temp_current_pos[a_ID], current_pos[a_ID])
			while len(temp_current_tasks) > 0:
				for a_ID in agents:
					path = self.get_nearest_task(graph, temp_current_pos[a_ID], temp_current_tasks)
					dist = len(path)					

					# if next_state == agent_pos:
					# 	print("\t", a_ID, temp_current_pos[a_ID], path[1], cost)	
					temp_current_pos[a_ID] = path[1]
					cost += 1

					if temp_current_pos[a_ID] in temp_current_tasks:
						temp_current_tasks.remove(temp_current_pos[a_ID])
					if len(temp_current_tasks) == 0:
						break

			if cost < min_cost:
				min_cost = cost
				best_move = next_state

			q_factors.append((next_state, cost))
			del temp_current_pos
			del temp_current_tasks

		assert best_move != None

		for factor in q_factors:
			if factor[1] == min_cost:
				best_move = factor[0]
				if factor[0] == agent_pos: ## wait move
					break

		if best_move == (agent_pos[0]+1,agent_pos[1]):
			ret = 'd'
		elif best_move == (agent_pos[0]-1,agent_pos[1]):
			ret = 'u'
		elif best_move == (agent_pos[0],agent_pos[1]+1):
			ret = 'r'
		elif best_move == (agent_pos[0],agent_pos[1]-1):
			ret = 'l'
		else:
			ret = 'q'

		return ret, min_cost

	def cluster_multiagent_rollout(self, leader):
		temp_tasks = leader.cluster_view['view_tasks'].copy()
		all_prev_moves = {}
		for a in leader.cluster_view['view_agents']:
			all_prev_moves[a.ID] = []
		cluster_agents = {a.ID: a.relative_position(leader.position()) for a in leader.cluster_view['view_agents']}
		print(cluster_agents, temp_tasks)

		while(len(temp_tasks)) > 0:
			wait_agents = []
			prev_moves = {}
			for a_ID in cluster_agents:
				agent_pos = cluster_agents[a_ID]
				move, c = self.multiagent_rollout(leader.cluster_view['view_graph'],
													cluster_agents, temp_tasks, 
													a_ID)
				# print(move, c)

				prev_moves[a_ID] = move
				all_prev_moves[a_ID].append(move)

				if move == 'u':
					cluster_agents[a_ID] = (agent_pos[0]-1, agent_pos[1])
				elif move == 'd':
					cluster_agents[a_ID] = (agent_pos[0]+1, agent_pos[1])
				elif move == 'l':
					cluster_agents[a_ID] = (agent_pos[0], agent_pos[1]-1)
				elif move == 'r':
					cluster_agents[a_ID] = (agent_pos[0], agent_pos[1]+1)
				else:
					cluster_agents[a_ID] = (agent_pos[0], agent_pos[1])
					wait_agents.append(a_ID)
					pass

				if cluster_agents[a_ID] in temp_tasks:
					temp_tasks.remove(cluster_agents[a_ID])

				if len(temp_tasks) == 0:
					break

		longest = 0
		for a_ID in all_prev_moves:
			if len(all_prev_moves[a_ID]) > longest:
				longest = len(all_prev_moves[a_ID])

		for a_ID in all_prev_moves:
			while len(all_prev_moves[a_ID]) < longest:
				all_prev_moves[a_ID].append('q')

		return all_prev_moves

	def get_view_vertices(self, agent, k):
		view_vertices = []
		for vertex in self.grid_graph.nodes:
			manh_dist = abs(agent.pos_x-vertex[0]) + abs(agent.pos_y-vertex[1])
			if manh_dist <= k:
				view_vertices.append(vertex)
		return view_vertices

	def clustering(self):
		## Sense views
		for a in self.agents:
			self.sense(a, self.view_radius)
		#### Make sure that agents that can't reach each other are not in the same cluster
		####
		#### -- Phase 1 -- Raise Flags
		####
		for agent in self.agents:
			agent.raise_task_flag(mode=0)
		self.synchronize_timelines()

		# #### -- Phase 1.5 -- Begin clusters
		# for agent in self.agents:
		# 	if agent.messages['task_flag'] >= 2:
		# 		agent.messages['clusterID'].append(agent.ID)

		#### -- Phase 2 -- Begin Clustering
		for agent in self.agents:
			if agent.messages['task_flag'] >= 2:
				flag = True

				for neighbor in agent.view['view_agents']:
					if len(neighbor.messages['clusterID']) > 0 and neighbor != agent:
						flag = False

				if flag:
					agent.messages_prime['clusterID'].append(agent.ID)
					agent.parent_prime = None
					agent.needs_merge = True

		self.synchronize_timelines()

		#### -- Phase 3 -- Round Robin
		for a in self.agents:
			if len(a.get_cluster()) > 0:
				for x in a.view['view_agents']:
					if len(x.get_cluster()) > 0 and x != a:
						if x.ID > a.ID:
							a.messages_prime['clusterID'] = []
							a.children_prime = []
							a.parent_prime = None

							a.needs_merge = True

		self.synchronize_timelines()

		## Update colors for centroids... 
		for a in self.agents:
			if len(a.get_cluster()) > 0 and a.ID == a.messages['clusterID'][0]:
				a.color_prime = self.visualizer.colors.pop(0)
				a.gui_split = True
				a.needs_merge = True

		self.synchronize_timelines()

		for _ in range(self.psi):
			#### -- Phase 4 -- Grow clusters
			for a in self.agents:
				if a.messages['task_flag'] >= 2 and len(a.get_cluster())==0:
					for x in a.view['view_agents']:
						if len(x.get_cluster()) > 0 and x != a:
							a.messages_prime['clusterID'].append(x.get_cluster()[0])
							a.parent_prime = x
							a.color_prime = x.color
							a.gui_split = True
							a.needs_merge = True

							if a not in x.children and a != x:
								x.children_prime.append(a)
								x.needs_merge = True

			self.synchronize_timelines()

			#### -- Phase 5 -- Super Cluster formation
			for a in self.agents:
				a.messages['clusterID'] = list(set(a.messages['clusterID']))
				a.messages_prime['clusterID'] = list(set(a.messages_prime['clusterID']))
				if len(a.messages['clusterID']) > 1:
					a.messages_prime['clusterID'] = [a.ID]
					a.messages_prime['super'] = True
					a.parent_prime = None
					a.color_prime = self.visualizer.colors.pop(0)
					a.gui_split = True
					a.needs_merge = True

					for x in a.view['view_agents']:
						if x.messages['task_flag'] >= 2 and x.messages['super'] == False and x != a:
							x.messages_prime['clusterID'] = [a.ID]
							x.messages_prime['super'] = True
							x.parent_prime = a
							x.color_prime = a.color_prime
							x.gui_split = True

							if x not in a.children:
								a.children_prime.append(x)
								a.needs_merge = True
							x.children_prime = []
							x.needs_merge = True

			self.synchronize_timelines()

			#### -- Phase 6 -- Propagate super cluster information
			for _ in range(2,self.N):
				for a in self.agents:
					if a.messages['super'] == True:
						for x in a.view['view_agents']:
							if x.messages['task_flag'] >= 2 and x.messages['super'] == False and x != a:
								x.messages_prime['clusterID'] = [a.get_cluster()[0]]
								x.messages_prime['super'] = True
								x.parent_prime = a
								x.color_prime = a.color
								x.gui_split = True

								x.needs_merge = True

								if x not in a.children:
									a.children_prime.append(x)
								x.children_prime = []

				self.synchronize_timelines()

			#### -- Phase 7 -- Reset message flag
			for a in self.agents:
				a.messages['super'] = False
				a.messages_prime['super'] = False

			#### -- Phase 8 -- remove any stray children (should not be necessary if everything is correct)
			for a in self.agents:
				if a.parent != None and len(a.get_cluster()) > 0:
					for b in a.view['view_agents']:
						if b!=a and b!=a.parent:
							if a in b.children:
								b.children.remove(a)

		#### -- Phase 9 -- Solo Clusters (other methods need to be implemented)
		for a in self.agents:
			if a.messages['task_flag'] == 1:
				a.messages_prime['clusterID'] = [a.ID]
				a.color_prime = self.visualizer.colors.pop(0)
				a.gui_split = True
				a.needs_merge = True
		self.synchronize_timelines()
		
		print("End of MOAC: ")
		print("-----------------------------------------------")
		for a in self.agents:
			print(a.ID, a.position(), a.messages, a.messages_prime, a.print_children(), a.print_parent())
		print()

	def grow_moac(self):
		#### -- Phase 10 -- Solo cluster agents search their view for flag >= 2 agents
		#### Accumulate requests
		for a in self.agents:
			if a.messages['task_flag'] == 1:
				for b in a.view['view_agents']:
					if b.messages['task_flag'] >= 2: 
						leader_ID = b.get_cluster()[0]
						##### !!!! Cheating! Fix with message passing!!! 
						for l in self.agents:
							if l.ID == leader_ID and leader_ID not in a.sent_requests:
								l.process_request(a.generate_request())
								a.sent_requests.append(leader_ID)
		# for a in self.agents:
		# 	if a.parent == None and a.messages['task_flag'] >= 2:
		# 		print(a.requests)
		# 		print(a.cluster_view['view_tasks'])

		#### -- Phase 11 -- Make decisions on requests
		for a in self.agents:
			if a.messages['task_flag'] >= 2 and len(a.get_cluster()) > 0 and a.parent == None:
				decisions = self.probabilistic_cluster_expansion(a)
				for i, request in enumerate(a.requests):
					if decisions[i] == 1:
						if len(request['tasks'].keys()) > 0:
							# print(request['ID'], request['tasks'], list(request['tasks'].keys()))
							# print(set(list(request['tasks'].keys())))
							a.cluster_view['view_tasks'] = \
								list(set(a.cluster_view['view_tasks']).union(set(list(request['tasks'].keys()))))
							self.communication_cost += 2*(10**(-3))*len(set(list(request['tasks'].keys())))
						for b in self.agents:
							if b.ID == request['ID']:
								request_view = self.generate_offset_view(b, (a.pos_x-b.pos_x,a.pos_y-b.pos_y))
								# print(b.ID, b.position(), a.position(), request_view['view_graph'].nodes(), 
								# 				request_view['view_tasks'])
								a.cluster_view['view_graph'] = nx.compose(a.cluster_view['view_graph'],
																		request_view['view_graph'])
								self.communication_cost += (2*len(request_view['view_graph'].nodes()) + \
															4*len(request_view['view_graph'].edges()))*(10**-3)
								# print(a.ID, b.ID, [x.ID for x in a.cluster_view['view_agents']])
								if b.ID in b.messages_prime['clusterID']:
									b.messages_prime['clusterID'].remove(b.ID)
								for x in a.cluster_view['view_agents']:
									if b in x.view['view_agents']:
										b.messages_prime['clusterID'].append(a.ID)
										b.parent_prime = x
										b.color_prime = x.color
										b.gui_split = True
										b.needs_merge = True

										x.children_prime.append(b)
										x.needs_merge = True
										break
								assert b not in a.cluster_view['view_agents']
								a.cluster_view['view_agents'].append(b)
								# print(a.cluster_view['view_graph'].nodes())

					else:
						pass

		self.synchronize_timelines()
		
		for a in self.agents:
			if len(a.get_cluster()) > 0 and a.parent == None and a.messages['task_flag'] >= 2:
				print(a.ID, a.position(), a.cluster_view['view_graph'].nodes(), a.cluster_view['view_tasks'], 
						[b.ID for b in a.cluster_view['view_agents']])

		#### -- Phase 12 -- Super cluster formation
		for a in self.agents:
			if len(a.messages['clusterID']) > 1:
				print(a.ID)

	def probabilistic_cluster_expansion(self, agent):
		decisions = []
		for request in agent.requests:
			num_cluster_agents = len(agent.cluster_view['view_agents'])+1

			# print("PCE:", list(request['tasks'].keys())[0])
			if list(request['tasks'].keys())[0] in agent.cluster_view['view_tasks']:
				num_cluster_tasks = len(agent.cluster_view['view_tasks'])
			else:
				num_cluster_tasks = len(agent.cluster_view['view_tasks'])+1
			r = num_cluster_agents/num_cluster_tasks
			print(agent.ID, agent.position(), math.exp(-self.beta * r))

			random.seed(time.time())
			p = random.uniform(0,1)
			print(p)
			random.seed(12345)
			if p > math.exp(-self.beta * r):
				decisions.append(0)
			else:
				decisions.append(1)
		return decisions

	def get_nearest_cluster_agent_dist(self, agent, task):
		assert task in agent.cluster_view['view_tasks']
		min_dist = float('inf')
		for a in agent.cluster_view['view_agents']:
			dist = nx.shortest_path_length(agent.cluster_view['view_graph'],
											source=a.relative_position(agent.position()),
											target=task)
			if dist < min_dist:
				min_dist = dist
		return min_dist

	def moac_rollout(self, agent):
		decisions = []

		for i, request in enumerate(agent.requests):
			min_cost = float('inf')
			best_decision = None
			q_factors = []
			move_cost = 0
			comm_cost = 0

			for decision in range(2):
				temp_cluster_agents = {}
				for a in agent.cluster_view['view_agents']:
					temp_cluster_agents[a.ID] = a.relative_position(agent.position())

				if decision == 0:
					## do not accept request
					## simulate to obtain cost-to-go
					for j in range(i+1,len(agent.requests)):
						curr_request = agent.requests[j]
						for task in curr_request['tasks']:
							if task in agent.cluster_view['view_tasks']:
								if curr_request['tasks'][task] < self.get_nearest_cluster_agent_dist(agent, task):
									temp_cluster_agents[curr_request['ID']] = curr_request['relative_position']
							else:
								temp_cluster_agents[curr_request['ID']] = curr_request['relative_position']

					move_cost += self.compute_movement_heuristic(temp_cluster_agents)
					comm_cost += self.compute_communication_heuristic(temp_cluster_agents)

				elif decision == 1:
					temp_cluster_agents[request['ID']] = request['position']

		raise NotImplementedError

	def share_map_information(self):
		for a in self.agents:
			if a.messages['task_flag'] >= 2 and len(a.get_cluster()) > 0 and len(a.children) == 0:
				a.messages['super'] = True
		for _ in range(self.N):
			for a in self.agents:
				if a.messages['task_flag'] >= 2 and len(a.get_cluster()) > 0 and a.messages['super'] == True:
					a.cluster_view = {}
					a.cluster_view['view_agents'] = [a]
					a.cluster_view['view_graph'] = a.view['view_graph']
					a.cluster_view['view_tasks'] = a.view['view_tasks']
					for b in a.children:
						a.cluster_view['view_agents'] = \
								list(set(a.cluster_view['view_agents']).union(set(b.cluster_view['view_agents'])))
						self.communication_cost += 3*(10**-3)*len(set(b.cluster_view['view_agents']))

						subcluster_view = self.generate_offset_view(b, (a.pos_x-b.pos_x, a.pos_y-b.pos_y))
						a.cluster_view['view_graph'] = nx.compose(a.view['view_graph'], 
																	subcluster_view['view_graph'])
						self.communication_cost += (2*len(subcluster_view['view_graph'].nodes()) + \
													4*len(subcluster_view['view_graph'].edges()))*(10**-3)

						a.cluster_view['view_tasks'] = list(set(a.view['view_tasks']).union(set(subcluster_view['view_tasks'])))
						self.communication_cost += 2*(10**-3)*len(set(subcluster_view['view_tasks']))

					try:
						a.parent.messages['super'] = True
						a.messages['super'] = False
					except AttributeError:
						assert a.ID == a.get_cluster()[0]

		print("End of LMA:")
		print("------------------------------------------------")
		for a in self.agents:
			if a.messages['task_flag'] >= 2: 
				# print(a.cluster_view)
				print(a.ID, a.get_cluster(), a.position(), a.cluster_view['view_graph'].nodes(), 
					a.cluster_view['view_tasks'], [b.ID for b in a.cluster_view['view_agents']])
		print()

	def compute_controls(self):
		cluster_moves = {}
		for a in self.agents:
			if a.messages['task_flag'] >= 2 and len(a.get_cluster()) > 0 and a.parent == None:
				cluster_moves[a.ID] = self.cluster_multiagent_rollout(a)
				print("Moves:", a.ID, a.position(), cluster_moves)

		for a in self.agents:
			if a.messages['task_flag'] >= 2 and a.parent == None and len(a.get_cluster()) > 0:
				a.messages['super'] = True
				a.moves = cluster_moves[a.get_cluster()[0]][a.ID]
			else:
				a.messages['super'] = False

		for _ in range(self.N):
			for a in self.agents:
				if a.messages['super'] == True:
					for b in a.children:
						if b.messages['super'] == False:
							b.moves = cluster_moves[b.get_cluster()[0]][b.ID]
							b.messages['super'] = True

		for a in self.agents:
			if a.messages['task_flag'] == 1 and a.get_cluster()[0] == a.ID:
				a.moves = self.get_greedy_trajectory(a)


		num_iterations = (2**self.psi)*self.view_radius
		temp_tasks = self.task_vertices.copy()

		for i in range(num_iterations):
			if len(self.task_vertices) == 0:
				break

			for a in self.agents:
				if len(a.get_cluster()) == 0:
					self.sense(a, self.view_radius)
					found = True if len(a.view['view_tasks']) > 0 else False
					if found == False:
						a.dir = self.get_exploration_move(a)
						if a.dir != 'q':
							self.movement_cost += 1
							self.exploration_cost += 1
						else: 
							self.wait_cost += 1
					else:
						a.dir = 'q'
						self.wait_cost += 1
				else:
					if (a.messages['task_flag'] >= 2) or \
						(a.messages['task_flag'] == 1 and a.get_cluster()[0] != a.ID):
						print(a.ID, a.moves)
						try:
							a.dir = a.moves.pop(0)
							if a.dir != 'q':
								self.movement_cost += 1
							else:
								self.wait_cost += 1
						except IndexError:
							a.dir = 'q'
							self.wait_cost += 1
					elif a.messages['task_flag'] == 1 and a.get_cluster()[0] == a.ID:
						try:
							a.dir = a.moves.pop(0)
							if a.dir != 'q':
								self.movement_cost += 1
							else:
								self.wait_cost += 1
						except IndexError:
							a.dir = 'q'
							self.wait_cost += 1

			self.state_update()
			time.sleep(self.wait_time)

		for a in self.agents:
			a.reset()
			a.reset_color()

		if self.visualizer != None:
			cost_label = Label(self.visualizer.root, text="Movement Cost: "+str(self.movement_cost))
			cost_label.grid(row=self.size+1, column=self.size-3, columnspan=4)
			cost_label = Label(self.visualizer.root, text="Communication Cost: "+str(self.communication_cost))
			cost_label.grid(row=self.size+3, column=self.size-3, columnspan=4)

	def state_update(self):
		for a in self.agents:
			if self.visualizer != None:
				if a.dir != 'q':
					self.visualizer.change_cell(a.pos_x, a.pos_y, 'blank', 0)

			if a.dir == 'q':
				pass
			elif a.dir == 'u':
				a.pos_x -= 1
			elif a.dir == 'd':
				a.pos_x += 1
			elif a.dir == 'l':
				a.pos_y -= 1
			elif a.dir == 'r':
				a.pos_y += 1

			if self.visualizer != None:
				self.visualizer.change_cell(a.pos_x, a.pos_y, 'agent', a.color)

			if (a.pos_x, a.pos_y) in self.task_vertices:
				self.task_vertices.remove(a.position())

	def synchronize_timelines(self):
		for a in self.agents:
			if a.needs_merge:
				a.merge_timelines()
				a.needs_merge = False

	def print_agents(self):
		return [a.position() for a in self.agents]

