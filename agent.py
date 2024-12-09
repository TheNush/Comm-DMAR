import networkx as nx
from copy import deepcopy

class Agent:
    def __init__(self, id, position, env, color=10, policy="dmar-comm"):
        self.ID = id

        self.pos_x = position[0]
        self.pos_y = position[1]
        self.offset_x = None
        self.offset_y = None

        self.env = env

        self.children = []
        self.parent = None
        self.children_prime = []
        self.parent_prime = None
        self.color = color
        self.color_prime = color
        self.requests = []
        self.sent_requests = []

        self.moves = []
        self.dir = None

        self.movement_cost = 0
        self.communication_cost = 0
        self.sensing_cost = 0

        self.view = {'view_graph': None, 'view_tasks':[], 'view_agents': []}
        self.cluster_view = {'view_graph': None, 'view_tasks':[], 'view_agents': []}
        self.cluster_view_prime = {'view_graph': None, 'view_tasks':[], 'view_agents': []}
        self.messages = {'task_flag': None, 'clusterID': [], 'super': False}
        self.messages_prime = {'task_flag': None, 'clusterID': [], 'super': False}
        self.needs_merge = False
        self.gui_split = False

        self.policy_type = policy

    def reset(self):
        self.children = []
        self.parent = None
        self.children_prime = []
        self.parent_prime = None
        self.requests = []
        self.sent_requests = []

        self.view = {'view_graph': None, 'view_tasks':[], 'view_agents': []}
        self.cluster_view = {'view_graph': None, 'view_tasks':[], 'view_agents': []}
        self.cluster_view_prime = {'view_graph': None, 'view_tasks':[], 'view_agents': []}
        self.messages = {'task_flag': None, 'clusterID': [], 'super': False}
        self.messages_prime = {'task_flag': None, 'clusterID': [], 'super': False}
        self.needs_merge = False
        self.gui_split = False

    def reset_color(self):
        self.color = 10
        self.color_prime = 10
        if self.env.visualizer != None:
            self.env.visualizer.change_cell(self.pos_x, self.pos_y, 
                                            'agent', self.color)

    def communicate(self, agent, message):
        """
            Sends message to agent (in its view). 
        """
        raise NotImplementedError

    def merge_timelines(self):
        self.messages = deepcopy(self.messages_prime)
        self.children = self.children_prime.copy()
        self.parent = self.parent_prime
        flag = False
        if self.color_prime != self.color:
            flag = True
            self.color = self.color_prime

        if self.gui_split and flag: 
            self.env.visualizer.change_cell(self.pos_x, self.pos_y, 
                                            'agent', self.color)

    def raise_task_flag(self,mode=0):
        # if len(self.view['view_tasks']) <= 1:
        #     self.messages_prime['task_flag'] = len(self.view['view_tasks'])
        # else:
        #     self.messages_prime['task_flag'] = 2
        print(self.view)
        if mode == 0:
            self.messages_prime['task_flag'] = len(self.view['view_tasks'])
        elif mode == 1:
            self.messages_prime['task_flag'] = 2
        self.needs_merge = True

    def generate_request(self):
        return {'ID': self. ID,
                'position': self.position(),
                'tasks': {task: nx.shortest_path_length(self.view['view_graph'], 
                                    source=(0,0), target=task) for task in self.view['view_tasks']}}

    def process_request(self, request):
        processed_request = {'ID': request['ID']}
        ## identify agent in view sending the request
        # print(request)

        processed_request['position'] = (request['position'][0]-self.pos_x, 
                                            request['position'][1]-self.pos_y)
        request_position = processed_request['position']

        processed_request['tasks'] = {}
        for task in request['tasks']:
            relative_task = (task[0]+request_position[0],task[1]+request_position[1])
            processed_request['tasks'][relative_task] = request['tasks'][task]
        # print(processed_request)
        self.requests.append(processed_request)

    def get_movement_heuristic(self, request):
        raise NotImplementedError

    def get_communication_heuristic(self, request):
        raise NotImplementedError

    def move_agent(self, move):
        if move == 'u':
            self.pos_x -= 1
        elif move == 'd':
            self.pos_x += 1
        elif move == 'l':
            self.pos_y -= 1
        elif move == 'r':
            self.pos_y += 1

    def print_children(self):
        return [a.ID for a in self.children]

    def print_parent(self):
        if self.parent is not None:
            return self.parent.ID
        else:
            return None

    def position(self):
        return (self.pos_x, self.pos_y)

    def relative_position(self, offset):
        return (self.pos_x-offset[0], self.pos_y-offset[1])

    def get_cluster(self):
        return self.messages['clusterID']
