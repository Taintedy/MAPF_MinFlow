import math
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

class Node:
    def __init__(self, pose, node_type="default", capacity=1, debug=""):
        self.debug = debug
        self.node_type = node_type
        self.pose = pose
        self.capacity = capacity

        self.children = []
        self.parents = []

        self.is_start = False
        self.is_goal = False
        self.is_auxiliary = False

        if node_type == "start":
            self.is_start = True
        elif node_type == "goal":
            self.is_goal = True
        elif node_type == "auxiliary":
            self.is_auxiliary = True

    def add_child(self, node, cost, capacity):
        residual_edge = InfoEdge(self, cost, 0)
        edge = InfoEdge(node, cost, capacity)
        edge.residual_edge = residual_edge
        residual_edge.residual_edge = edge

        self.children.append(edge)
        node.parents.append(residual_edge)

    def print_pose(self):
        if self.is_start:
            return str(self.pose) + " is start " + self.debug
        elif self.is_goal:
            return str(self.pose) + " is goal " + self.debug
        elif self.is_auxiliary:
            return str(self.pose) + " is auxiliary " + self.debug
        else:
            return str(self.pose) + " is default " + self.debug


class Edge:
    def __init__(self, nodeFrom, nodeTo, orientated=True, cost=1, capacity=1):
        self.nodeFrom = nodeFrom
        self.nodeTo = nodeTo
        self.capacity = capacity
        self.orientated = orientated
        self.cost = cost


class InfoEdge:
    def __init__(self, node, cost, capacity):
        self.node = node
        self.capacity = capacity
        self.cost = cost
        self.flow = 0
        self.residual_edge = None
        self.parent = None

    def remaining_capacity(self):
        return self.capacity - self.flow

    def augment(self, bottleneck):
        if self.capacity == 0:
            self.flow -= bottleneck
        else:
            self.flow += bottleneck


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def draw_graph(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        for edge in self.edges:
            ax.plot([edge.nodeFrom.pose[0], edge.nodeTo.pose[0]], [edge.nodeFrom.pose[1], edge.nodeTo.pose[1]],
                    [edge.nodeFrom.pose[2], edge.nodeTo.pose[2]], c="#59c2ff")
            if edge.nodeFrom.is_start:
                ax.scatter(edge.nodeFrom.pose[0], edge.nodeFrom.pose[1], edge.nodeFrom.pose[2], c="r")
            elif edge.nodeFrom.is_goal:
                ax.scatter(edge.nodeFrom.pose[0], edge.nodeFrom.pose[1], edge.nodeFrom.pose[2], c="b")
            else:
                ax.scatter(edge.nodeFrom.pose[0], edge.nodeFrom.pose[1], edge.nodeFrom.pose[2], c="black")
            if edge.nodeTo.is_start:
                ax.scatter(edge.nodeTo.pose[0], edge.nodeTo.pose[1], edge.nodeTo.pose[2], c="r")
            elif edge.nodeTo.is_goal:
                ax.scatter(edge.nodeTo.pose[0], edge.nodeTo.pose[1], edge.nodeTo.pose[2], c="b")
            else:
                ax.scatter(edge.nodeTo.pose[0], edge.nodeTo.pose[1], edge.nodeTo.pose[2], c="black")

        plt.show()

    def draw_graph_with_path(self, paths):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        #
        for edge in self.edges:
            ax.plot([edge.nodeFrom.pose[0], edge.nodeTo.pose[0]], [edge.nodeFrom.pose[1], edge.nodeTo.pose[1]],
                    [edge.nodeFrom.pose[2], edge.nodeTo.pose[2]], c="#59c2ff")
            if edge.nodeFrom.is_start:
                ax.scatter(edge.nodeFrom.pose[0], edge.nodeFrom.pose[1], edge.nodeFrom.pose[2], c="r")
            elif edge.nodeFrom.is_goal:
                ax.scatter(edge.nodeFrom.pose[0], edge.nodeFrom.pose[1], edge.nodeFrom.pose[2], c="b")
            else:
                ax.scatter(edge.nodeFrom.pose[0], edge.nodeFrom.pose[1], edge.nodeFrom.pose[2], c="black")
            if edge.nodeTo.is_start:
                ax.scatter(edge.nodeTo.pose[0], edge.nodeTo.pose[1], edge.nodeTo.pose[2], c="r")
            elif edge.nodeTo.is_goal:
                ax.scatter(edge.nodeTo.pose[0], edge.nodeTo.pose[1], edge.nodeTo.pose[2], c="b")
            else:
                ax.scatter(edge.nodeTo.pose[0], edge.nodeTo.pose[1], edge.nodeTo.pose[2], c="black")

        for path in paths:
            for i in range(len(path) - 1):
                ax.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]],
                        [path[i][2], path[i+1][2]], c="g", linewidth=4)
        plt.show()


class TimeExpandedFlowGraph:
    def __init__(self, graph):
        self.clean_graph = copy.deepcopy(graph)
        self.nodes = self.clean_graph.nodes
        self.current_edges = self.clean_graph.edges
        self.edges = []
        self.paths = []
        self.T = 1
        self.master_source = Node("Master source", node_type="auxiliary")
        for node in self.nodes:
            if node.is_start:
                self.master_source.add_child(node, 0, 1)
        path_found = False
        j = 0
        while not path_found:
            i = self.T
            next_edges = []
            print("starting graph")
            for edge in self.current_edges:
                node_from = edge.nodeFrom
                node_to = edge.nodeTo

                # add auxilary edge w -> w', A -> w and B -> w
                auxiliary_node1 = Node("w" + str(j), node_type="auxiliary", debug="time " + str(i))
                auxiliary_node2 = Node("w'" + str(j), node_type="auxiliary", debug="time " + str(i))
                auxiliary_node1.add_child(auxiliary_node2, 1, 1)  # node cost capacity
                node_from.add_child(auxiliary_node1, 0, 1)
                node_to.add_child(auxiliary_node1, 0, 1)

                # adding edge from current node to auxiliary node A -> A' and B -> B'
                if len(node_from.children) < 2:
                    # create auxiliary nodes of A and B
                    auxiliary_from = Node(node_from.pose, node_type="auxiliary", debug="time " + str(i))
                    node_from.add_child(auxiliary_from, 1, 1)
                    # adding next time step nodes A' -> A(t+1)
                    next_timestep_node_from = Node(node_from.pose, node_type=node_from.node_type, debug="time " + str(i + 1))
                    auxiliary_from.add_child(next_timestep_node_from, 0, 1)
                else:
                    auxiliary_from = node_from.children[1].node

                if len(node_to.children) < 2:
                    auxiliary_to = Node(edge.nodeTo.pose, node_type="auxiliary", debug="time " + str(i))
                    node_to.add_child(auxiliary_to, 1, 1)
                    # adding next time step nodes B' -> B(t+1)
                    next_timestep_node_to = Node(node_to.pose, node_type=node_to.node_type, debug="time " + str(i + 1))
                    auxiliary_to.add_child(next_timestep_node_to, 0, 1)
                else:
                    auxiliary_to = node_to.children[1].node

                # add auxilary edges w' -> A' and w' -> B'
                auxiliary_node2.add_child(auxiliary_from, 0, 1)
                auxiliary_node2.add_child(auxiliary_to, 0, 1)

                j += 1
                next_edges.append(Edge(auxiliary_from.children[0].node, auxiliary_to.children[0].node))

            self.current_edges = copy.copy(next_edges)

            self.master_sink = Node("Master sink", node_type="auxiliary")

            self.num_of_goals = 0
            for edge in self.current_edges:
                node_from = edge.nodeFrom
                node_to = edge.nodeTo
                if len(node_from.children) == 0 and node_from.is_goal:
                    print(node_from.print_pose(), " is sink")
                    node_from.add_child(self.master_sink, 0, 1)
                    self.num_of_goals += 1
                if len(node_to.children) == 0 and node_to.is_goal:
                    print(node_to.print_pose(), " is sink")
                    node_to.add_child(self.master_sink, 0, 1)
                    self.num_of_goals += 1

            print("end graph")
            print("start path")
            self.paths = self.find_paths()
            print("end path")
            # print(len(self.paths) == len(self.master_source.children))
            print(self.T)
            self.T += 1
            path_found = len(self.paths) == self.num_of_goals or len(self.paths) == len(self.master_source.children)
            print(len(self.paths) == len(self.master_source.children))

            for edge in self.master_sink.parents:
                edge.node.children.remove(edge.residual_edge)


    def print_flow_graph(self):
        explored = []
        q = []
        explored.append(self.master_source)
        q.append(self.master_source)
        while len(q) > 0:
            v = q.pop(0)
            edges = v.children + v.parents
            for edge in edges:
                if edge.node not in explored:
                    explored.append(edge.node)
                    q.append(edge.node)
                print(v.print_pose(), " -> ", edge.node.print_pose(), " has flow: " , edge.flow, " is residual: ", edge.capacity == 0)


    def find_paths(self):
        bottleneck = math.inf
        explored = []
        explored_edges = []
        path = []
        q = []
        explored.append(self.master_source)
        q = q + self.master_source.children
        # print("finding flows")
        while len(q) > 0:
            e = q.pop()

            if e.node == self.master_sink:
                # print("got to master sink from ", e.node.print_pose())
                current_edge = e
                while current_edge is not None:
                    path.append(current_edge)
                    current_edge = current_edge.parent
                    # if current_edge is not None:
                    #     if current_edge.node == self.master_source:
                    #         break

                for edge in path:
                    bottleneck = min(bottleneck, edge.remaining_capacity())
                # print("bottelneck for: ", path[-1].node.print_pose(), " is ", bottleneck)
                for edge in path:
                    edge.augment(bottleneck)
                    edge.residual_edge.augment(bottleneck)
                explored = []
                # explored.append(self.master_source)
                bottleneck = math.inf
                path = []
            else:
                explored.append(e.node)
                explored_edges.append(e)
                edges = e.node.children + e.node.parents
                for edge in edges:
                    if edge not in explored_edges and edge.node not in explored:
                        if edge.node != self.master_source:
                            edge.parent = e
                        if edge.remaining_capacity() == 1:
                            q.append(edge)

        # print("finished finding flows")
        explored = []
        paths = []
        path = []
        q = []
        q.append(self.master_source)
        # print("finding paths")
        while len(q) > 0:
            v = q.pop()
            # print(v.print_pose())
            if not v.is_auxiliary:
                path.append(v.pose)
            if v == self.master_sink:
                # print("Got to master_sink")
                # print(path)
                paths.append(path)
                path = []
            else:
                explored.append(v)
                for edge in v.children:
                    if edge.flow > 0 and edge.node not in explored:
                        q.append(edge.node)
                    edge.flow = 0
                    edge.residual_edge.flow = 0
                    edge.parent = None
        print(paths)
        return paths




class drone:
    def __init__(self):
        self.pose = np.array([0, 0, 0])
        self.path = []
        self.speed = 0.5

    def move_path(self, path):
        self.pose = np.array(path[0])
        self.path.append(self.pose)
        for point in path:
            target = np.array(point)
            while np.linalg.norm(abs(self.pose - target)) > 0.1:
                vel = (target - self.pose) / np.linalg.norm(target - self.pose)
                self.pose = self.pose + vel * self.speed
                self.path.append(self.pose)
        print(self.path)




def create_world(x, y, z, starts, goals):
    nodes = []
    edges = []
    step = 1
    for i in range(x):
        nodes.append([])
        for j in range(y):
            nodes[i].append([])
            for k in range(z):
                node = Node([i*step, j*step, k*step])
                nodes[i][j].append(node)
    print(nodes)
    for point in starts:
        nodes[point[0]][point[1]][point[2]] = Node([step*point[0], step * point[1], step * point[2]], node_type="start")
        print(nodes[point[0]][point[1]][point[2]].is_start, " adding start node")
    for point in goals:
        nodes[point[0]][point[1]][point[2]] = Node([step*point[0], step * point[1], step * point[2]], node_type="goal")

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if i + 1 < x:
                    edges.append(Edge(nodes[i][j][k], nodes[i + 1][j][k]))
                if j + 1 < y:
                    edges.append(Edge(nodes[i][j][k], nodes[i][j + 1][k]))
                if k + 1 < z:
                    edges.append(Edge(nodes[i][j][k], nodes[i][j][k + 1]))
    nodes = list(chain.from_iterable(nodes))
    nodes = list(chain.from_iterable(nodes))
    print("anount of nodes ", len(nodes))
    return Graph(nodes, edges)

if __name__ == "__main__":
    node1 = Node([1, 0, 0], node_type="start")
    node2 = Node([0, 1, 0], node_type="start")
    node3 = Node([0, 0, 0])
    node4 = Node([0, 0, 1])
    node5 = Node([1, 0, 1], node_type="goal")
    node6 = Node([0, 1, 1], node_type="goal")
    node7 = Node([1, 1, 1])
    node8 = Node([1, 1, 0])


    nodes = [node1, node2, node3, node4, node5, node6]
    edges = [Edge(node1, node3),
             Edge(node2, node3),
             Edge(node3, node4),
             Edge(node4, node5),
             Edge(node4, node6),
             Edge(node1, node8),
             Edge(node2, node8),
             Edge(node8, node7),
             Edge(node7, node5),
             Edge(node7, node6)]
             #Edge(node1, node5),
             #Edge(node2, node6)]

    # node1 = Node([0, 0, 0], node_type="start", debug="time " + str(0))
    # node2 = Node([1, 0, 0], node_type="start", debug="time " + str(0))
    # node3 = Node([2, 0, 0], debug="time " + str(0))
    # node4 = Node([3, 0, 0], node_type="goal", debug="time " + str(0))
    # node5 = Node([2, 1, 0], node_type="goal", debug="time " + str(0))
    # nodes = [node1, node2, node3, node4, node5]
    # edges = [Edge(node1, node2),
    #          Edge(node2, node3),
    #          Edge(node3, node4),
    #          Edge(node3, node5)]
    world = Graph(nodes, edges)

    # starts = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    # goals = [[5, 5, 5], [0, 5, 5], [1, 5, 5], [2, 5, 5]]
    # # starts = [[0, 0, 0], [0, 1, 0]]
    # # goals = [[1, 1, 1], [0, 1, 1]]
    #
    # world = create_world(6, 6, 6, starts, goals)

    texpgr = TimeExpandedFlowGraph(world)
    print(len(texpgr.paths))

    swarm_paths = []
    for path in texpgr.paths:
        robot = drone()
        robot.move_path(path)
        swarm_paths.append(robot.path)

    world.draw_graph_with_path(swarm_paths)

    # texpgr.print_flow_graph()
