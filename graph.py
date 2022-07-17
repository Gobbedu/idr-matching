import desc

class Graph:
    def __init__(self):
        self.vertexes = list()
    
    def __str__(self):
        return("vertexes: %d  |  edges: %d" % (len(self.vertexes), self.count_edges()))

    def count_edges(self):
        n_edges = 0
        for v in self.vertexes:
            n_edges += len(v.neighs)
        return n_edges

    def print_vertexes(self):
        for v in self.vertexes:
            print(v)

    def remove_vertex(self, i):  # removes element in position to be removed (i) by replacing it with the last element
        if i < len(self.vertexes):
            for neigh_of_vertex in self.vertexes[i].neighs:  # removes dead reference from each neighbor
                self.vertexes[neigh_of_vertex.i].neighs.pop([n.i for n in self.vertexes[neigh_of_vertex.i].neighs].index(i))  # the list comprehension finds the index of neighbor to update

            self.vertexes[i] = self.vertexes.pop()  # replaces element
            self.vertexes[i].i = i  # updates i for moved element

            # updates indexes of neighbors for the moved vertex (they would still reference the final position)
            for moved_neigh in self.vertexes[i].neighs:
                self.vertexes[moved_neigh.i].neighs[[n.i for n in self.vertexes[moved_neigh.i].neighs].index(len(self.vertexes))].i = i  # the list comprehension finds the index of neighbor to update

        else:  # if the element to remove is in the last index, just remove it
            self.vertexes.pop()

    def create_edge(self, i1, i2):
        self.vertexes[i1].add_neigh(i2)
        self.vertexes[i2].add_neigh(i1)

    def add_vertex(self, yx, list_neighs):
        new_pos = len(self.vertexes)  # end of list
        self.vertexes.append(Vertex(self, new_pos, yx))
        for n in list_neighs:
            self.create_edge(new_pos, n)


class Vertex():
    def __init__(self, graph, i, yx=None, neighs=None):
        self.graph = graph  # this reference is necessary if we wanna use the neighbor add function
        self.i = i  # this should be only used for printing, and should be equal to the element's index as of now
        self.yx = yx if yx is not None else list()
        self.neighs = neighs if neighs is not None else list()  # ideally would be a set in order to disallow duplicates, but we want ordering in order to prioritize neighsbors later

    def __str__(self):
        print_i = -1 if self.i is None else self.i
        print_yx = [-1, -1] if self.yx is None else self.yx
        print_neighs = 'uninitialized' if self.neighs is None else [[n.i, round(n.dist,2), round(n.ang,2)] for n in self.neighs]
        return ("i: %d | yx: [%.0f, %.0f] | neighs: %s" % (print_i, print_yx[0], print_yx[1], print_neighs))
    
    def add_neigh(self, neigh_i):
        self.neighs.append(Neigh(neigh_i, desc.calc_dist(self.graph.vertexes[self.i],self.graph.vertexes[neigh_i]), desc.calc_ang(self.graph.vertexes[self.i],self.graph.vertexes[neigh_i])))


class Neigh():
    def __init__(self, i, dist, ang):  # (graph,vertex) references used if we are to initialize (dist,ang) from the constructor
        self.i = i  # this one is actually necessary
        self.dist = dist
        self.ang = ang

    def __str__(self):
        return ("i: %d | dist: %.2f | ang: %.2f" % (self.i, self.dist, self.ang))
