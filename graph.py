
import desc

"""             ESTRUTURA DO GRAPH

    vertexes[] = um array de objetos do tipo Vertex

"""
class Graph:
    def __init__(self):
        self.vertexes:list[Vertex] = list()
        self.avg_distances:int
    
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
        for neigh_of_vertex in self.vertexes[i].neighs:  # removes dead reference from each neighbor
            self.vertexes[neigh_of_vertex.i].neighs.pop([n.i for n in self.vertexes[neigh_of_vertex.i].neighs].index(i))  # the list comprehension finds the index of neighbor to update

        if i < len(self.vertexes)-1:  # indexes range from 0..n-1. removing last element has simpler logic, since we don't need to move anything
            self.vertexes[i] = self.vertexes.pop()  # replaces element
            self.vertexes[i].i = i  # updates i for moved element
            for moved_neigh in self.vertexes[i].neighs:     # updates indexes of neighbors for the moved vertex (they would still reference the final position)
                self.vertexes[moved_neigh.i].neighs[[n.i for n in self.vertexes[moved_neigh.i].neighs].index(len(self.vertexes))].i = i  # the list comprehension finds the index of neighbor to update
        else:  # if the element to remove is in the last index, just remove it
            self.vertexes.pop()
            
    def find_vertex(self, yx: list) :
        for vertex in self.vertexes :
            if (vertex.yx[0] == yx[0] and vertex.yx[1] == yx[1]) :
                return vertex

        return 0

    def create_edge(self, i1, i2):
        self.vertexes[i1].add_neigh(i2)
        self.vertexes[i2].add_neigh(i1)

    def add_vertex(self, yx:"list[int]", list_neighs:"list[Neigh]"):
        new_pos = len(self.vertexes)  # end of list
        self.vertexes.append(Vertex(self, new_pos, yx))
        for n in list_neighs:
            self.create_edge(new_pos, n)


"""             ESTRUTURA DO VERTEX

    graph = contem o ponteiro para o objeto Graph o qual esse Vertex pertence
    
    i = o indice que esse Vertex se encontra no objeto Graph
    
    yx = lista com as cordenadas y , x
    
    neighs[] = um array de objetos do tipo Neigh

"""
class Vertex():
    def __init__(self, graph:Graph, i, yx=None, neighs=None):
        self.graph:Graph = graph                                            # this reference is necessary if we wanna use the neighbor add function
        self.i = i                                                          # this should be only used for printing, and should be equal to the element's index as of now
        self.yx = yx if yx is not None else list()
        self.neighs:list[Neigh] = neighs if neighs is not None else list()  # ideally would be a set in order to disallow duplicates, but we want ordering in order to prioritize neighsbors later

    def __str__(self):
        print_i = -1 if self.i is None else self.i
        print_yx = [-1, -1] if self.yx is None else self.yx
        print_neighs = 'uninitialized' if self.neighs is None else [[n.i, round(n.dist,2), round(n.ang,2)] for n in self.neighs]
        return ("i: %d | yx: [%.1f, %.1f] | neighs: %s" % (print_i, print_yx[0], print_yx[1], print_neighs))
    
    def get_neigh_vertex(self, neigh_i) :
        return self.graph.vertexes[self.neighs[neigh_i].i]
    
    def add_neigh(self, neigh_i):
        if neigh_i in [getattr(n,'i') for n in self.neighs]:  # if neighbor already exists, update neighbor by removing it and adding it again
            self.neighs.pop([getattr(n,'i') for n in self.neighs].index(neigh_i))
        self.neighs.append(Neigh(neigh_i, desc.calc_dist(self.graph.vertexes[self.i],self.graph.vertexes[neigh_i]), desc.calc_ang(self.graph.vertexes[self.i],self.graph.vertexes[neigh_i])))


"""             ESTRUTURA DO NEIGH

    i = o indice que esse vizinho se encontra no objeto Graph
    
    dist = a distancia entre o Vertex e esse vizinho
    
    ang = o angulo relativo entre o Vertex e esse vizinho
    ( esse angulo é relativo aa horizontal, com valores entre 180 e -180 )

"""
class Neigh():
    def __init__(self, i, dist, ang):  # (graph,vertex) references used if we are to initialize (dist,ang) from the constructor
        self.i = i  # this one is actually necessary
        self.dist = dist
        self.ang = ang

    def __str__(self):
        return ("i: %d | dist: %.2f | ang: %.2f" % (self.i, self.dist, self.ang))
