import networkx as nx
import math


def sort_by_eco(g, rest_list):
    # stop recursion once we've dealt with every restraint
    if not rest_list:
        return []

    sorted_rests = sorted(rest_list,
                          key=lambda x: nx.shortest_path_length(g, x[0], x[1]))

    lowest_eco = sorted_rests.pop(0)
    g.add_edge(lowest_eco[0], lowest_eco[1])
    return [lowest_eco] + sort_by_eco(g, sorted_rests)


def get_variable_grouped_By_eco(n_groups):
    # read in data from files
    ca_list = [int(line) for line in open('ca_list.dat')]
    rests = []
    for line in open('variable_bonds0.dat'):
        i, j, r = line.split()
        i = int(i) - 1
        j = int(j) - 1
        r = float(r)
        r1 = max(0.0, r - 0.1)
        r2 = r
        r3 = r
        r4 = r + 0.1
        rests.append((i, j, r1, r2, r3, r4))

    # construct initial networkx graph
    g = nx.Graph()
    for i in ca_list:
        g.add_node(i)
    for i, j in zip(ca_list[:-1], ca_list[1:]):
        g.add_edge(i, j)

    # now we'll get our sorted list of contacts by
    # recursively calling sort_by_eco.
    sorted_contacts = sort_by_eco(g, rests)

    # divide into the appropriate number of groups
    n = len(sorted_contacts)
    n_per_group = int(math.ceil(float(n) / n_groups))
    chunks = [sorted_contacts[x:x+n_per_group] for x in range(0, n, n_per_group)]

    return chunks
