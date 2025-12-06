import osmnx as ox
import networkx as nx

chosen_nodes = [
    3903978756,  # George Bush & Wellborn
    222634168,   # George Bush & Olsen
    4075789145,  # Wellborn & Joe Routt
    6122967573   # Olsen & Joe Routt
]

center = (30.61387, -96.34474)

print("Downloading graph...")
G = ox.graph_from_point(center, dist=1500, network_type="drive")

# Convert graph to undirected using new API
Gu = ox.to_undirected(G)

# Nearest node lookup API (new)
def nearest_in_graph(x, y):
    return ox.distance.nearest_nodes(Gu, X=x, Y=y)

# Map OSN coordinates → nearest true graph intersections
projected_nodes = []
for osm_id in chosen_nodes:
    x = G.nodes[osm_id]['x']
    y = G.nodes[osm_id]['y']
    real = nearest_in_graph(x, y)
    projected_nodes.append(real)

print("\nProjected OSM nodes:", projected_nodes)

# Build adjacency (distance-based connectivity)
adj_dict = {i: [] for i in range(len(projected_nodes))}

for i, ni in enumerate(projected_nodes):
    for j, nj in enumerate(projected_nodes):
        if i != j:
            try:
                d = nx.shortest_path_length(Gu, ni, nj, weight="length")
                # roads < 300 meters apart considered neighbors
                if d < 300:
                    adj_dict[i].append(j)
            except Exception:
                pass

print("\n=== Smart Adjacency ===")
for k, v in adj_dict.items():
    print(k, ":", v)

print("\n(If empty → we apply real-world adjacency)")
