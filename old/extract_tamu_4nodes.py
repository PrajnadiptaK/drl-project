import osmnx as ox
import networkx as nx

# 1) Define the 4 real intersections as text addresses
labels = [
    "George Bush Drive and Wellborn Road, College Station, TX",
    "George Bush Drive and Olsen Boulevard, College Station, TX",
    "Wellborn Road and Joe Routt Boulevard, College Station, TX",
    "Olsen Boulevard and Joe Routt Boulevard, College Station, TX",]

print("Geocoding intersections...")
points = []

for label in labels:
    # geocode each intersection to (lat, lon)
    gdf = ox.geocode_to_gdf(label)
    lat = gdf.geometry.y.values[0]
    lon = gdf.geometry.x.values[0]
    points.append((lat, lon))
    print(f"{label}: lat={lat:.6f}, lon={lon:.6f}")

# 2) Build a road network around the region (centered at the first point)
center_lat, center_lon = points[0]
print("\nDownloading road network around first intersection...")
G = ox.graph_from_point(
    (center_lat, center_lon),
    dist=1500,          # 1.5 km radius, adjust if needed
    network_type="drive"
)

print("Original nodes:", len(G.nodes))
G_s = ox.simplify_graph(G)
print("Simplified nodes:", len(G_s.nodes))

# 3) For each geocoded intersection, find the nearest graph node
print("\nFinding nearest graph node to each intersection...")
chosen_node_ids = []
for (lat, lon), label in zip(points, labels):
    node = ox.nearest_nodes(G_s, X=lon, Y=lat)
    chosen_node_ids.append(node)
    print(f"{label} -> node_id={node}")

# 4) Build index mapping: 0..3 -> node_id
index_mapping = {i: node for i, node in enumerate(chosen_node_ids)}
print("\nIndex mapping (env index -> OSM node id):")
for i, nid in index_mapping.items():
    print(f"  {i}: {nid}  ({labels[i]})")

# 5) Build adjacency among these 4 nodes (only if directly connected by an edge)
adj_dict = {i: [] for i in range(len(chosen_node_ids))}

for u, v in G_s.edges():
    if u in chosen_node_ids and v in chosen_node_ids:
        ui = chosen_node_ids.index(u)
        vi = chosen_node_ids.index(v)
        # undirected adjacency for our RL env
        if vi not in adj_dict[ui]:
            adj_dict[ui].append(vi)
        if ui not in adj_dict[vi]:
            adj_dict[vi].append(ui)

print("\nAdjacency (using indices 0..3):")
for i in range(len(chosen_node_ids)):
    print(f"  {i}: neighbors -> {adj_dict[i]}")

# 6) Also print adjacency matrix (for use in CoLight)
print("\nAdjacency matrix (N x N):")
N = len(chosen_node_ids)
adj_mat = [[0]*N for _ in range(N)]
for i in range(N):
    for j in adj_dict[i]:
        adj_mat[i][j] = 1
        adj_mat[j][i] = 1  # symmetric

for row in adj_mat:
    print(row)
