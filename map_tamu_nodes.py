import osmnx as ox

# HARD-CODED coordinates for your 4 intersections
coords = [
    (30.61387, -96.34474),  # George Bush & Wellborn
    (30.61395, -96.33828),  # George Bush & Olsen
    (30.60720, -96.34502),  # Wellborn & Joe Routt
    (30.60727, -96.33833)   # Olsen & Joe Routt
]

center = coords[0]

print("Downloading College Station graph...")
G = ox.graph_from_point(center, dist=1500, network_type="drive")

# If graph is already simplified, just use it
# Otherwise simplify
try:
    G_s = ox.simplify_graph(G)
    print("Graph simplified.")
except Exception:
    print("Graph already simplified.")
    G_s = G

chosen_nodes = []

for lat, lon in coords:
    nid = ox.nearest_nodes(G_s, X=lon, Y=lat)
    chosen_nodes.append(nid)
    print(f"lat={lat}, lon={lon} -> node_id={nid}")

mapping = {i: node for i, node in enumerate(chosen_nodes)}
print("\nMapping env index -> OSM node id:")
print(mapping)
