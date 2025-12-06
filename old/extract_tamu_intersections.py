import osmnx as ox
import shapely
import pandas as pd

ox.settings.log_console = True

# Helper: fetch street geometry by street name
def get_street_geometry(place, street_name):
    gdf = ox.features_from_place(place, tags={'name': street_name})
    road_lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    return road_lines.geometry.unary_union  # combine segments

# Define references
place = "College Station, Texas"

street_pairs = [
    ("George Bush Drive", "Wellborn Road"),
    ("George Bush Drive", "Olsen Boulevard"),
    ("Wellborn Road", "Joe Routt Boulevard"),
    ("Olsen Boulevard", "Joe Routt Boulevard")
]

points = []

for s1, s2 in street_pairs:
    print(f"\nProcessing intersection of {s1} and {s2}...")

    g1 = get_street_geometry(place, s1)
    g2 = get_street_geometry(place, s2)

    inter = g1.intersection(g2)

    if isinstance(inter, shapely.Point):
        lat, lon = inter.y, inter.x
        print(f"✓ found point: lat={lat:.6f}, lon={lon:.6f}")
        points.append((lat, lon))
    else:
        print("⚠️ NO direct point intersection — geometry type:", type(inter))
        # fallback — pick centroid
        centroid = inter.centroid
        lat, lon = centroid.y, centroid.x
        print(f"✓ using centroid approx: lat={lat:.6f}, lon={lon:.6f}")
        points.append((lat, lon))

# Output results
print("\n=== Final extracted intersection coordinates ===")
for pair, (lat, lon) in zip(street_pairs, points):
    print(f"{pair[0]} & {pair[1]} -> lat={lat}, lon={lon}")
