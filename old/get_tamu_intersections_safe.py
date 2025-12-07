import osmnx as ox

ox.settings.log_console = True

place = "College Station, Texas"

# intersection-friendly descriptions
queries = [
    "George Bush Dr & Wellborn Rd, College Station, TX",
    "George Bush Dr & Olsen Blvd, College Station, TX",
    "Wellborn Rd & Joe Routt Blvd, College Station, TX",
    "Olsen Blvd & Joe Routt Blvd, College Station, TX"
]

coords = []

for q in queries:
    print("\nTrying:", q)
    try:
        gdf = ox.geocode_to_gdf(q)
        lat = gdf.geometry.y.values[0]
        lon = gdf.geometry.x.values[0]
        print("✓ Found via direct geocode:", lat, lon)
        coords.append((lat, lon))
    except Exception:
        print("⚠ Direct geocode failed, falling back")

        # fallback: geocode each road name separately
        street1, street2 = q.split("&")
        street1 = street1.replace("Dr", "Drive").replace("Rd", "Road")
        street2 = street2.replace("Blvd", "Boulevard").replace("Rd", "Road")

        g1 = ox.geocode_to_gdf(street1 + ", College Station, TX")
        g2 = ox.geocode_to_gdf(street2 + ", College Station, TX")

        # use midpoint between best match points
        lat = (g1.geometry.y.values[0] + g2.geometry.y.values[0]) / 2
        lon = (g1.geometry.x.values[0] + g2.geometry.x.values[0]) / 2

        print("✓ Using midpoint approximation:", lat, lon)
        coords.append((lat, lon))

print("\n=== Final Intersection Coordinates ===")
for q, (lat, lon) in zip(queries, coords):
    print(f"{q} -> lat={lat:.6f}, lon={lon:.6f}")
