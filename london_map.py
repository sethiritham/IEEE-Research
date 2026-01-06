import osmnx as ox
import matplotlib.pyplot as plt

place_name = "London, UK"
graph = ox.graph_from_place(place_name, network_type='drive', simplify=True)
graph_proj = ox.project_graph(graph)

graph_proj = ox.add_edge_speeds(graph_proj)
graph_proj = ox.add_edge_travel_times(graph_proj)

nodes, edges = ox.graph_to_gdfs(graph_proj)

nodes = nodes.reset_index()
edges = edges.reset_index()

static_edge_features = edges[['length', 'speed_kph', 'highway']].copy()

node_id_to_idx = {osm_id: i for i, osm_id in enumerate(nodes['osmid'])}
nodes['original_osmid'] = nodes['osmid']
nodes['osmid'] = nodes['osmid'].map(node_id_to_idx)

edges['u'] = edges['u'].map(node_id_to_idx)
edges['v'] = edges['v'].map(node_id_to_idx)

edges = edges.dropna(subset=['u', 'v'])
edges['u'] = edges['u'].astype(int)
edges['v'] = edges['v'].astype(int)

unique_edge_ids = edges['osmid'].astype(str).unique()
edge_id_to_idx = {oid: i for i, oid in enumerate(unique_edge_ids)}

edges['original_osmid'] = edges['osmid']
edges['osmid'] = edges['osmid'].astype(str).map(edge_id_to_idx)

# Save to CSV
nodes.to_csv('london_nodes_master.csv', index=False)
edges.to_csv('london_edges_master.csv', index=False)

print("Master map files saved")