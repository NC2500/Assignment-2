"""
Graph Builder for Boroondara Road Network
Constructs a graph from SCATS metadata where nodes are intersections and edges are road segments
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import os

from config import GraphConfig


@dataclass
class SCATSNode:
    """Represents a SCATS intersection node"""
    id: str
    location: str
    latitude: float
    longitude: float
    data_points: int

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class BoroondaraGraphBuilder:
    """
    Builds a road network graph for the Boroondara area from SCATS metadata
    Nodes are intersections, edges are road segments with distances
    """

    def __init__(self, metadata_file: str = 'processed_data/scats_metadata.csv'):
        self.metadata_file = metadata_file
        self.nodes: Dict[str, SCATSNode] = {}
        self.graph: Dict[str, List[Tuple[str, float]]] = {}  # node_id -> [(neighbor, distance)]
        self.node_coordinates: Dict[str, Tuple[float, float]] = {}

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth
        Returns distance in kilometers
        """
        R = 6371.0  # Earth radius in kilometers

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def extract_road_name(self, location: str) -> str:
        """
        Extract the main road name from location string
        Examples:
          "WARRIGAL_RD N of TOORAK_RD" -> "WARRIGAL_RD"
          "HIGH_ST NE of BARKERS_RD" -> "HIGH_ST"
        """
        if not location or pd.isna(location):
            return ""

        # Remove direction and 'of' parts
        parts = location.split()
        if len(parts) >= 1:
            # The first part is typically the road name
            road = parts[0]
            # Clean up any non-alphanumeric chars at end
            return road.upper().strip()
        return ""

    def load_metadata(self) -> pd.DataFrame:
        """Load and parse the SCATS metadata CSV"""
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        df = pd.read_csv(self.metadata_file)
        print(f"Loaded metadata for {len(df)} SCATS sites")
        return df

    def build_nodes(self, df: pd.DataFrame, site_ids: Optional[List[str]] = None):
        """Create node objects from metadata"""
        if site_ids:
            df = df[df['SCATS_ID'].astype(str).isin(site_ids)]

        for _, row in df.iterrows():
            scats_id = str(row['SCATS_ID'])
            self.nodes[scats_id] = SCATSNode(
                id=scats_id,
                location=row['Location'],
                latitude=float(row['Latitude']) if pd.notna(row['Latitude']) else 0.0,
                longitude=float(row['Longitude']) if pd.notna(row['Longitude']) else 0.0,
                data_points=int(row['Data_Points'])
            )
            self.node_coordinates[scats_id] = (self.nodes[scats_id].latitude,
                                               self.nodes[scats_id].longitude)

        print(f"Built {len(self.nodes)} nodes in Boroondara network")

    def build_edges_by_road_name(self, df: pd.DataFrame):
        """
        Build edges by connecting SCATS sites that share the same main road.
        This is a heuristic approach: we connect consecutive intersections on the same road.
        """
        # Add road name and sorting position based on location description
        df['SCATS_ID_str'] = df['SCATS_ID'].astype(str)
        df['Road_Name'] = df['Location'].apply(self.extract_road_name)

        edges_added = 0

        # For each road, connect sites in order (we need to determine ordering)
        # We'll use latitude/longitude to approximate order along the road
        for road_name in df['Road_Name'].unique():
            if not road_name or road_name == "":
                continue

            road_sites = df[df['Road_Name'] == road_name].copy()

            # For this simple version, we connect ALL pairs on the same road (complete connectivity)
            # A more refined version would order them and connect only adjacent ones
            # But since we have limited data, a sparse connectivity approach is safer

            # Get valid sites (with coordinates) from our node set
            valid_sites = []
            for _, row in road_sites.iterrows():
                scats_id = str(row['SCATS_ID'])
                if scats_id in self.nodes:
                    valid_sites.append(scats_id)

            # Connect each site to other sites on the same road (bidirectional)
            for i, site_a in enumerate(valid_sites):
                for site_b in valid_sites[i+1:]:
                    # Calculate distance
                    lat1, lon1 = self.node_coordinates[site_a]
                    lat2, lon2 = self.node_coordinates[site_b]

                    # Skip if coordinates are invalid
                    if lat1 == 0 or lat2 == 0:
                        continue

                    dist = self.haversine_distance(lat1, lon1, lat2, lon2)

                    # Only add edges for reasonable distances (e.g., < 3 km)
                    # For our small area, most segments will be < 2km
                    if dist < 0.1:  # Skip if too close (duplicate or same location)
                        continue
                    if dist > 5.0:  # Too far to be directly connected
                        continue

                    # Add bidirectional edges
                    if site_a not in self.graph:
                        self.graph[site_a] = []
                    if site_b not in self.graph:
                        self.graph[site_b] = []

                    self.graph[site_a].append((site_b, dist))
                    self.graph[site_b].append((site_a, dist))
                    edges_added += 2

        print(f"Added {edges_added} edges to the graph (bidirectional count)")

    def build_edges_by_nearest_neighbor(self, max_distance: float = 2.0, k_neighbors: int = 5):
        """
        Alternative edge building: connect each node to its k nearest neighbors
        within max_distance. This creates a more connected graph.
        """
        nodes_list = list(self.nodes.keys())
        edges_added = 0

        for i, node_a in enumerate(nodes_list):
            lat_a, lon_a = self.node_coordinates[node_a]
            if lat_a == 0:
                continue

            # Compute distances to all other nodes
            distances = []
            for node_b in nodes_list:
                if node_b == node_a:
                    continue
                lat_b, lon_b = self.node_coordinates[node_b]
                if lat_b == 0:
                    continue
                dist = self.haversine_distance(lat_a, lon_a, lat_b, lon_b)
                distances.append((node_b, dist))

            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[1])
            nearest = distances[:k_neighbors]

            for neighbor, dist in nearest:
                if dist <= max_distance:
                    if node_a not in self.graph:
                        self.graph[node_a] = []
                    if neighbor not in self.graph:
                        self.graph[neighbor] = []

                    # Check if edge already exists
                    exists = any(neighbor == n for n, d in self.graph[node_a])
                    if not exists:
                        self.graph[node_a].append((neighbor, dist))
                        self.graph[neighbor].append((node_a, dist))
                        edges_added += 2

        print(f"Added {edges_added} edges using nearest neighbor approach")

    def ensure_all_nodes_exist(self):
        """Ensure all nodes have an entry in the graph dict"""
        for node_id in self.nodes:
            if node_id not in self.graph:
                self.graph[node_id] = []

    def sort_adjacency(self):
        """Sort adjacency lists by node ID for deterministic tie-breaking"""
        for node_id in self.graph:
            self.graph[node_id].sort(key=lambda item: item[0])

    def build_graph(self, method: str = 'hybrid') -> Dict[str, List[Tuple[str, float]]]:
        """
        Build the complete graph

        Args:
            method: 'road_name' or 'nearest' or 'hybrid'

        Returns:
            Graph dictionary: {node_id: [(neighbor, distance_km), ...]}
        """
        print(f"\n=== Building Boroondara Graph (method: {method}) ===")

        df = self.load_metadata()
        self.build_nodes(df, site_ids=GraphConfig.BOROONDARA_SITES)

        if method == 'road_name':
            self.build_edges_by_road_name(df)
        elif method == 'nearest':
            self.build_edges_by_nearest_neighbor(max_distance=2.0, k_neighbors=4)
        elif method == 'hybrid':
            # First use road name approach, then fill gaps with nearest neighbor
            self.build_edges_by_road_name(df)
            self.build_edges_by_nearest_neighbor(max_distance=2.0, k_neighbors=2)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.ensure_all_nodes_exist()
        self.sort_adjacency()

        # Print statistics
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        print(f"Graph built: {len(self.graph)} nodes, ~{total_edges} edges")
        print(f"Sample connections:")
        for node_id in list(self.graph.keys())[:3]:
            node_name = self.nodes[node_id].location if node_id in self.nodes else node_id
            connections = self.graph[node_id]
            print(f"  {node_id} ({node_name}): {len(connections)} neighbors")
            for neighbor, dist in connections[:3]:
                neighbor_name = self.nodes[neighbor].location if neighbor in self.nodes else neighbor
                print(f"    -> {neighbor} ({neighbor_name}): {dist:.3f} km")

        return self.graph

    def get_coordinates_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return node coordinates for use in pathfinding"""
        return self.node_coordinates

    def get_nodes_dict(self) -> Dict[str, SCATSNode]:
        """Return all SCATS nodes"""
        return self.nodes


def main():
    """Test the graph builder"""
    builder = BoroondaraGraphBuilder()
    graph = builder.build_graph(method='hybrid')
    coords = builder.get_coordinates_dict()
    nodes = builder.get_nodes_dict()

    print(f"\nTotal nodes: {len(graph)}")
    print(f"Sample node 2000: {nodes.get('2000', 'Not found')}")
    print(f"Sample node 3002: {nodes.get('3002', 'Not found')}")

    # Check if example origin/destination are connected
    if '2000' in graph and '3002' in graph:
        # Simple BFS to see if path exists
        from collections import deque
        visited = set()
        queue = deque([('2000', [])])
        found = False
        while queue:
            current, path = queue.popleft()
            if current == '3002':
                print(f"\nPath from 2000 to 3002 exists! Example: {path + [current]}")
                found = True
                break
            if current in visited:
                continue
            visited.add(current)
            for neighbor, _ in graph.get(current, []):
                queue.append((neighbor, path + [current]))
        if not found:
            print("\nNo direct path from 2000 to 3002 (need to add more connections)")


if __name__ == '__main__':
    main()
