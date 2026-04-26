"""
Traffic-Based Route Guidance System (TBRGS)
Integrates ML traffic prediction with graph search to find optimal routes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math
import os

from graph_builder import BoroondaraGraphBuilder, SCATSNode
from traffic_predictor import TrafficPredictor, load_scats_data
from traffic_converter import TrafficFlowConverter
from config import TrafficConfig, PathConfig, DATA_DIR, MODELS_DIR
from search import read_file, format_output  # From Part A
from DFS import DFS
from BFS import BFS
from GBFS import GBFS
from AStar import AStar
from CUS1 import CUS1
from CUS2 import CUS2


@dataclass
class Route:
    """Represents a calculated route with travel time and path"""
    path: List[str]  # List of SCATS IDs
    total_travel_time: float  # seconds
    edge_times: List[float]  # travel time for each edge
    origin: str
    destination: str

    @property
    def travel_time_minutes(self) -> float:
        return self.total_travel_time / 60.0

    def __str__(self):
        path_str = " -> ".join(self.path)
        return f"Route: {path_str}\nTime: {self.travel_time_minutes:.1f} min ({self.total_travel_time:.0f} sec)"


class TBRGS:
    """
    Traffic-Based Route Guidance System
    Combines traffic prediction, travel time estimation, and path finding
    """

    def __init__(self, model_type: str = 'lstm', model_path: Optional[str] = None,
                 sequence_length: int = 24, use_cached_predictions: bool = True):
        """
        Initialize the TBRGS system

        Args:
            model_type: ML model type for traffic prediction
            model_path: Path to trained model weights
            sequence_length: Historical sequence length for predictions
            use_cached_predictions: Whether to cache traffic flow predictions
        """
        print("\n=== Initializing TBRGS ===\n")

        # Initialize components
        self.converter = TrafficFlowConverter(
            speed_limit=TrafficConfig.speed_limit_kmh,
            intersection_delay=TrafficConfig.intersection_delay_seconds
        )

        self.graph_builder = BoroondaraGraphBuilder()
        self.graph_builder.build_graph(method='hybrid')

        self.graph = self.graph_builder.graph
        self.coordinates = self.graph_builder.get_coordinates_dict()
        self.nodes = self.graph_builder.get_nodes_dict()

        # Initialize traffic predictor
        self.predictor = TrafficPredictor(
            model_type=model_type,
            model_path=model_path,
            sequence_length=sequence_length
        )

        # Load historical traffic data for all sites
        print("\nLoading historical traffic data...")
        self.historical_data = load_scats_data(DATA_DIR)

        # Prediction cache
        self.use_cached_predictions = use_cached_predictions
        self.flow_cache: Dict[str, float] = {}  # site_id -> predicted flow

        print("\nTBRGS initialized successfully!")
        self.print_status()

    def print_status(self):
        """Print system status"""
        print(f"Graph nodes: {len(self.graph)}")
        print(f"Historical data available for {len(self.historical_data)} sites")
        print(f"ML Model: {self.predictor.model_type.upper()}")
        print(f"Speed limit: {self.converter.speed_limit} km/h")
        print(f"Intersection delay: {self.converter.intersection_delay} s")

    def predict_flow(self, scats_id: str, future_steps: int = 4) -> float:
        """
        Predict traffic flow for a SCATS site

        Args:
            scats_id: SCATS site identifier
            future_steps: Number of 15-min intervals ahead to predict

        Returns:
            Predicted traffic flow (vehicles per hour)
        """
        cache_key = f"{scats_id}_{future_steps}"

        if self.use_cached_predictions and cache_key in self.flow_cache:
            return self.flow_cache[cache_key]

        if scats_id not in self.historical_data:
            # No historical data, return average of all sites
            all_flows = []
            for data in self.historical_data.values():
                all_flows.extend(data)
            prediction = np.nanmean(all_flows) if all_flows else 200.0
        else:
            data = self.historical_data[scats_id]
            prediction = self.predictor.predict(data, future_steps=future_steps)
            prediction = prediction[0] if len(prediction) > 0 else 0.0

        if self.use_cached_predictions:
            self.flow_cache[cache_key] = prediction

        return float(prediction)

    def get_edge_travel_time(self, from_node: str, to_node: str,
                            hour_of_day: Optional[int] = None) -> float:
        """
        Calculate travel time for an edge (road segment) based on predicted traffic

        Args:
            from_node: Origin SCATS ID
            to_node: Destination SCATS ID
            hour_of_day: Current hour (for context, not used yet)

        Returns:
            Travel time in seconds
        """
        # Get distance between nodes from precomputed graph edges
        distance_km = None
        for neighbor, dist in self.graph.get(from_node, []):
            if neighbor == to_node:
                distance_km = dist
                break

        if distance_km is None:
            # Fallback: calculate from coordinates
            lat1, lon1 = self.coordinates.get(from_node, (0, 0))
            lat2, lon2 = self.coordinates.get(to_node, (0, 0))
            if lat1 == 0 or lat2 == 0:
                return float('inf')
            distance_km = self.converter.haversine_distance(lat1, lon1, lat2, lon2)

        # Predict traffic flow for the destination node (as per assignment)
        # "travel time from a SCATS site A to a SCATS site B can be approximated
        # by ... the accumulated volume per hour at the SCATS site B"
        flow = self.predict_flow(to_node, future_steps=1)

        # Calculate travel time based on flow at destination
        travel_time = self.converter.calculate_travel_time(flow, distance_km)

        return travel_time

    def calculate_route_travel_time(self, path: List[str]) -> Tuple[float, List[float]]:
        """
        Calculate total travel time for a given path

        Args:
            path: List of SCATS IDs representing the route

        Returns:
            (total_time_seconds, edge_times_list)
        """
        if len(path) < 2:
            return 0.0, []

        total_time = 0.0
        edge_times = []

        # Time for each road segment (link)
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i+1]
            edge_time = self.get_edge_travel_time(from_node, to_node)
            edge_times.append(edge_time)
            total_time += edge_time

        # Add intersection delays (one per intermediate intersection)
        # Number of intersections passed = len(path) - 2 (excluding start and end)
        num_intersections = max(0, len(path) - 2)
        intersection_delay_total = num_intersections * self.converter.intersection_delay
        total_time += intersection_delay_total

        return total_time, edge_times

    def find_top_k_paths(self, origin: str, destination: str, k: int = 5,
                        method: str = 'astar') -> List[Route]:
        """
        Find top-k routes from origin to destination using traffic-weighted costs.
        Implements a penalty-based approach to generate alternative distinct paths.

        Args:
            origin: Origin SCATS ID
            destination: Destination SCATS ID
            k: Number of routes to return
            method: Search method ('astar', 'bfs', 'dfs', 'gbfs', 'cus1', 'cus2')

        Returns:
            List of Route objects sorted by travel time
        """
        print(f"\nFinding top {k} routes from {origin} to {destination} using {method.upper()}...")

        # Precompute base edge travel times for all directed edges
        base_edge_times = {}
        for u in self.graph:
            for v, _ in self.graph[u]:
                base_edge_times[(u, v)] = self.get_edge_travel_time(u, v)

        # Helper to build weighted graph from edge times with optional penalties
        def build_weighted_graph(penalty_edges=None, penalty_factor=2.0):
            if penalty_edges is None:
                penalty_edges = set()
            weighted = {}
            for u in self.graph:
                weighted[u] = []
                for v, _ in self.graph[u]:
                    t = base_edge_times[(u, v)]
                    if (u, v) in penalty_edges:
                        t = t * penalty_factor
                    weighted[u].append((v, t))
            return weighted

        # Helper to execute search algorithm
        def execute_search(graph):
            goals = [destination]
            if method == 'astar':
                _, _, path = AStar(graph, self.coordinates, origin, goals)
            elif method == 'bfs':
                _, _, path = BFS(graph, origin, goals)
            elif method == 'dfs':
                _, _, path = DFS(graph, origin, goals)
            elif method == 'gbfs':
                _, _, path = GBFS(graph, self.coordinates, origin, goals)
            elif method == 'cus1':
                _, _, path = CUS1(graph, origin, goals)
            elif method == 'cus2':
                _, _, path = CUS2(graph, self.coordinates, origin, goals)
            else:
                _, _, path = AStar(graph, self.coordinates, origin, goals)
            return path

        # Obtain the initial (best) path
        base_graph = build_weighted_graph()
        base_path = execute_search(base_graph)

        if not base_path or len(base_path) < 2:
            print(f"No route found from {origin} to {destination}")
            return []

        routes = []
        paths_seen = set()  # store tuple of nodes to avoid duplicates

        def add_route(path):
            if tuple(path) in paths_seen:
                return False
            paths_seen.add(tuple(path))
            total_time, edge_times_list = self.calculate_route_travel_time(path)
            route = Route(
                path=path,
                total_travel_time=total_time,
                edge_times=edge_times_list,
                origin=origin,
                destination=destination
            )
            routes.append(route)
            return True

        # Add base route
        add_route(base_path)

        # For methods that don't consider edge costs, we likely only get one path.
        # For cost-sensitive methods, we use edge penalty to find alternatives.
        cost_sensitive = method in ['astar']  # only A* uses accumulated cost; others can also be extended but let's keep it simple

        if cost_sensitive and k > 1:
            used_edges = set()
            # Add edges from base path to used set
            for i in range(len(base_path)-1):
                used_edges.add((base_path[i], base_path[i+1]))

            # Try to find additional distinct paths
            max_attempts = 50
            attempts = 0
            while len(routes) < k and attempts < max_attempts:
                attempts += 1
                # Build graph with penalty on used edges
                penalty_graph = build_weighted_graph(penalty_edges=used_edges, penalty_factor=2.0)
                alt_path = execute_search(penalty_graph)

                if not alt_path:
                    continue

                # Check if it's a new distinct path
                if tuple(alt_path) not in paths_seen:
                    add_route(alt_path)
                    # Add its edges to used_edges to encourage further diversity
                    for i in range(len(alt_path)-1):
                        used_edges.add((alt_path[i], alt_path[i+1]))
                else:
                    # Even if duplicate, we still add its edges to used_edges to push search further?
                    # Actually we want to penalize those edges more, so maybe add anyway
                    for i in range(len(alt_path)-1):
                        used_edges.add((alt_path[i], alt_path[i+1]))

        # If still not enough routes and initial path length > 3, try simple node skipping (if allowed)
        # Only as fallback
        if len(routes) < k and len(base_path) > 3:
            for skip_idx in range(1, len(base_path)-1):
                if len(routes) >= k:
                    break
                alt_path = base_path[:skip_idx] + base_path[skip_idx+1:]
                # Check simple connectivity: for each consecutive pair, ensure there is an edge in graph
                valid = True
                for i in range(len(alt_path)-1):
                    u, v = alt_path[i], alt_path[i+1]
                    # check if v in neighbors of u
                    if v not in [n for n,_ in self.graph.get(u,[])]:
                        valid = False
                        break
                if valid and tuple(alt_path) not in paths_seen:
                    add_route(alt_path)

        # Sort routes by total travel time
        routes.sort(key=lambda r: r.total_travel_time)

        print(f"Found {len(routes)} routes")
        return routes[:k]

    def _find_path(self, origin: str, destination: str, method: str) -> List[str]:
        """
        Find a path using specified search method with dynamic travel times

        Returns:
            List of node IDs representing the path
        """
        # Build directed weighted graph for this query
        # Edge cost from A to B = travel time based on predicted flow at B
        weighted_graph = {}
        for node in self.graph:
            weighted_graph[node] = []
            for neighbor, _ in self.graph.get(node, []):
                cost = self.get_edge_travel_time(node, neighbor)
                weighted_graph[node].append((neighbor, cost))

        goals = [destination]

        if method == 'astar':
            result_goal, num_nodes, path = AStar(weighted_graph, self.coordinates, origin, goals)
        elif method == 'bfs':
            result_goal, num_nodes, path = BFS(weighted_graph, origin, goals)
        elif method == 'dfs':
            result_goal, num_nodes, path = DFS(weighted_graph, origin, goals)
        elif method == 'gbfs':
            result_goal, num_nodes, path = GBFS(weighted_graph, self.coordinates, origin, goals)
        elif method == 'cus1':
            result_goal, num_nodes, path = CUS1(weighted_graph, origin, goals)
        elif method == 'cus2':
            result_goal, num_nodes, path = CUS2(weighted_graph, self.coordinates, origin, goals)
        else:
            # Fallback to A*
            result_goal, num_nodes, path = AStar(weighted_graph, self.coordinates, origin, goals)

        return path

    def print_route_details(self, routes: List[Route]):
        """Print detailed information about routes"""
        print("\n" + "="*60)
        print("ROUTE RESULTS")
        print("="*60)

        for i, route in enumerate(routes):
            print(f"\nRoute {i+1}:")
            print(f"  Path: {' -> '.join(route.path)}")
            print(f"  Total travel time: {route.travel_time_minutes:.1f} minutes")

            if route.edge_times:
                print(f"  Edge times:")
                for j, (edge_time, (from_n, to_n)) in enumerate(zip(route.edge_times, zip(route.path[:-1], route.path[1:]))):
                    from_name = self.nodes[from_n].location if from_n in self.nodes else from_n
                    to_name = self.nodes[to_n].location if to_n in self.nodes else to_n
                    print(f"    {from_n}->{to_n} ({from_name} -> {to_name}): {edge_time/60:.1f} min")

            # Intersection count
            num_intersections = max(0, len(route.path) - 2)
            if num_intersections > 0:
                print(f"  Intersections passed: {num_intersections}")
                print(f"  Intersection delay: {num_intersections * self.converter.intersection_delay:.0f} seconds")

        print("="*60)

    def get_node_info(self, scats_id: str) -> Optional[SCATSNode]:
        """Get information about a SCATS node"""
        return self.nodes.get(scats_id)

    def get_available_sites(self) -> List[Tuple[str, str]]:
        """Get list of available SCATS sites as (id, location) tuples"""
        return [(node_id, node.location) for node_id, node in self.nodes.items()]


def quick_route_query(origin: str, destination: str, model_type: str = 'lstm',
                     k: int = 5, method: str = 'astar') -> List[Route]:
    """
    Quick function to get routes without full initialization

    Args:
        origin: Origin SCATS ID
        destination: Destination SCATS ID
        model_type: ML model type
        k: Number of routes
        method: Search algorithm

    Returns:
        List of Route objects
    """
    tbrgs = TBRGS(model_type=model_type)
    return tbrgs.find_top_k_paths(origin, destination, k=k, method=method)


if __name__ == "__main__":
    # Quick test
    print("Testing TBRGS system...")
    system = TBRGS(model_type='lstm')

    # Example from assignment
    origin = '2000'
    destination = '3002'

    routes = system.find_top_k_paths(origin, destination, k=3, method='astar')

    if routes:
        system.print_route_details(routes)
    else:
        print("No routes found. Graph may not be sufficiently connected.")
