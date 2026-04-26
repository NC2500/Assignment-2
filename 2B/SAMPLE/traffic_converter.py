"""
Traffic Flow to Travel Time Converter for TBRGS Assignment
Implements the conversion formula from the provided PDF document
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional

class TrafficFlowConverter:
    """
    Converts traffic flow (vehicles per hour) to travel time based on 
    the fundamental diagram of traffic flow and assumptions from the assignment
    """
    
    def __init__(self, speed_limit: float = 60.0, intersection_delay: float = 30.0):
        """
        Initialize the converter with parameters from the assignment
        
        Args:
            speed_limit (float): Speed limit in km/h (default: 60 km/h as per assignment)
            intersection_delay (float): Average delay per intersection in seconds (default: 30 seconds)
        """
        self.speed_limit = speed_limit  # km/h
        self.intersection_delay = intersection_delay  # seconds
        
        # Parameters from the fundamental diagram equation in the PDF:
        # flow = -1.4648375 * (speed)^2 + 93.75 * (speed)
        # Rearranged to solve for speed given flow:
        # -1.4648375 * speed^2 + 93.75 * speed - flow = 0
        self.a = -1.4648375
        self.b = 93.75
        self.c = 0  # Will be set to -flow when solving
        
        # Calculate critical flow (maximum flow before congestion)
        # This occurs at the vertex of the parabola: speed = -b/(2a)
        self.critical_speed = -self.b / (2 * self.a)
        self.critical_flow = self.a * self.critical_speed**2 + self.b * self.critical_speed
        
        print(f"TrafficFlowConverter initialized:")
        print(f"  Speed limit: {self.speed_limit} km/h")
        print(f"  Intersection delay: {self.intersection_delay} seconds")
        print(f"  Critical speed: {self.critical_speed:.2f} km/h")
        print(f"  Critical flow: {self.critical_flow:.2f} vehicles/hour")
    
    def flow_to_speed(self, flow: float) -> float:
        """
        Convert traffic flow to speed using the fundamental diagram
        
        Args:
            flow (float): Traffic flow in vehicles per hour
            
        Returns:
            float: Speed in km/h
        """
        # Handle edge cases
        if flow <= 0:
            return self.speed_limit  # Free flow condition
        
        # Solve the quadratic equation: -1.4648375 * speed^2 + 93.75 * speed - flow = 0
        # Using quadratic formula: speed = (-b ± sqrt(b^2 - 4ac)) / (2a)
        # Where a = -1.4648375, b = 93.75, c = -flow
        
        a = self.a
        b = self.b
        c = -flow
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            # No real solution, return critical speed as fallback
            return self.critical_speed
        
        # Two possible solutions
        sqrt_discriminant = math.sqrt(discriminant)
        speed1 = (-b + sqrt_discriminant) / (2*a)
        speed2 = (-b - sqrt_discriminant) / (2*a)
        
        # Choose the physically meaningful solution (positive speed)
        # For the fundamental diagram, we want the lower speed branch for congested conditions
        # and the upper speed branch for free flow
        
        # Determine if we're in free flow or congested regime
        if flow <= self.critical_flow:
            # Free flow condition - take the higher speed solution
            speed = max(speed1, speed2)
        else:
            # Congested condition - take the lower speed solution
            speed = min(speed1, speed2)
        
        # Ensure speed is within reasonable bounds
        speed = max(0, min(speed, self.speed_limit))
        
        return speed
    
    def calculate_travel_time(self, flow: float, distance_km: float) -> float:
        """
        Calculate travel time for a link based on traffic flow and distance
        
        Args:
            flow (float): Traffic flow in vehicles per hour
            distance_km: Distance of the link in kilometers
            
        Returns:
            float: Travel time in seconds
        """
        # Get speed based on flow
        speed_kmh = self.flow_to_speed(flow)
        
        # Calculate travel time for the link (time = distance / speed)
        if speed_kmh > 0:
            travel_time_hours = distance_km / speed_kmh
            travel_time_seconds = travel_time_hours * 3600
        else:
            # If speed is zero (complete gridlock), use a large travel time
            travel_time_seconds = float('inf')
        
        return travel_time_seconds
    
    def calculate_route_travel_time(self, flows: list, distances: list, 
                                  num_intersections: int) -> float:
        """
        Calculate total travel time for a route consisting of multiple links
        
        Args:
            flows (list): List of traffic flows for each link (vehicles/hour)
            distances (list): List of distances for each link (km)
            num_intersections (int): Number of intersections to pass through
            
        Returns:
            float: Total travel time in seconds
        """
        if len(flows) != len(distances):
            raise ValueError("Flows and distances lists must have the same length")
        
        total_time = 0.0
        
        # Calculate travel time for each link
        for flow, distance in zip(flows, distances):
            link_time = self.calculate_travel_time(flow, distance)
            total_time += link_time
        
        # Add intersection delays
        total_time += num_intersections * self.intersection_delay
        
        return total_time
    
    def get_flow_speed_mapping(self, flow_range: Tuple[float, float] = (0, 500), 
                             num_points: int = 50) -> Dict[float, float]:
        """
        Generate a mapping of flow values to speeds for visualization/testing
        
        Args:
            flow_range (tuple): Min and max flow values to consider
            num_points (int): Number of points to generate
            
        Returns:
            dict: Mapping of flow values to speeds
        """
        flows = np.linspace(flow_range[0], flow_range[1], num_points)
        speeds = [self.flow_to_speed(flow) for flow in flows]
        
        return dict(zip(flows, speeds))

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points on Earth (km)"""
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

def main():
    """
    Demonstrate the traffic flow converter
    """
    print("=== Traffic Flow to Travel Time Converter Demo ===\n")
    
    # Initialize converter
    converter = TrafficFlowConverter(speed_limit=60.0, intersection_delay=30.0)
    
    # Test flow to speed conversion
    print("Flow to Speed Conversion:")
    test_flows = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for flow in test_flows:
        speed = converter.flow_to_speed(flow)
        print(f"  Flow: {flow:3.0f} veh/h -> Speed: {speed:6.2f} km/h")
    
    print()
    
    # Test travel time calculation
    print("Travel Time Calculation (for 1 km link):")
    distance_km = 1.0
    for flow in [0, 100, 200, 300, 400, 500]:
        time_sec = converter.calculate_travel_time(flow, distance_km)
        time_min = time_sec / 60
        print(f"  Flow: {flow:3.0f} veh/h -> Time: {time_sec:6.1f} s ({time_min:4.1f} min)")
    
    print()
    
    # Test route calculation
    print("Route Travel Time Calculation:")
    # Example: 3 links, each 0.5 km, with 2 intersections
    flows = [200, 300, 150]  # vehicles/hour
    distances = [0.5, 0.5, 0.5]  # km
    num_intersections = 2
    
    total_time = converter.calculate_route_travel_time(flows, distances, num_intersections)
    total_time_min = total_time / 60
    
    print(f"  Links: {len(flows)} segments")
    print(f"  Flows: {flows} veh/h")
    print(f"  Distances: {distances} km")
    print(f"  Intersections: {num_intersections}")
    print(f"  Total travel time: {total_time:.1f} s ({total_time_min:.1f} min)")
    
    # Show critical flow point
    print()
    print(f"Critical flow analysis:")
    print(f"  Maximum free-flow speed: {converter.flow_to_speed(0):.2f} km/h")
    print(f"  Speed at critical flow: {converter.flow_to_speed(converter.critical_flow):.2f} km/h")
    print(f"  Speed at saturation (high flow): {converter.flow_to_speed(1000):.2f} km/h")

if __name__ == "__main__":
    main()
