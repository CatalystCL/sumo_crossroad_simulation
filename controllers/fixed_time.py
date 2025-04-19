#!/usr/bin/env python

import os
import sys
import traci
import numpy as np
from collections import defaultdict

class FixedTimeController:
    """
    Fixed-time traffic light controller with pre-defined phase durations.
    Supports multi-lane intersections.
    """
    
    def __init__(self, intersection_id="center", yellow_time=4):
        self.intersection_id = intersection_id
        self.yellow_time = yellow_time
        self.phase_durations = [31, 31]  # WE, NS green phases
        self.lane_groups = {"WE": [], "NS": []}
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self._total_completed = 0  # New counter for reliable tracking
        self.prev_arrived = 0
        self.vehicle_speeds = []
        self.queue_lengths = defaultdict(list)
        self.current_episode = 1  # Default episode number
        
    def run(self, sumo_cmd, sim_duration=3600, epoch=1):
        """
        Run the fixed time controller simulation.
        
        Args:
            sumo_cmd: SUMO command to run
            sim_duration: Simulation duration in seconds
            epoch: Epoch number for logging
            
        Returns:
            Dictionary with performance metrics
        """
        # Store the epoch number for logging
        self.current_episode = epoch
        
        # Add warning suppression
        sumo_cmd = list(sumo_cmd)  # Make a copy
        if "--no-warnings" not in sumo_cmd:
            sumo_cmd.append("--no-warnings")
        
        traci.start(sumo_cmd)
        self._detect_lane_groups()
        step = 0
        while step < sim_duration:
            traci.simulationStep()
            self._collect_metrics()
            step += 1
        
        metrics = self._calculate_metrics(sim_duration)
        traci.close()
        return metrics
    
    def _detect_lane_groups(self):
        all_lanes = traci.lane.getIDList()
        incoming_lanes = [lane for lane in all_lanes if "to_center" in lane and lane.split("_")[-1].isdigit()]
        
        # Reset lane groups
        self.lane_groups = {"WE": [], "NS": []}
        
        for lane in incoming_lanes:
            if "west_to_center" in lane or "east_to_center" in lane:
                self.lane_groups["WE"].append(lane)
            elif "north_to_center" in lane or "south_to_center" in lane:
                self.lane_groups["NS"].append(lane)
            else:
                print(f"WARNING: Lane {lane} doesn't match expected naming pattern")
        
        print(f"Detected lane groups: WE={len(self.lane_groups['WE'])} lanes, NS={len(self.lane_groups['NS'])} lanes")
    
    def _collect_metrics(self):
        """Collect performance metrics with improved logging format"""
        vehicle_ids = traci.vehicle.getIDList()
        self.total_vehicles += len(vehicle_ids)
        
        # Get controller name for logging
        controller_name = self.__class__.__name__.replace("Controller", "")
        
        # Get current arrived count directly from SUMO
        current_arrived = traci.simulation.getArrivedNumber()
        
        # Calculate newly arrived vehicles
        newly_arrived = current_arrived - self.prev_arrived
        
        # Update counters
        if newly_arrived > 0:
            self._total_completed += newly_arrived
            # Also update the completed_vehicles attribute which might be used elsewhere
            self.completed_vehicles = self._total_completed
            
            # New format: [Epoch_number][Controller] vehicles completed routes. Total: X
            print(f"[Epoch {self.current_episode}][{controller_name}] {newly_arrived} vehicles completed routes. Total: {self._total_completed}")
        
        # Update previous arrived count for next step
        self.prev_arrived = current_arrived
        
        # Collect other metrics
        for vehicle_id in vehicle_ids:
            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
            self.total_waiting_time += waiting_time
            speed = traci.vehicle.getSpeed(vehicle_id)
            self.vehicle_speeds.append(speed)
        
        for lane_id in self.lane_groups["WE"] + self.lane_groups["NS"]:
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            self.queue_lengths[lane_id].append(queue_length)
    
    def _calculate_metrics(self, sim_duration):
        """Calculate final performance metrics with reliable throughput calculation"""
        avg_waiting_time = self.total_waiting_time / max(1, self.total_vehicles)
        avg_speed = np.mean(self.vehicle_speeds) if self.vehicle_speeds else 0
        
        # Use the _total_completed attribute for throughput calculation
        completed = self._total_completed
        
        # Calculate throughput - force to float
        throughput = float((completed / sim_duration) * 3600)
        
        # Get controller name
        controller_name = self.__class__.__name__.replace("Controller", "")
        
        print(f"\n==== {controller_name} Controller Performance ====")
        print(f"Total vehicles: {self.total_vehicles}")
        print(f"Completed vehicles: {completed}")
        print(f"Simulation duration: {sim_duration}s")
        print(f"Throughput: {throughput:.2f} vehicles/hour")
        print(f"Average waiting time: {avg_waiting_time:.2f} seconds")
        print(f"Average speed: {avg_speed:.2f} m/s")
        
        avg_queue_length = {lane: np.mean(lengths) for lane, lengths in self.queue_lengths.items()}
        we_avg_queue = np.mean([avg_queue_length[lane] for lane in self.lane_groups["WE"]]) if self.lane_groups["WE"] else 0
        ns_avg_queue = np.mean([avg_queue_length[lane] for lane in self.lane_groups["NS"]]) if self.lane_groups["NS"] else 0
        
        # Create a results dictionary with explicit type conversion
        return {
            "controller_type": "Fixed-Time",
            "avg_waiting_time": float(avg_waiting_time),
            "avg_speed": float(avg_speed),
            "total_vehicles": int(self.total_vehicles),
            "throughput": float(throughput),
            "we_avg_queue": float(we_avg_queue),
            "ns_avg_queue": float(ns_avg_queue),
            "completed_vehicles": int(completed),
            "duration": float(sim_duration)  # Add duration for recalculation if needed
        }

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    controller = FixedTimeController()
    sumo_cmd = ["sumo", "-c", "../network/simulation.sumocfg", "--no-warnings"]
    metrics = controller.run(sumo_cmd)
    print("Fixed-Time Controller Results:")
    print(f"Average Waiting Time: {metrics['avg_waiting_time']:.2f} seconds")
    print(f"Average Speed: {metrics['avg_speed']:.2f} m/s")
    print(f"Throughput: {metrics['throughput']:.2f} vehicles/hour")