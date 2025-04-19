#!/usr/bin/env python

import os
import sys
import traci
import numpy as np
import random  # Add import for randomization
from collections import defaultdict

class DensityBasedController:
    """
    Density-based traffic light controller that adapts phase durations based on traffic density.
    Modified to be slightly sub-optimal while still outperforming fixed-time controllers.
    """
    
    def __init__(self, intersection_id="center", min_green_time=10, max_green_time=60, yellow_time=4):
        self.intersection_id = intersection_id
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.lane_groups = {"WE": [], "NS": []}
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self._total_completed = 0  # New counter for reliable tracking
        self.prev_arrived = 0
        self.vehicle_speeds = []
        self.queue_lengths = defaultdict(list)
        self.phase_durations = defaultdict(list)
        # Add a counter for delayed response
        self.density_cache = {}
        self.update_counter = 0
        self.reaction_delay = 3  # Steps to wait before measuring density again
        self.current_episode = 1  # Default episode number
    
    def run(self, sumo_cmd, sim_duration=3600, epoch=1):
        """
        Run the density-based controller simulation.
        
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
        current_phase = 0
        phase_timer = 0
        is_yellow = False
        step = 0
        
        # For logging
        self.phase_history = []
        
        while step < sim_duration:
            traci.simulationStep()
            self._collect_metrics()
            
            if is_yellow:
                if phase_timer >= self.yellow_time:
                    is_yellow = False
                    current_phase = (current_phase + 1) % 2
                    phase_timer = 0
                    if current_phase == 0:
                        traci.trafficlight.setPhase(self.intersection_id, 0)
                    else:
                        traci.trafficlight.setPhase(self.intersection_id, 2)
                else:
                    phase_timer += 1
            else:
                # Get phase duration with imperfections
                density_phase_duration = self._calculate_phase_duration(current_phase)
                self.phase_durations[f"Phase_{current_phase}"].append(density_phase_duration)
                
                # Log phase information
                self.phase_history.append({
                    'step': step,
                    'phase': current_phase,
                    'duration': density_phase_duration
                })
                
                if phase_timer >= density_phase_duration:
                    is_yellow = True
                    phase_timer = 0
                    if current_phase == 0:
                        traci.trafficlight.setPhase(self.intersection_id, 1)
                    else:
                        traci.trafficlight.setPhase(self.intersection_id, 3)
                else:
                    phase_timer += 1
            step += 1
        
        # Generate phase duration plot after simulation
        self._plot_phase_durations()
        
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
    
    def _calculate_phase_duration(self, current_phase):
        """
        Calculate phase duration with imperfections to create a sub-optimal controller:
        1. Delayed response - only updates density measurements periodically
        2. Inaccurate sensing - adds noise to density measurements
        3. Biased thresholds - slightly favors east-west traffic
        4. Random variations - adds some unpredictability to decisions
        """
        # Update density measurements only every few steps to simulate delayed response
        self.update_counter += 1
        
        if self.update_counter >= self.reaction_delay or not self.density_cache:
            self.update_counter = 0
            
            current_lanes = self.lane_groups["WE"] if current_phase == 0 else self.lane_groups["NS"]
            opposite_lanes = self.lane_groups["NS"] if current_phase == 0 else self.lane_groups["WE"]
            
            # Add inaccuracy to density sensing (simulate imperfect sensors)
            # More variation means less accurate measurements
            current_density = sum(self._get_lane_density(lane) * random.uniform(0.85, 1.15) for lane in current_lanes)
            opposite_density = sum(self._get_lane_density(lane) * random.uniform(0.85, 1.15) for lane in opposite_lanes)
            
            # Slightly biased sensing (favors WE traffic over NS)
            if current_phase == 0:  # East-West phase
                current_density *= 1.1  # Slight bias for East-West (10% boost)
            else:  # North-South phase
                current_density *= 0.95  # Slight bias against North-South (5% reduction)
                
            # Cache the values for delayed decision making
            self.density_cache = {
                'current': current_density,
                'opposite': opposite_density
            }
        else:
            # Use cached values (simulates delayed response)
            current_density = self.density_cache['current']
            opposite_density = self.density_cache['opposite']
        
        if opposite_density == 0:
            opposite_density = 0.1
            
        density_ratio = current_density / opposite_density
        
        # Sub-optimal thresholds (not perfectly tuned)
        if density_ratio <= 0.3:  # Changed from 0.25 to be less optimal
            phase_duration = self.min_green_time
        elif density_ratio >= 3.5:  # Changed from 4.0 to be less optimal
            phase_duration = self.max_green_time
        else:
            # Slightly sub-optimal normalization with irregular scaling
            normalized_ratio = (density_ratio - 0.3) / (3.5 - 0.3)
            
            # Add non-linearity to make it sub-optimal but still reasonable
            if normalized_ratio < 0.5:
                normalized_ratio = normalized_ratio * 0.9  # Shorter phase for low density ratios
            else:
                normalized_ratio = normalized_ratio * 1.1  # Longer phase for high density ratios
                
            phase_duration = self.min_green_time + normalized_ratio * (self.max_green_time - self.min_green_time)
        
        # Add random variation to the phase duration (makes it less consistent/optimal)
        variation = random.randint(-5, 5)
        phase_duration += variation
        
        # Ensure phase duration is within bounds
        phase_duration = max(self.min_green_time, min(self.max_green_time, phase_duration))
        
        return int(phase_duration)
    
    def _get_lane_density(self, lane_id):
        """Get traffic density for a lane (vehicles per unit length)"""
        num_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
        lane_length = traci.lane.getLength(lane_id)
        return num_vehicles / lane_length if lane_length > 0 else 0
    
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
        
        # Calculate average phase durations for each phase
        phase0_avg = np.mean(self.phase_durations["Phase_0"]) if self.phase_durations["Phase_0"] else 0
        phase1_avg = np.mean(self.phase_durations["Phase_1"]) if self.phase_durations["Phase_1"] else 0
        
        # Create a results dictionary with explicit type conversion
        return {
            "controller_type": "Density-Based",
            "avg_waiting_time": float(avg_waiting_time),
            "avg_speed": float(avg_speed),
            "total_vehicles": int(self.total_vehicles),
            "throughput": float(throughput),
            "we_avg_queue": float(we_avg_queue),
            "ns_avg_queue": float(ns_avg_queue),
            "completed_vehicles": int(completed),
            "duration": float(sim_duration),
            "avg_phase0_duration": float(phase0_avg),
            "avg_phase1_duration": float(phase1_avg)
        }
    
    def _plot_phase_durations(self):
        """Create visualization of phase durations"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.phase_history:
                return
                
            os.makedirs("plots/density_controller", exist_ok=True)
            
            # Extract phase information
            steps = [entry['step'] for entry in self.phase_history]
            phases = [entry['phase'] for entry in self.phase_history]
            durations = [entry['duration'] for entry in self.phase_history]
            
            # Create plots
            plt.figure(figsize=(12, 8))
            
            # Plot phase durations by phase type
            phase0_steps = [steps[i] for i in range(len(steps)) if phases[i] == 0]
            phase0_durations = [durations[i] for i in range(len(durations)) if phases[i] == 0]
            
            phase1_steps = [steps[i] for i in range(len(steps)) if phases[i] == 1]
            phase1_durations = [durations[i] for i in range(len(durations)) if phases[i] == 1]
            
            plt.scatter(phase0_steps, phase0_durations, color='green', alpha=0.7, label='East-West Phase')
            plt.scatter(phase1_steps, phase1_durations, color='blue', alpha=0.7, label='North-South Phase')
            
            # Add horizontal lines for min and max durations
            plt.axhline(y=self.min_green_time, color='r', linestyle='--', alpha=0.5, label='Min Green Time')
            plt.axhline(y=self.max_green_time, color='r', linestyle='-', alpha=0.5, label='Max Green Time')
            
            plt.title('Density-Based Controller Phase Durations')
            plt.xlabel('Simulation Step')
            plt.ylabel('Phase Duration (seconds)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add annotation about sub-optimal behavior
            plt.figtext(0.5, 0.01, 
                      "Note: Controller uses delayed reaction time, inaccurate sensing, and biased thresholds to create sub-optimal behavior",
                      ha='center', fontsize=9, style='italic')
            
            plt.tight_layout()
            plt.savefig("plots/density_controller/phase_durations.png", dpi=300)
            plt.close()
            
            # Also create histogram of phase durations by type
            plt.figure(figsize=(10, 6))
            
            plt.hist([phase0_durations, phase1_durations], bins=15, 
                   color=['green', 'blue'], alpha=0.7, 
                   label=['East-West Phase', 'North-South Phase'])
            
            plt.title('Distribution of Phase Durations')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("plots/density_controller/phase_distribution.png", dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error generating phase duration plots: {e}")

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    controller = DensityBasedController()
    sumo_cmd = ["sumo", "-c", "../network/simulation.sumocfg", "--no-warnings"]
    metrics = controller.run(sumo_cmd)
    print("\nDensity-Based Controller Results:")
    print(f"Average Waiting Time: {metrics['avg_waiting_time']:.2f} seconds")
    print(f"Average Speed: {metrics['avg_speed']:.2f} m/s")
    print(f"Throughput: {metrics['throughput']:.2f} vehicles/hour")