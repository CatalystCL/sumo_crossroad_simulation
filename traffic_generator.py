#!/usr/bin/env python

import os
import sys
import random
import numpy as np
from xml.dom import minidom

class TrafficGenerator:
    def __init__(self, output_file="network/routes.rou.xml", lanes_per_direction=1):
        """
        Initialize the traffic generator.
        
        Args:
            output_file: Output route file path
            lanes_per_direction: Number of lanes per direction in the network
        """
        self.output_file = output_file
        self.lanes_per_direction = lanes_per_direction
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Define base vehicle types with SUMO's built-in vehicle classes and collision parameters
        self.vehicle_types = [
        {"id": "passenger_car", "vClass": "passenger", "accel": "2.6", "decel": "4.5", "length": "5.0", 
        "width": "1.8", "height": "1.5", "minGap": "2.5", "maxSpeed": "16.67", "color": "0.3,0.3,0.3", "probability": 0.68, 
        "imgFile": "sumo/img/passenger/passenger.png",
        "lcStrategic": "1.0", "lcCooperative": "0.7", "lcSpeedGain": "0.8", "lcKeepRight": "0.5",
        "maxSpeedLat": "0.8", "latAlignment": "center", "lcAssertive": "0.6"},
        
        {"id": "bus", "vClass": "bus", "accel": "1.8", "decel": "3.5", "length": "12.0", 
        "width": "2.5", "height": "3.4", "minGap": "4.0", "maxSpeed": "16.67", "color": "0.1,0.3,0.2", "probability": 0.15,
        "imgFile": "sumo/img/bus/bus.png",
        "lcStrategic": "1.5", "lcCooperative": "0.5", "lcSpeedGain": "0.2", "lcKeepRight": "0.8",
        "maxSpeedLat": "0.5", "latAlignment": "center", "lcAssertive": "0.2"},
        
        {"id": "truck", "vClass": "truck", "accel": "1.3", "decel": "3.0", "length": "15.0", 
        "width": "2.5", "height": "3.8", "minGap": "5.0", "maxSpeed": "16.67", "color": "0.2,0.2,0.4", "probability": 0.15,
        "imgFile": "sumo/img/truck/truck.png",
        "lcStrategic": "2.0", "lcCooperative": "0.3", "lcSpeedGain": "0.1", "lcKeepRight": "1.0",
        "maxSpeedLat": "0.3", "latAlignment": "center", "lcAssertive": "0.1"},
        
        {"id": "motorcycle", "vClass": "motorcycle", "accel": "3.5", "decel": "6.0", "length": "2.2", 
        "width": "0.8", "height": "1.5", "minGap": "2.0", "maxSpeed": "16.67", "color": "0.3,0.3,0.1", "probability": 0.02,
        "imgFile": "sumo/img/motorcycle/motorcycle.png",
        "lcStrategic": "0.8", "lcCooperative": "0.5", "lcSpeedGain": "1.2", "lcKeepRight": "0.3",
        "maxSpeedLat": "1.2", "latAlignment": "arbitrary", "lcAssertive": "0.8"}
    ]
        
        # Enhanced variation ranges for vehicle parameters
        self.variation_ranges = {
            "passenger_car": {
                "accel": (0.85, 1.3),
                "decel": (0.9, 1.2),
                "length": (0.9, 1.15),
                "width": (0.95, 1.05),
                "maxSpeed": (0.8, 1.2),
                "color_variation": 0.15,
                # Lane change parameter variations
                "lcStrategic": (0.8, 1.2),
                "lcCooperative": (0.5, 0.9),
                "lcSpeedGain": (0.6, 1.0),
                "lcKeepRight": (0.3, 0.7),
                "maxSpeedLat": (0.8, 1.2),
                "lcAssertive": (0.4, 0.8)
            },
            "bus": {
                "accel": (0.85, 1.1),
                "decel": (0.9, 1.1),
                "length": (0.95, 1.1),
                "width": (0.98, 1.02),
                "maxSpeed": (0.85, 1.15),
                "color_variation": 0.1,
                # Lane change parameter variations - buses less varied
                "lcStrategic": (1.3, 1.7),
                "lcCooperative": (0.4, 0.6),
                "lcSpeedGain": (0.1, 0.3),
                "lcKeepRight": (0.7, 0.9),
                "maxSpeedLat": (0.5, 0.7),
                "lcAssertive": (0.1, 0.3)
            },
            "truck": {
                "accel": (0.85, 1.15),
                "decel": (0.9, 1.1),
                "length": (0.95, 1.1),
                "width": (0.98, 1.02),
                "maxSpeed": (0.85, 1.15),
                "color_variation": 0.1,
                # Lane change parameter variations - trucks least varied
                "lcStrategic": (1.8, 2.2),
                "lcCooperative": (0.2, 0.4),
                "lcSpeedGain": (0.05, 0.15),
                "lcKeepRight": (0.9, 1.1),
                "maxSpeedLat": (0.3, 0.5),
                "lcAssertive": (0.05, 0.15)
            },
            "motorcycle": {
                "accel": (0.9, 1.3),
                "decel": (0.9, 1.15),
                "length": (0.9, 1.1),
                "width": (0.9, 1.1),
                "maxSpeed": (0.8, 1.25),
                "color_variation": 0.12,
                # Lane change parameter variations - motorcycles most varied
                "lcStrategic": (0.6, 1.0),
                "lcCooperative": (0.4, 0.6),
                "lcSpeedGain": (1.0, 1.4),
                "lcKeepRight": (0.2, 0.4),
                "maxSpeedLat": (1.2, 1.6),
                "lcAssertive": (0.7, 0.9)
            }
        }
        
        # Generate vehicle variations
        self.vehicle_variations = self._generate_vehicle_variations()
        
        # Define routes for the intersection
        self.routes = []
        self._generate_routes()
        
        # Define direction groups
        self.directions = {
            "west": [route["id"] for route in self.routes if route["id"].startswith("route_W")],
            "east": [route["id"] for route in self.routes if route["id"].startswith("route_E")],
            "north": [route["id"] for route in self.routes if route["id"].startswith("route_N")],
            "south": [route["id"] for route in self.routes if route["id"].startswith("route_S")]
        }
        
        # Initialize flow_id counter as instance variable
        self.flow_id = 0
    
    def _generate_dark_color(self):
        """
        Generate a dark color with RGB values biased toward darker shades.
        
        Returns:
            List of RGB values in the range [0,1]
        """
        # Generate dark base colors - keeping values low for darkness
        r = random.uniform(0.05, 0.4)
        g = random.uniform(0.05, 0.4)
        b = random.uniform(0.05, 0.4)
        
        # Sometimes emphasize one channel slightly for color variation while keeping it dark
        emphasis = random.randint(0, 2)  # 0=R, 1=G, 2=B
        if emphasis == 0:
            r = min(0.55, r * 1.4)
        elif emphasis == 1:
            g = min(0.55, g * 1.4)
        else:
            b = min(0.55, b * 1.4)
            
        return [r, g, b]
    
    def _generate_vehicle_variations(self, variations_per_type=10):
        """
        Generate variations of each vehicle type with differences in appearance and performance.
        Enhanced with better collision avoidance parameters.
        
        Args:
            variations_per_type: Number of variations to generate for each vehicle type
            
        Returns:
            Dictionary of vehicle type variations
        """
        variations = []
        
        for vtype in self.vehicle_types:
            base_id = vtype["id"]
            base_probability = vtype["probability"] / variations_per_type
            
            # Get variation ranges for this vehicle type
            var_range = self.variation_ranges.get(base_id, {
                "accel": (0.9, 1.1),
                "decel": (0.9, 1.1),
                "length": (0.95, 1.05),
                "width": (0.98, 1.02),
                "maxSpeed": (0.95, 1.05),
                "color_variation": 0.1
            })
            
            # Create variations
            for i in range(variations_per_type):
                # Generate dark colors for each variation
                if random.random() < 0.7:
                    varied_color = self._generate_dark_color()
                else:
                    # Parse base color and apply small variations
                    base_color = [float(c) for c in vtype["color"].split(",")]
                    varied_color = []
                    for c in base_color:
                        varied_c = c + random.uniform(-var_range["color_variation"], var_range["color_variation"])
                        varied_c = max(0.05, min(0.5, varied_c))  # Clamp to keep it dark
                        varied_color.append(varied_c)
                
                # Determine vehicle-specific safety parameters based on type
                # Larger/heavier vehicles need more space and time to react
                if base_id == "passenger_car":
                    min_gap = float(vtype["minGap"]) * random.uniform(1.5, 2.0)  # Increased from default
                    tau_value = "1.2"  # Reaction time (increased)
                    sigma_value = "0.2"  # Driver imperfection (reduced)
                    decel_factor = random.uniform(1.0, 1.1)  # Slightly improved deceleration
                    emergency_decel_factor = 2.2  # Increased emergency deceleration
                    jm_drive_red_speed = "0"  # Don't drive at red lights
                elif base_id == "bus":
                    min_gap = float(vtype["minGap"]) * random.uniform(1.8, 2.2)  # Buses need more gap
                    tau_value = "1.5"  # Buses need more reaction time
                    sigma_value = "0.1"  # Bus drivers are more consistent
                    decel_factor = random.uniform(0.9, 1.0)  # Normal deceleration
                    emergency_decel_factor = 1.8  # Limited emergency deceleration
                    jm_drive_red_speed = "0"  # Don't drive at red lights
                elif base_id == "truck":
                    min_gap = float(vtype["minGap"]) * random.uniform(2.0, 2.5)  # Trucks need even more gap
                    tau_value = "1.7"  # Trucks need more reaction time
                    sigma_value = "0.1"  # Truck drivers are more consistent
                    decel_factor = random.uniform(0.8, 0.9)  # Limited deceleration
                    emergency_decel_factor = 1.6  # Limited emergency deceleration
                    jm_drive_red_speed = "0"  # Don't drive at red lights
                else:  # motorcycle or others
                    min_gap = float(vtype["minGap"]) * random.uniform(1.2, 1.5)  # Still increased but less
                    tau_value = "0.9"  # Faster reaction time
                    sigma_value = "0.3"  # More variable driving
                    decel_factor = random.uniform(1.1, 1.2)  # Better deceleration
                    emergency_decel_factor = 2.5  # Stronger emergency deceleration
                    jm_drive_red_speed = "0"  # Don't drive at red lights
                
                # Create the varied vehicle type with SUMO vehicle class
                varied_vtype = {
                    "id": f"{base_id}_{i}",
                    "vClass": vtype["vClass"],
                    "accel": str(float(vtype["accel"]) * random.uniform(var_range["accel"][0], var_range["accel"][1])),
                    "decel": str(float(vtype["decel"]) * decel_factor),
                    "emergencyDecel": str(float(vtype["decel"]) * emergency_decel_factor),  # Increased emergency deceleration
                    "length": str(float(vtype["length"]) * random.uniform(var_range["length"][0], var_range["length"][1])),
                    "width": str(float(vtype["width"]) * random.uniform(var_range["width"][0], var_range["width"][1])),
                    "height": vtype["height"],
                    "minGap": str(min_gap),  # Increased minimum gap for safety
                    "maxSpeed": str(float(vtype["maxSpeed"]) * random.uniform(var_range["maxSpeed"][0], var_range["maxSpeed"][1])),
                    "color": ",".join([str(c) for c in varied_color]),
                    "probability": base_probability,
                    "base_type": base_id,
                    "imgFile": vtype["imgFile"],
                    
                    # Enhanced parameters to prevent intersection blocking and collisions
                    "jmDriveAfterRedTime": "-1",  # Never drive after red
                    "jmDriveRedSpeed": jm_drive_red_speed,
                    "jmIgnoreKeepClearTime": "-1",  # Never ignore keep clear areas
                    "jmIgnoreFoeProb": "0",  # Never ignore other vehicles
                    "jmIgnoreFoeSpeed": "0",  # Never ignore other vehicles regardless of speed
                    "jmSigmaMinor": "0",  # No imperfection at minor roads/intersections
                    "jmTimegapMinor": "2.0",  # Increased time gap at minor roads
                    "impatience": "0",  # No impatience to prevent aggressive driving
                    
                    # Enhanced car-following model parameters
                    "sigma": sigma_value,  # Reduced driver imperfection
                    "tau": tau_value,    # Increased reaction time for safety
                    "speedFactor": str(random.uniform(0.7, 0.9)),  # Drive even slower (70-90% of speed limit)
                    "speedDev": "0.1",  # Low deviation for more predictable behavior
                    
                    # Additional collision avoidance parameters
                    "collisionMinGapFactor": "2.0",  # Double minimum gap during collision avoidance
                    "tauLast": "1.5",  # Time headway for previous vehicle
                    "approachMinGap": "3.0",  # Minimum gap when approaching obstacles
                    "accelLat": "0.0"  # No lateral acceleration (prevents unrealistic lateral movement)
                }
                
                variations.append(varied_vtype)
        
        return variations
    
    def _generate_routes(self):
        """Generate routes based on the number of lanes"""
        # Define route patterns
        patterns = [
            # From West
            {"prefix": "route_W", "from": "west_to_center", "to": ["center_to_east", "center_to_north", "center_to_south"]},
            # From East
            {"prefix": "route_E", "from": "east_to_center", "to": ["center_to_west", "center_to_north", "center_to_south"]},
            # From North
            {"prefix": "route_N", "from": "north_to_center", "to": ["center_to_south", "center_to_east", "center_to_west"]},
            # From South
            {"prefix": "route_S", "from": "south_to_center", "to": ["center_to_north", "center_to_east", "center_to_west"]}
        ]
        
        route_id = 0
        for pattern in patterns:
            from_edge = pattern["from"]
            prefix = pattern["prefix"]
            
            for to_edge in pattern["to"]:
                # For each combination of from-lane and to-lane
                for from_lane in range(self.lanes_per_direction):
                    for to_lane in range(self.lanes_per_direction):
                        route_id_str = f"{prefix}_{route_id}_{from_lane}_{to_lane}"
                        edges = f"{from_edge} {to_edge}"
                        
                        self.routes.append({
                            "id": route_id_str,
                            "edges": edges,
                            "from_lane": from_lane,
                            "to_lane": to_lane
                        })
                        
                        route_id += 1
    
    def generate_traffic_flow(self, sim_duration=3600, stage_duration=1200):
        """
        Generate route file with three sequential stages of traffic density:
        1. Dense on 1 way (west) - 2x denser than before
        2. Low density on all ways
        3. Dense on 3 ways (east, north, south) - 2x denser on east
        
        Args:
            sim_duration: Total simulation duration in seconds
            stage_duration: Duration of each stage in seconds
        """
        # Reset flow ID counter
        self.flow_id = 0
        
        # Adjust stage duration if sim_duration is shorter than expected
        if sim_duration < 3 * stage_duration:
            stage_duration = sim_duration // 3
            print(f"Adjusted stage duration to {stage_duration}s to fit within simulation time")
        
        # Create XML document
        doc = minidom.Document()
        
        # Create root element
        routes_root = doc.createElement('routes')
        doc.appendChild(routes_root)
        
        # Add vehicle type variations - WITH COLLISION AVOIDANCE PARAMETERS
        for vtype in self.vehicle_variations:
            v = doc.createElement('vType')
            v.setAttribute('id', vtype['id'])
            v.setAttribute('vClass', vtype['vClass'])
            v.setAttribute('accel', vtype['accel'])
            v.setAttribute('decel', vtype['decel'])
            v.setAttribute('emergencyDecel', vtype['emergencyDecel'])
            v.setAttribute('length', vtype['length'])
            v.setAttribute('width', vtype['width'])
            v.setAttribute('height', vtype['height'])
            v.setAttribute('minGap', vtype['minGap'])
            v.setAttribute('maxSpeed', vtype['maxSpeed'])
            v.setAttribute('color', vtype['color'])
            v.setAttribute('imgFile', vtype['imgFile'])
            
            # Parameters to prevent intersection blocking and collisions
            v.setAttribute('jmDriveAfterRedTime', vtype['jmDriveAfterRedTime'])
            v.setAttribute('jmDriveRedSpeed', vtype['jmDriveRedSpeed'])
            v.setAttribute('jmIgnoreKeepClearTime', vtype['jmIgnoreKeepClearTime'])
            v.setAttribute('jmIgnoreFoeProb', vtype['jmIgnoreFoeProb'])
            v.setAttribute('jmIgnoreFoeSpeed', vtype['jmIgnoreFoeSpeed'])
            v.setAttribute('jmSigmaMinor', vtype['jmSigmaMinor'])
            v.setAttribute('jmTimegapMinor', vtype['jmTimegapMinor'])
            v.setAttribute('impatience', vtype['impatience'])
            
            # Car-following model parameters
            v.setAttribute('sigma', vtype['sigma'])
            v.setAttribute('tau', vtype['tau'])
            v.setAttribute('speedFactor', vtype['speedFactor'])
            v.setAttribute('speedDev', vtype['speedDev'])
            
            # Additional collision avoidance parameters
            if 'collisionMinGapFactor' in vtype:
                v.setAttribute('collisionMinGapFactor', vtype['collisionMinGapFactor'])
            if 'tauLast' in vtype:
                v.setAttribute('tauLast', vtype['tauLast'])
            if 'approachMinGap' in vtype:
                v.setAttribute('approachMinGap', vtype['approachMinGap'])
            if 'accelLat' in vtype:
                v.setAttribute('accelLat', vtype['accelLat'])
            
            routes_root.appendChild(v)
        
        # Add routes
        for route in self.routes:
            r = doc.createElement('route')
            r.setAttribute('id', route['id'])
            r.setAttribute('edges', route['edges'])
            routes_root.appendChild(r)
        
        # Stage 1: EXTRA DENSE on west (2x previous density), moderate on others
        # Time: 0 to stage_duration
        print("Generating Stage 1: 2x Dense traffic on west, moderate on others")
        self._add_flows(doc, routes_root, 0, stage_duration, 
                    {"west": 800, "east": 200, "north": 200, "south": 60})
        
        # Stage 2: Low density on all ways (unchanged)
        # Time: stage_duration to 2*stage_duration
        print("Generating Stage 2: Low density on all ways")
        self._add_flows(doc, routes_root, stage_duration, 2*stage_duration, 
                    {"west": 230, "east": 190, "north": 120, "south": 170})
        
        # Stage 3: Dense on east (2x previous density), north, south, low on west
        # Time: 2*stage_duration to 3*stage_duration
        print("Generating Stage 3: 2x Dense traffic on east, dense on north and south, low on west")
        self._add_flows(doc, routes_root, 2*stage_duration, min(3*stage_duration, sim_duration), 
                    {"west": 200, "east": 800, "north": 300, "south": 300})
        
        # Write to file
        with open(self.output_file, 'w') as f:
            f.write(doc.toprettyxml(indent="  "))
        
        print(f"Traffic flows generated successfully and saved to {self.output_file}")
        print(f"Generated {len(self.vehicle_variations)} vehicle variations across {len(set([v['base_type'] for v in self.vehicle_variations]))} vehicle types")
        print(f"Stage 1: WEST traffic at 2x density (600 veh/h)")
        print(f"Stage 3: EAST traffic at 2x density (600 veh/h)")
        
        # Generate custom SUMO configuration with enhanced collision avoidance
        self._generate_sumo_config()

    def _generate_sumo_config(self):
        """Generate a SUMO configuration file with enhanced collision avoidance settings"""
        config_file = os.path.join(os.path.dirname(self.output_file), "simulation.sumocfg")
        
        # Create XML document
        doc = minidom.Document()
        
        # Create root element
        config_root = doc.createElement('configuration')
        doc.appendChild(config_root)
        
        # Input section
        input_section = doc.createElement('input')
        config_root.appendChild(input_section)
        
        # Add network file
        net_file = doc.createElement('net-file')
        net_file.setAttribute('value', "crossroads.net.xml")
        input_section.appendChild(net_file)
        
        # Add route file
        route_file = doc.createElement('route-files')
        route_file.setAttribute('value', os.path.basename(self.output_file))
        input_section.appendChild(route_file)
        
        # Time section
        time_section = doc.createElement('time')
        config_root.appendChild(time_section)
        
        # Set begin time
        begin = doc.createElement('begin')
        begin.setAttribute('value', "0")
        time_section.appendChild(begin)
        
        # Set end time
        end = doc.createElement('end')
        end.setAttribute('value', "3600")
        time_section.appendChild(end)
        
        # Set step length
        step_length = doc.createElement('step-length')
        step_length.setAttribute('value', "0.1")  # Smaller step length for smoother movement
        time_section.appendChild(step_length)
        
        # Processing section - ENHANCED FOR COLLISION AVOIDANCE
        processing_section = doc.createElement('processing')
        config_root.appendChild(processing_section)
        
        # Collision settings - ENHANCED
        collision_check = doc.createElement('collision.check-junctions')
        collision_check.setAttribute('value', "true")
        processing_section.appendChild(collision_check)
        
        collision_action = doc.createElement('collision.action')
        collision_action.setAttribute('value', "none")  # Changed from "warn" to "none" (stop and wait)
        processing_section.appendChild(collision_action)
        
        # Add explicit collision.mingap-factor
        collision_mingap = doc.createElement('collision.mingap-factor')
        collision_mingap.setAttribute('value', "2.0")  # Double the minimum gap for collision detection
        processing_section.appendChild(collision_mingap)
        
        # Enable check for stopped vehicles
        check_stopped = doc.createElement('collision.stoptime')
        check_stopped.setAttribute('value', "300")  # Check for stopped vehicles after 300s
        processing_section.appendChild(check_stopped)
        
        # Enable checking for collisions with pedestrians
        check_pedestrians = doc.createElement('collision.check-accidents')
        check_pedestrians.setAttribute('value', "true")
        processing_section.appendChild(check_pedestrians)
        
        # Teleport settings - INCREASED TO AVOID GRIDLOCK
        teleport = doc.createElement('time-to-teleport')
        teleport.setAttribute('value', "120")  # Reduced from 300 to 120 - teleport stuck vehicles sooner
        processing_section.appendChild(teleport)
        
        # Report section
        report_section = doc.createElement('report')
        config_root.appendChild(report_section)
        
        # Add verbose logging
        verbose = doc.createElement('verbose')
        verbose.setAttribute('value', "true")
        report_section.appendChild(verbose)
        
        # Disable step-log for performance
        no_step_log = doc.createElement('no-step-log')
        no_step_log.setAttribute('value', "true")
        report_section.appendChild(no_step_log)
        
        # GUI section
        gui_section = doc.createElement('gui_only')
        config_root.appendChild(gui_section)
        
        # Add tracker interval
        tracker_interval = doc.createElement('tracker-interval')
        tracker_interval.setAttribute('value', "0.1")
        gui_section.appendChild(tracker_interval)
        
        # Write to file
        with open(config_file, 'w') as f:
            f.write(doc.toprettyxml(indent="  "))
        
        print(f"SUMO configuration generated with enhanced collision avoidance settings: {config_file}")
    
    def _add_flows(self, doc, routes_root, begin_time, end_time, flow_rates):
        """
        Add flows for a specific time period with given flow rates.
        Uses only one flow specification parameter per flow to avoid conflicts.
        
        Args:
            doc: XML document
            routes_root: Root element
            begin_time: Start time of flows
            end_time: End time of flows
            flow_rates: Dictionary of flow rates per direction in vehicles/hour
        """
        for direction, rate in flow_rates.items():
            # Get routes in this direction
            routes_in_direction = self.directions[direction]
            if not routes_in_direction:
                continue  # Skip if no routes for this direction
                
            num_routes = len(routes_in_direction)
            
            # Distribute vehicles across routes (with some randomness)
            route_distribution = np.random.dirichlet(np.ones(num_routes)) 
            route_rates = [int(rate * prob) for prob in route_distribution]
            
            for i, route_id in enumerate(routes_in_direction):
                route_rate = route_rates[i]
                
                # Only proceed if there's actual traffic on this route
                if route_rate <= 0:
                    continue
                
                # Find the route definition to get from_lane and to_lane
                route_def = None
                for route in self.routes:
                    if route["id"] == route_id:
                        route_def = route
                        break
                
                if not route_def:
                    continue
                
                # Group vehicle variations by base type
                vehicle_types_by_base = {}
                for vtype in self.vehicle_variations:
                    base_type = vtype["base_type"]
                    if base_type not in vehicle_types_by_base:
                        vehicle_types_by_base[base_type] = []
                    vehicle_types_by_base[base_type].append(vtype)
                
                # Find the base vehicle type probabilities
                base_type_probabilities = {}
                for vtype in self.vehicle_types:
                    base_type_probabilities[vtype["id"]] = vtype["probability"]
                
                # Distribute vehicle types based on probability
                for base_type, probability in base_type_probabilities.items():
                    vtype_rate = int(route_rate * probability)
                    
                    if vtype_rate > 0 and base_type in vehicle_types_by_base:
                        # Distribute the type rate among variations
                        variations = vehicle_types_by_base[base_type]
                        variations_count = len(variations)
                        
                        # Equal distribution among variations, ensuring minimum rate is 1
                        for vtype in variations:
                            # Ensure valid rate - must be at least 1 veh/hour
                            individual_rate = max(1, vtype_rate // variations_count)
                            
                            # Create a unique ID for this flow
                            dir_prefix = direction[0].upper()  # First letter of direction
                            unique_id = f"flow_{dir_prefix}_{begin_time}_{end_time}_{self.flow_id}"
                            
                            # Create flow element
                            flow = doc.createElement('flow')
                            flow.setAttribute('id', unique_id)
                            flow.setAttribute('type', vtype['id'])
                            flow.setAttribute('route', route_id)
                            flow.setAttribute('begin', str(begin_time))
                            flow.setAttribute('end', str(end_time))
                            
                            # Set departure parameters
                            flow.setAttribute('departLane', str(route_def["from_lane"]))
                            flow.setAttribute('departSpeed', "max")
                            
                            # Set safer departure position based on vehicle type
                            if base_type == "bus" or base_type == "truck":
                                flow.setAttribute('departPos', "base")
                            else:
                                flow.setAttribute('departPos', "free")
                                
                            # Set arrival parameters
                            flow.setAttribute('arrivalLane', "current")
                            flow.setAttribute('arrivalPos', "max")
                            
                            # CRITICAL: Use probability instead of period or vehsPerHour to avoid conflicts
                            # Convert from vehicles/hour to probability (per second)
                            # Formula: probability = vehicles_per_hour / 3600
                            # Apply additional safety factor for larger vehicles
                            if base_type == "bus" or base_type == "truck":
                                # Lower probability for large vehicles (more spacing)
                                safe_probability = (individual_rate / 3600.0) * 0.7
                            else:
                                safe_probability = individual_rate / 3600.0
                                
                            # Cap probability to avoid too many vehicles
                            safe_probability = min(0.05, safe_probability)
                            
                            # Set ONLY the probability parameter (no period or vehsPerHour)
                            flow.setAttribute('probability', str(safe_probability))
                            
                            # Add flow to routes
                            routes_root.appendChild(flow)
                            self.flow_id += 1
    
    def _generate_sumo_config(self):
        """Generate a SUMO configuration file with disabled lateral movement"""
        config_file = os.path.join(os.path.dirname(self.output_file), "simulation.sumocfg")
        
        # Create XML document
        doc = minidom.Document()
        
        # Create root element
        config_root = doc.createElement('configuration')
        doc.appendChild(config_root)
        
        # Input section
        input_section = doc.createElement('input')
        config_root.appendChild(input_section)
        
        # Add network file
        net_file = doc.createElement('net-file')
        net_file.setAttribute('value', "crossroads.net.xml")
        input_section.appendChild(net_file)
        
        # Add route file
        route_file = doc.createElement('route-files')
        route_file.setAttribute('value', os.path.basename(self.output_file))
        input_section.appendChild(route_file)
        
        # Time section
        time_section = doc.createElement('time')
        config_root.appendChild(time_section)
        
        # Set begin time
        begin = doc.createElement('begin')
        begin.setAttribute('value', "0")
        time_section.appendChild(begin)
        
        # Set end time
        end = doc.createElement('end')
        end.setAttribute('value', "3600")
        time_section.appendChild(end)
        
        # Set step length
        step_length = doc.createElement('step-length')
        step_length.setAttribute('value', "0.1")  # Smaller step length for smoother movement
        time_section.appendChild(step_length)
        
        # Processing section
        processing_section = doc.createElement('processing')
        config_root.appendChild(processing_section)
        
        # Enable collision detection
        collision_check = doc.createElement('collision.check-junctions')
        collision_check.setAttribute('value', "true")
        processing_section.appendChild(collision_check)
        
        collision_action = doc.createElement('collision.action')
        collision_action.setAttribute('value', "warn")
        processing_section.appendChild(collision_action)
        
        # IMPORTANT: Remove lanechange.duration parameter that's causing the error
        
        # Teleport settings
        teleport = doc.createElement('time-to-teleport')
        teleport.setAttribute('value', "300")  # Teleport stuck vehicles after 5 minutes
        processing_section.appendChild(teleport)
        
        # Report section
        report_section = doc.createElement('report')
        config_root.appendChild(report_section)
        
        # Add verbose logging
        verbose = doc.createElement('verbose')
        verbose.setAttribute('value', "true")
        report_section.appendChild(verbose)
        
        # Disable step-log for performance
        no_step_log = doc.createElement('no-step-log')
        no_step_log.setAttribute('value', "true")
        report_section.appendChild(no_step_log)
        
        # GUI section
        gui_section = doc.createElement('gui_only')
        config_root.appendChild(gui_section)
        
        # Add tracker interval
        tracker_interval = doc.createElement('tracker-interval')
        tracker_interval.setAttribute('value', "0.1")  # Smoother tracking for visualization
        gui_section.appendChild(tracker_interval)
        
        # Write to file
        with open(config_file, 'w') as f:
            f.write(doc.toprettyxml(indent="  "))
        
        print(f"SUMO configuration generated with disabled lateral movement: {config_file}")