#!/usr/bin/env python

import os
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

class NetworkGenerator:
    """
    Class for generating custom SUMO network configurations
    """
    
    def __init__(self, output_dir="network"):
        """
        Initialize the network generator.
        
        Args:
            output_dir: Directory to save generated network files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_four_way_crossroad(self, lanes_per_direction=1, lane_width=3.2, 
                                   speed_limit=16.67, edge_length=200, 
                                   junction_radius=10, output_prefix="crossroads"):
        """
        Generate a four-way crossroad network with specified parameters.
        
        Args:
            lanes_per_direction: Number of lanes per direction (1-4)
            lane_width: Width of each lane in meters
            speed_limit: Speed limit in m/s (default: 16.67 m/s = 60 km/h)
            edge_length: Length of each edge in meters
            junction_radius: Radius of the junction in meters
            output_prefix: Prefix for output files
            
        Returns:
            Path to the generated network file
        """
        if not 1 <= lanes_per_direction <= 4:
            raise ValueError("Number of lanes must be between 1 and 4")
        
        # Create node file
        node_file = os.path.join(self.output_dir, f"{output_prefix}.nod.xml")
        self._create_node_file(node_file, junction_radius)
        
        # Create edge file
        edge_file = os.path.join(self.output_dir, f"{output_prefix}.edg.xml")
        self._create_edge_file(edge_file, lanes_per_direction, speed_limit, edge_length)
        
        # Create connection file
        connection_file = os.path.join(self.output_dir, f"{output_prefix}.con.xml")
        self._create_connection_file(connection_file, lanes_per_direction)
        
        # Create network file using SUMO netconvert
        net_file = os.path.join(self.output_dir, f"{output_prefix}.net.xml")
        self._run_netconvert(node_file, edge_file, connection_file, net_file, lane_width)
        
        # Create SUMO configuration file
        config_file = os.path.join(self.output_dir, "simulation.sumocfg")
        self._create_sumo_config(config_file, net_file)
        
        return net_file
    
    def _create_node_file(self, node_file, junction_radius):
        """
        Create a node file for the network.
        
        Args:
            node_file: Output node file path
            junction_radius: Radius of the junction in meters
        """
        # Create root element
        root = ET.Element("nodes")
        
        # Create center node (junction)
        center = ET.SubElement(root, "node")
        center.set("id", "center")
        center.set("x", "100")
        center.set("y", "100")
        center.set("type", "traffic_light")
        center.set("radius", str(junction_radius))
        
        # Create peripheral nodes (endpoints)
        # North
        north = ET.SubElement(root, "node")
        north.set("id", "north")
        north.set("x", "100")
        north.set("y", "300")
        north.set("type", "priority")
        
        # East
        east = ET.SubElement(root, "node")
        east.set("id", "east")
        east.set("x", "300")
        east.set("y", "100")
        east.set("type", "priority")
        
        # South
        south = ET.SubElement(root, "node")
        south.set("id", "south")
        south.set("x", "100")
        south.set("y", "-100")
        south.set("type", "priority")
        
        # West
        west = ET.SubElement(root, "node")
        west.set("id", "west")
        west.set("x", "-100")
        west.set("y", "100")
        west.set("type", "priority")
        
        # Write to file
        self._write_xml(root, node_file)
    
    def _create_edge_file(self, edge_file, lanes_per_direction, speed_limit, edge_length):
        """
        Create an edge file for the network.
        
        Args:
            edge_file: Output edge file path
            lanes_per_direction: Number of lanes per direction
            speed_limit: Speed limit in m/s
            edge_length: Length of each edge in meters
        """
        # Create root element
        root = ET.Element("edges")
        
        # Create edges for each direction
        # From North to Center
        edge_n_to_c = ET.SubElement(root, "edge")
        edge_n_to_c.set("id", "north_to_center")
        edge_n_to_c.set("from", "north")
        edge_n_to_c.set("to", "center")
        edge_n_to_c.set("numLanes", str(lanes_per_direction))
        edge_n_to_c.set("speed", str(speed_limit))
        
        # From Center to North
        edge_c_to_n = ET.SubElement(root, "edge")
        edge_c_to_n.set("id", "center_to_north")
        edge_c_to_n.set("from", "center")
        edge_c_to_n.set("to", "north")
        edge_c_to_n.set("numLanes", str(lanes_per_direction))
        edge_c_to_n.set("speed", str(speed_limit))
        
        # From East to Center
        edge_e_to_c = ET.SubElement(root, "edge")
        edge_e_to_c.set("id", "east_to_center")
        edge_e_to_c.set("from", "east")
        edge_e_to_c.set("to", "center")
        edge_e_to_c.set("numLanes", str(lanes_per_direction))
        edge_e_to_c.set("speed", str(speed_limit))
        
        # From Center to East
        edge_c_to_e = ET.SubElement(root, "edge")
        edge_c_to_e.set("id", "center_to_east")
        edge_c_to_e.set("from", "center")
        edge_c_to_e.set("to", "east")
        edge_c_to_e.set("numLanes", str(lanes_per_direction))
        edge_c_to_e.set("speed", str(speed_limit))
        
        # From South to Center
        edge_s_to_c = ET.SubElement(root, "edge")
        edge_s_to_c.set("id", "south_to_center")
        edge_s_to_c.set("from", "south")
        edge_s_to_c.set("to", "center")
        edge_s_to_c.set("numLanes", str(lanes_per_direction))
        edge_s_to_c.set("speed", str(speed_limit))
        
        # From Center to South
        edge_c_to_s = ET.SubElement(root, "edge")
        edge_c_to_s.set("id", "center_to_south")
        edge_c_to_s.set("from", "center")
        edge_c_to_s.set("to", "south")
        edge_c_to_s.set("numLanes", str(lanes_per_direction))
        edge_c_to_s.set("speed", str(speed_limit))
        
        # From West to Center
        edge_w_to_c = ET.SubElement(root, "edge")
        edge_w_to_c.set("id", "west_to_center")
        edge_w_to_c.set("from", "west")
        edge_w_to_c.set("to", "center")
        edge_w_to_c.set("numLanes", str(lanes_per_direction))
        edge_w_to_c.set("speed", str(speed_limit))
        
        # From Center to West
        edge_c_to_w = ET.SubElement(root, "edge")
        edge_c_to_w.set("id", "center_to_west")
        edge_c_to_w.set("from", "center")
        edge_c_to_w.set("to", "west")
        edge_c_to_w.set("numLanes", str(lanes_per_direction))
        edge_c_to_w.set("speed", str(speed_limit))
        
        # Write to file
        self._write_xml(root, edge_file)
    
    def _create_connection_file(self, connection_file, lanes_per_direction):
        """
        Create a connection file with lane-specific turn restrictions to prevent gridlock.
        
        Args:
            connection_file: Output connection file path
            lanes_per_direction: Number of lanes per direction
        """
        root = ET.Element("connections")
        
        turn_mapping = {
            "north_to_center": {"straight": "center_to_south", "left": "center_to_east", "right": "center_to_west"},
            "south_to_center": {"straight": "center_to_north", "left": "center_to_west", "right": "center_to_east"},
            "east_to_center": {"straight": "center_to_west", "left": "center_to_north", "right": "center_to_south"},
            "west_to_center": {"straight": "center_to_east", "left": "center_to_south", "right": "center_to_north"}
        }
        
        N = lanes_per_direction
        
        for from_edge, turns in turn_mapping.items():
            for from_lane in range(N):
                if N == 1:
                    allowed_turns = ["left", "straight", "right"]
                else:
                    if from_lane == 0:
                        allowed_turns = ["left", "straight"]
                    elif from_lane == N - 1:
                        allowed_turns = ["right", "straight"]
                    else:
                        allowed_turns = ["straight"]
                
                for turn in allowed_turns:
                    to_edge = turns[turn]
                    if turn == "straight":
                        to_lane = from_lane
                    elif turn == "left":
                        to_lane = 0
                    elif turn == "right":
                        to_lane = N - 1
                    
                    conn = ET.SubElement(root, "connection")
                    conn.set("from", from_edge)
                    conn.set("to", to_edge)
                    conn.set("fromLane", str(from_lane))
                    conn.set("toLane", str(to_lane))
        
        self._write_xml(root, connection_file)
    
    def _run_netconvert(self, node_file, edge_file, connection_file, net_file, lane_width):
        """
        Run SUMO netconvert to generate the network file.
        
        Args:
            node_file: Input node file path
            edge_file: Input edge file path
            connection_file: Input connection file path
            net_file: Output network file path
            lane_width: Width of each lane in meters
        """
        try:
            # Run netconvert command with compatible options
            cmd = [
                "netconvert",
                "--node-files", node_file,
                "--edge-files", edge_file,
                "--connection-files", connection_file,
                "--output", net_file,
                "--default.lanewidth", str(lane_width),
                "--no-turnarounds", "true",
                "--no-internal-links", "false",
                "--tls.guess", "true",  # Auto-detect traffic lights
                "--tls.default-type", "static",  # Use static traffic light type
                "--lefthand", "false"
            ]
            
            subprocess.run(cmd, check=True)
            print(f"Network file generated: {net_file}")
            
            # After network creation, we'll create a separate additional file for traffic light timings
            self._create_traffic_light_config(net_file)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running netconvert: {e}")
            
            # Create a fallback network as a last resort
            self._create_fallback_network(net_file, lane_width)
        except FileNotFoundError:
            print("Error: netconvert command not found. Make sure SUMO is installed and in your PATH.")
            
            # Create a minimal valid net.xml file as a fallback
            self._create_fallback_network(net_file, lane_width)
    
    def _create_traffic_light_config(self, net_file):
        """
        Create a separate configuration for traffic light timings.
        
        Args:
            net_file: Path to the network file
        """
        # Get directory and name of the net file
        dirname = os.path.dirname(net_file)
        basename = os.path.basename(net_file).split('.')[0]
        
        # Create additional file for traffic light programs
        add_file = os.path.join(dirname, f"{basename}.tll.xml")
        
        # Create XML document
        root = ET.Element("additional")
        
        # Create traffic light program for the center junction
        tl_logic = ET.SubElement(root, "tlLogic")
        tl_logic.set("id", "center")
        tl_logic.set("type", "static")
        tl_logic.set("programID", "0")
        tl_logic.set("offset", "0")
        
        # Create phases
        # Phase 1: North-South green, East-West red
        phase1 = ET.SubElement(tl_logic, "phase")
        phase1.set("duration", "31")
        phase1.set("state", "GGgrrrGGgrrr")
        
        # Phase 2: North-South yellow, East-West red
        phase2 = ET.SubElement(tl_logic, "phase")
        phase2.set("duration", "4")
        phase2.set("state", "yyyrrryyyrrrr")
        
        # Phase 3: North-South red, East-West green
        phase3 = ET.SubElement(tl_logic, "phase")
        phase3.set("duration", "31")
        phase3.set("state", "rrrGGgrrrGGg")
        
        # Phase 4: North-South red, East-West yellow
        phase4 = ET.SubElement(tl_logic, "phase")
        phase4.set("duration", "4")
        phase4.set("state", "rrryyyrrryyy")
        
        # Write to file
        self._write_xml(root, add_file)
        
        # Update the SUMO configuration file to include this additional file
        config_file = os.path.join(dirname, "simulation.sumocfg")
        if os.path.exists(config_file):
            tree = ET.parse(config_file)
            root = tree.getroot()
            
            # Find or create the input section
            input_section = None
            for section in root.findall("input"):
                input_section = section
                break
            
            if input_section is None:
                input_section = ET.SubElement(root, "input")
            
            # Add or update additional-files element
            additional_files = None
            for add in input_section.findall("additional-files"):
                additional_files = add
                break
            
            if additional_files is None:
                additional_files = ET.SubElement(input_section, "additional-files")
                additional_files.set("value", os.path.basename(add_file))
            else:
                val = additional_files.get("value", "")
                if val:
                    additional_files.set("value", val + "," + os.path.basename(add_file))
                else:
                    additional_files.set("value", os.path.basename(add_file))
            
            # Write updated config back to file
            tree.write(config_file)
            print(f"Updated SUMO configuration with traffic light settings")
    
    def _create_fallback_network(self, net_file, lane_width):
        """
        Create a fallback network file if netconvert is not available.
        This is a very simplified version and not recommended for production use.
        
        Args:
            net_file: Output network file path
            lane_width: Width of each lane in meters
        """
        print("Creating fallback network file. This is a simplified version.")
        
        # Read the template file from the project
        template_file = os.path.join(self.output_dir, "crossroads.net.xml")
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                content = f.read()
            
            # Write the template content to the output file
            with open(net_file, 'w') as f:
                f.write(content)
            
            print(f"Fallback network file created: {net_file}")
        else:
            print("Error: Template network file not found.")
            raise FileNotFoundError("Template network file not found")
    
    def _create_sumo_config(self, config_file, net_file):
        """
        Create a SUMO configuration file.
        
        Args:
            config_file: Output configuration file path
            net_file: Path to the network file
        """
        # Create root element
        root = ET.Element("configuration")
        
        # Input section
        input_section = ET.SubElement(root, "input")
        net_file_elem = ET.SubElement(input_section, "net-file")
        net_file_elem.set("value", os.path.basename(net_file))
        route_files = ET.SubElement(input_section, "route-files")
        route_files.set("value", "routes.rou.xml")
        
        # Time section
        time_section = ET.SubElement(root, "time")
        begin = ET.SubElement(time_section, "begin")
        begin.set("value", "0")
        end = ET.SubElement(time_section, "end")
        end.set("value", "3600")
        step_length = ET.SubElement(time_section, "step-length")
        step_length.set("value", "1.0")
        
        # Report section
        report_section = ET.SubElement(root, "report")
        verbose = ET.SubElement(report_section, "verbose")
        verbose.set("value", "true")
        no_step_log = ET.SubElement(report_section, "no-step-log")
        no_step_log.set("value", "true")
        log = ET.SubElement(report_section, "log")
        log.set("value", "simulation.log")
        
        # Processing section
        processing_section = ET.SubElement(root, "processing")
        teleport = ET.SubElement(processing_section, "time-to-teleport")
        teleport.set("value", "-1")
        lanechange = ET.SubElement(processing_section, "lanechange.duration")
        lanechange.set("value", "2")
        
        # GUI section
        gui_section = ET.SubElement(root, "gui_only")
        start = ET.SubElement(gui_section, "start")
        start.set("value", "false")
        tracker = ET.SubElement(gui_section, "tracker-interval")
        tracker.set("value", "0.5")
        
        # Write to file
        self._write_xml(root, config_file)
    
    def _write_xml(self, root, file_path):
        """
        Write an XML element tree to file with pretty formatting.
        
        Args:
            root: Root XML element
            file_path: Output file path
        """
        # Convert to string with pretty formatting
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(pretty_string)


if __name__ == "__main__":
    # Test the network generator
    generator = NetworkGenerator()
    net_file = generator.generate_four_way_crossroad(lanes_per_direction=4)
    print(f"Generated network file: {net_file}")