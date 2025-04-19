#!/usr/bin/env python

import os
import sys
import argparse
from traffic_generator import TrafficGenerator
from simulation_runner import SimulationRunner
from analysis import SimulationAnalyzer
from network_generator import NetworkGenerator

def debug_print_args(args):
    """Debug function to print all arguments"""
    print("\n=== DEBUG: Command Arguments ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("================================\n")

def main():
    """
    Main entry point for the SUMO crossroad simulation project.
    """
    print("Running SUMO Crossroad Simulation v1.3 - With Ensemble RL Support")
    
    parser = argparse.ArgumentParser(description="SUMO Crossroad Traffic Light Simulation")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Network generation command
    net_parser = subparsers.add_parser('create_network', help='Create a custom network')
    net_parser.add_argument('--lanes', type=int, default=4,
                           help='Number of lanes per direction (1-4, default: 4)')
    net_parser.add_argument('--speed', type=float, default=13.89,
                           help='Speed limit in m/s (default: 13.89 m/s = 50 km/h)')
    net_parser.add_argument('--output-dir', type=str, default='network',
                           help='Output directory (default: network)')
    
    # Traffic generation command
    gen_parser = subparsers.add_parser('generate', help='Generate traffic patterns')
    gen_parser.add_argument('--output', type=str, default='network/routes.rou.xml',
                           help='Output route file (default: network/routes.rou.xml)')
    gen_parser.add_argument('--lanes', type=int, default=1,
                           help='Number of lanes per direction (default: 1)')
    gen_parser.add_argument('--duration', type=int, default=3600,
                           help='Simulation duration in seconds (default: 3600)')
    gen_parser.add_argument('--variations', type=int, default=10,
                           help='Number of variations per vehicle type (default: 10)')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run traffic simulations')
    sim_parser.add_argument('--gui', action='store_true', help='Run with SUMO GUI')
    sim_parser.add_argument('--no-train', dest='train', action='store_false', 
                          help='Skip RL controller training')
    sim_parser.add_argument('--duration', type=int, default=3600,
                          help='Simulation duration in seconds (default: 3600)')
    sim_parser.add_argument('--save', type=str, default='results.pkl',
                          help='Save results to file (default: results.pkl)')
    sim_parser.add_argument('--controller', type=str, choices=['all', 'fixed', 'density', 'rl'],
                          default='all', help='Controller to run (default: all)')
    sim_parser.add_argument('--rl-episodes', type=int, default=5,
                          help='Number of episodes for RL training (default: 5)')
    sim_parser.add_argument('--ensemble-size', type=int, default=3,
                          help='Number of models in the RL ensemble (default: 3)')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Analyze simulation results')
    analysis_parser.add_argument('--results', type=str, default='results.pkl',
                               help='Path to results file (.json or .pkl)')
    analysis_parser.add_argument('--output-dir', type=str, default='analysis',
                               help='Directory to save analysis results (default: analysis)')
    
    # All-in-one command for 4-lane crossroad simulation
    fourlane_parser = subparsers.add_parser('fourlane', 
                                          help='Create 4-lane crossroad, generate traffic, and run simulation')
    fourlane_parser.add_argument('--gui', action='store_true', help='Run with SUMO GUI')
    fourlane_parser.add_argument('--duration', type=int, default=3600,
                               help='Simulation duration in seconds (default: 3600)')
    fourlane_parser.add_argument('--variations', type=int, default=10,
                               help='Number of variations per vehicle type (default: 10)')
    fourlane_parser.add_argument('--rl-episodes', type=int, default=5,
                               help='Number of episodes for RL training (default: 5)')
    fourlane_parser.add_argument('--ensemble-size', type=int, default=3,
                               help='Number of models in the RL ensemble (default: 3)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'create_network':
        # Create a custom network
        print(f"Creating a {args.lanes}-lane crossroad network...")
        generator = NetworkGenerator(output_dir=args.output_dir)
        net_file = generator.generate_four_way_crossroad(
            lanes_per_direction=args.lanes,
            speed_limit=args.speed
        )
        print(f"Network created successfully: {net_file}")
    
    elif args.command == 'generate':
        # Generate traffic patterns
        print(f"Generating traffic patterns for {args.duration} seconds with {args.lanes} lanes and {args.variations} vehicle variations...")
        generator = TrafficGenerator(output_file=args.output, lanes_per_direction=args.lanes)
        # Apply the specified number of variations
        generator.vehicle_variations = generator._generate_vehicle_variations(variations_per_type=args.variations)
        generator.generate_traffic_flow(sim_duration=args.duration)
        print(f"Traffic patterns generated successfully and saved to {args.output}")
    
    elif args.command == 'simulate':
        # Print debug info
        debug_print_args(args)
        
        # Check if models directory exists
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
            print("Creating models directory as it doesn't exist")
            
        # Check if we need to force training if no models exist
        force_train = args.train
        if not force_train and not os.path.exists('models/rl_controller_final.weights.h5') and not os.path.exists('models/rl_controller_final.pt'):
            print("No pretrained model found. Will train a new model.")
            force_train = True
        
        # Run simulations
        print(f"Running traffic simulations with RL episodes: {args.rl_episodes}, Ensemble size: {args.ensemble_size}")
        sumo_binary = "sumo-gui" if args.gui else "sumo"
        runner = SimulationRunner(sumo_binary=sumo_binary)
        
        if args.controller == 'all':
            print(f"Running all controllers with {args.rl_episodes} RL episodes and ensemble size {args.ensemble_size}")
            results = runner.run_all_controllers(
                train_rl=force_train, 
                sim_duration=args.duration,
                rl_episodes=args.rl_episodes,
                ensemble_size=args.ensemble_size
            )
            comparison = runner.compare_results(save_plots=True)
        else:
            print(f"Running {args.controller} controller" + 
                 (f" with {args.rl_episodes} episodes and ensemble size {args.ensemble_size}" if args.controller == 'rl' else ""))
            results = runner.run_single_controller(
                controller_type=args.controller, 
                train_rl=force_train, 
                sim_duration=args.duration,
                rl_episodes=args.rl_episodes,
                ensemble_size=args.ensemble_size
            )
            print(f"\n--- {args.controller.capitalize()} Controller Results ---")
            for key, value in results.items():
                if not isinstance(value, dict):  # Skip nested dictionaries
                    print(f"{key}: {value}")
        
        # Save results
        analyzer = SimulationAnalyzer()
        analyzer.save_results(runner.results, args.save)
        print(f"Simulation results saved to {args.save}")
    
    elif args.command == 'analyze':
        # Analyze results
        print(f"Analyzing simulation results from {args.results}...")
        analyzer = SimulationAnalyzer(args.results)
        analyzer.analyze_traffic_flow(save_dir=args.output_dir)
        print(f"Analysis results saved to {args.output_dir} directory")
    
    elif args.command == 'fourlane':
        # Print debug info
        debug_print_args(args)
        
        # Create 4-lane crossroad
        print("Creating 4-lane crossroad network...")
        network_generator = NetworkGenerator()
        net_file = network_generator.generate_four_way_crossroad(lanes_per_direction=4)
        
        # Generate traffic for 4-lane crossroad
        print(f"Generating traffic patterns for {args.duration} seconds with 4 lanes and {args.variations} vehicle variations...")
        traffic_generator = TrafficGenerator(lanes_per_direction=4)
        # Apply the specified number of variations
        traffic_generator.vehicle_variations = traffic_generator._generate_vehicle_variations(variations_per_type=args.variations)
        traffic_generator.generate_traffic_flow(sim_duration=args.duration)
        
        # Check if models directory exists
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        
        # Run simulations
        print(f"Running traffic simulations on 4-lane crossroad with {args.rl_episodes} RL episodes and ensemble size {args.ensemble_size}...")
        sumo_binary = "sumo-gui" if args.gui else "sumo"
        runner = SimulationRunner(sumo_binary=sumo_binary)
        results = runner.run_all_controllers(
            train_rl=True, 
            sim_duration=args.duration,
            rl_episodes=args.rl_episodes,
            ensemble_size=args.ensemble_size
        )
        comparison = runner.compare_results(save_plots=True)
        
        # Save and analyze results
        analyzer = SimulationAnalyzer()
        analyzer.save_results(results, "results_4lane.pkl")
        analyzer.analyze_traffic_flow(save_dir="analysis_4lane")
        print("4-lane crossroad simulation completed!")
        print("Results saved to results_4lane.pkl")
        print("Analysis saved to analysis_4lane directory")
    
    else:
        # No command specified
        parser.print_help()

if __name__ == "__main__":
    main()