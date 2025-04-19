#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from controllers.fixed_time import FixedTimeController
from controllers.density_based import DensityBasedController
from controllers.rl_controller import RLController

class SimulationRunner:
    """
    Class to run simulations with different traffic light controllers
    and compare their performance.
    """
    
    def __init__(self, sumo_binary="sumo", config_file="network/simulation.sumocfg"):
        """
        Initialize the simulation runner.
        
        Args:
            sumo_binary: SUMO binary to use ("sumo" or "sumo-gui")
            config_file: Path to SUMO configuration file
        """
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")
            
        import traci
        
        if not os.path.exists(config_file):
            print(f"Warning: Config file {config_file} not found!")
            print("Please ensure you've generated the network first.")
        
        self.sumo_cmd = [sumo_binary, "-c", config_file]
        self.results = {}
    
    def run_all_controllers(self, train_rl=True, sim_duration=3600, rl_episodes=5, ensemble_size=3):
        """
        Run simulations with all three traffic light controllers.
        
        Args:
            train_rl: Whether to train the RL controller
            sim_duration: Simulation duration in seconds
            rl_episodes: Number of episodes for RL training (default: 5)
            ensemble_size: Number of models in the RL ensemble (default: 3)
            
        Returns:
            Dictionary of performance metrics for all controllers
        """
        print("Running simulations with all controllers...")
        
        os.makedirs("models", exist_ok=True)
        
        # Add warning suppression to SUMO command
        sumo_cmd = self.sumo_cmd.copy()
        if "--no-warnings" not in sumo_cmd:
            sumo_cmd.append("--no-warnings")
        
        # 1. Fixed-time controller
        print("\n--- Running Fixed-Time Controller ---")
        fixed_controller = FixedTimeController()
        fixed_metrics = fixed_controller.run(sumo_cmd, sim_duration, epoch=1)
        self.results["Fixed-Time"] = fixed_metrics
        
        # 2. Density-based controller
        print("\n--- Running Density-Based Controller ---")
        density_controller = DensityBasedController()
        density_metrics = density_controller.run(sumo_cmd, sim_duration, epoch=1)
        self.results["Density-Based"] = density_metrics
        
        # 3. RL controller
        print(f"\n--- Running RL Controller (Ensemble Size: {ensemble_size}) ---")
        rl_controller = RLController()
        # Set ensemble size
        rl_controller.ensemble_size = ensemble_size
        
        # Determine the correct model file path
        if os.path.exists("models/rl_controller_final.pt"):
            model_path = "models/rl_controller_final.pt"
        else:
            model_path = "models/rl_controller_final.weights.h5"
        
        if train_rl:
            print(f"Training RL controller for {rl_episodes} episodes with ensemble size {ensemble_size}...")
            rl_controller.train(sumo_cmd, episodes=rl_episodes, sim_duration=sim_duration)
        
        print(f"Running RL controller with model: {model_path}")
        rl_metrics = rl_controller.run(sumo_cmd, model_path=model_path, sim_duration=sim_duration, epoch=1)
        self.results["RL-Based"] = rl_metrics
        
        # Verify results before returning
        self.verify_results()
        
        return self.results
    
    def run_single_controller(self, controller_type="fixed", train_rl=True, sim_duration=3600, rl_episodes=5, ensemble_size=3):
        """
        Run simulation with a single controller.
        
        Args:
            controller_type: Type of controller ("fixed", "density", or "rl")
            train_rl: Whether to train the RL controller (if controller_type is "rl")
            sim_duration: Simulation duration in seconds
            rl_episodes: Number of episodes for RL training (default: 5)
            ensemble_size: Number of models in the RL ensemble (default: 3)
            
        Returns:
            Dictionary of performance metrics for the controller
        """
        # Add warning suppression to SUMO command
        sumo_cmd = self.sumo_cmd.copy()
        if "--no-warnings" not in sumo_cmd:
            sumo_cmd.append("--no-warnings")
            
        if controller_type == "fixed":
            controller = FixedTimeController()
            # Pass epoch=1 to the run method
            metrics = controller.run(sumo_cmd, sim_duration, epoch=1)
            self.results["Fixed-Time"] = metrics
            self.verify_results()
            return metrics
            
        elif controller_type == "density":
            controller = DensityBasedController()
            # Pass epoch=1 to the run method
            metrics = controller.run(sumo_cmd, sim_duration, epoch=1)
            self.results["Density-Based"] = metrics
            self.verify_results()
            return metrics
            
        elif controller_type == "rl":
            controller = RLController()
            # Set ensemble size
            controller.ensemble_size = ensemble_size
            print(f"Using RL ensemble with {ensemble_size} models")
            
            # Determine the correct model file path
            if os.path.exists("models/rl_controller_final.pt"):
                model_path = "models/rl_controller_final.pt"
            else:
                model_path = "models/rl_controller_final.weights.h5"
            
            if train_rl:
                print(f"Training RL controller for {rl_episodes} episodes with ensemble size {ensemble_size}...")
                controller.train(sumo_cmd, episodes=rl_episodes, sim_duration=sim_duration)
            
            print(f"Running RL controller with model: {model_path}")
            # Pass epoch=1 to the run method
            metrics = controller.run(sumo_cmd, model_path=model_path, sim_duration=sim_duration, epoch=1)
            self.results["RL-Based"] = metrics
            self.verify_results()
            return metrics
            
        else:
            print(f"Unknown controller type: {controller_type}")
            return None
    
    def verify_results(self):
        """Verify that results contain valid metrics"""
        print("\nVerifying simulation results:")
        
        for controller, metrics in self.results.items():
            print(f"\n{controller} metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value}")
                    
                    # Check for potential issues
                    if np.isnan(value) or np.isinf(value):
                        print(f"    WARNING: {key} is {value} (invalid)")
                        # Fix the value
                        self.results[controller][key] = 0.0
                        print(f"    FIXED: Set {key} to 0.0")
                    elif key == "throughput" and value == 0:
                        print(f"    WARNING: {key} is zero, might indicate tracking issue")
                    elif key == "throughput" and value < 0:
                        print(f"    WARNING: {key} is negative, this is invalid")
                        # Fix negative values
                        self.results[controller][key] = abs(value)
                        print(f"    FIXED: Set {key} to {abs(value)}")
        
        return self.results
    
    def compare_results(self, save_plots=True):
        """
        Compare the performance of different controllers.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            print("No results to compare. Run simulations first.")
            return None
        
        comparison = {
            "Controller": [],
            "Avg Waiting Time (s)": [],
            "Avg Speed (m/s)": [],
            "Throughput (veh/h)": [],
            "WE Avg Queue": [],
            "NS Avg Queue": []
        }
        
        for controller_name, metrics in self.results.items():
            comparison["Controller"].append(controller_name)
            comparison["Avg Waiting Time (s)"].append(metrics["avg_waiting_time"])
            comparison["Avg Speed (m/s)"].append(metrics["avg_speed"])
            comparison["Throughput (veh/h)"].append(metrics["throughput"])
            comparison["WE Avg Queue"].append(metrics.get("we_avg_queue", 0))
            comparison["NS Avg Queue"].append(metrics.get("ns_avg_queue", 0))
        
        df = pd.DataFrame(comparison)
        
        print("\n--- Performance Comparison ---")
        print(df.to_string(index=False))
        
        if save_plots:
            os.makedirs("plots", exist_ok=True)
            
            # Plot waiting time comparison
            plt.figure(figsize=(10, 6))
            plt.bar(df["Controller"], df["Avg Waiting Time (s)"])
            plt.title("Average Waiting Time Comparison")
            plt.ylabel("Average Waiting Time (seconds)")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig("plots/waiting_time_comparison.png", dpi=300)
            plt.close()
            
            # Plot speed comparison
            plt.figure(figsize=(10, 6))
            plt.bar(df["Controller"], df["Avg Speed (m/s)"])
            plt.title("Average Speed Comparison")
            plt.ylabel("Average Speed (m/s)")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig("plots/speed_comparison.png", dpi=300)
            plt.close()
            
            # Plot throughput comparison with debug info
            plt.figure(figsize=(10, 6))
            throughputs = df["Throughput (veh/h)"].values
            print(f"DEBUG: Throughput values for plotting: {throughputs}")
            
            # Ensure all values are float
            throughputs = [float(val) for val in throughputs]
            
            # Add small value to zero throughputs to make bars visible
            for i, val in enumerate(throughputs):
                if val < 0.1:
                    throughputs[i] = 0.1
                    print(f"Setting zero throughput to 0.1 for {df['Controller'][i]} to make bar visible")
            
            bars = plt.bar(df["Controller"], throughputs)
            
            # Add data labels to bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.title("Throughput Comparison")
            plt.ylabel("Throughput (vehicles/hour)")
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/throughput_comparison.png", dpi=300)
            plt.close()
            
            # Plot queue length comparison
            plt.figure(figsize=(12, 6))
            x = np.arange(len(df["Controller"]))
            width = 0.35
            plt.bar(x - width/2, df["WE Avg Queue"], width, label='WE Direction')
            plt.bar(x + width/2, df["NS Avg Queue"], width, label='NS Direction')
            plt.xlabel('Controller')
            plt.ylabel('Average Queue Length (vehicles)')
            plt.title('Average Queue Length by Direction')
            plt.xticks(x, df["Controller"])
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/queue_length_comparison.png", dpi=300)
            plt.close()
            
            print("Plots saved to 'plots' directory.")
        
        return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run traffic light simulations")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument("--no-train", dest="train", action="store_false", 
                        help="Skip RL controller training")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Simulation duration in seconds (default: 3600)")
    parser.add_argument("--controller", type=str, choices=["all", "fixed", "density", "rl"], default="all",
                        help="Controller to run (default: all)")
    parser.add_argument("--config", type=str, default="network/simulation.sumocfg",
                        help="Path to SUMO configuration file")
    parser.add_argument("--output", type=str, default="results.pkl",
                        help="Output file for results")
    parser.add_argument("--rl-episodes", type=int, default=5,
                        help="Number of episodes for RL training (default: 5)")
    parser.add_argument("--ensemble-size", type=int, default=3,
                        help="Number of models in the RL ensemble (default: 3)")
    
    args = parser.parse_args()
    
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    
    print(f"Running simulation_runner.py with RL episodes: {args.rl_episodes}, Ensemble size: {args.ensemble_size}")
    
    runner = SimulationRunner(sumo_binary=sumo_binary, config_file=args.config)
    
    if args.controller == "all":
        results = runner.run_all_controllers(
            train_rl=args.train, 
            sim_duration=args.duration,
            rl_episodes=args.rl_episodes,
            ensemble_size=args.ensemble_size
        )
        comparison = runner.compare_results(save_plots=True)
    else:
        results = runner.run_single_controller(
            controller_type=args.controller, 
            train_rl=args.train, 
            sim_duration=args.duration,
            rl_episodes=args.rl_episodes,
            ensemble_size=args.ensemble_size
        )
        print(f"\n--- {args.controller.capitalize()} Controller Results ---")
        for key, value in results.items():
            if not isinstance(value, dict):
                print(f"{key}: {value}")
    
    from analysis import SimulationAnalyzer
    analyzer = SimulationAnalyzer()
    analyzer.save_results(runner.results, args.output)