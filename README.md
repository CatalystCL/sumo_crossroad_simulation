# SUMO Crossroad Simulation

A traffic light simulation project that models and compares three different traffic light control strategies at a crossroad intersection using SUMO (Simulation of Urban MObility) framework and AI techniques.

## Overview

This project implements a customizable traffic simulation for crossroad intersections with different traffic light control strategies. The simulation allows for varying road configurations, vehicle types, and traffic patterns to evaluate the effectiveness of different control methods.

## Key Features

- **Customizable crossroad networks** with 1-4 lanes per direction
- **Diverse vehicle types** (passenger cars, buses, trucks, motorcycles) with realistic variation
- **Three traffic light control strategies**:
  - Fixed-time controller (pre-defined timing)
  - Density-based controller (adapts based on traffic density)
  - RL-based controller (uses ensemble DQN to learn optimal control patterns)
- **Realistic vehicle behavior**:
  - Enhanced collision avoidance
  - Vehicle type-specific safety parameters
- **Realistic traffic patterns** with three stages:
  - Dense traffic on west direction (Stage 1)
  - Low density on all directions (Stage 2)
  - Dense traffic on east, north, and south directions (Stage 3)
- **Comprehensive analysis tools** for performance comparison

## Project Structure

```
.
├── controllers/             # Traffic light controller implementations
│   ├── __init__.py
│   ├── fixed_time.py        # Fixed-time controller
│   ├── density_based.py     # Density-based controller 
│   └── rl_controller.py     # RL-based controller with ensemble approach
├── network/                 # Network configuration files
├── models/                  # Trained RL models
├── plots/                   # Performance comparison plots
├── analysis/                # Detailed analysis results
├── main.py                  # Main entry point
├── network_generator.py     # Network generation module
├── traffic_generator.py     # Traffic flow generator with realistic behavior
├── simulation_runner.py     # Simulation runner
└── analysis.py              # Analysis module
```

## Requirements

- SUMO (Simulation of Urban MObility) version 1.8.0 or later
- Python 3.6 or later
- PyTorch (for RL controller)
- NumPy, Pandas, Matplotlib
- Other dependencies listed in requirements.txt

## Installation

1. Install SUMO following the [official instructions](https://sumo.dlr.de/docs/Installing/index.html)
2. Set up the SUMO environment variable:
   ```bash
   export SUMO_HOME=/path/to/sumo
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage Examples

### Create a network
```bash
python main.py create_network --lanes 4
```

### Generate traffic with variations
```bash
python main.py generate --lanes 4 --variations 15
```

### Run simulation with GUI
```bash
python main.py simulate --gui
```

### Run all controllers for comparison
```bash
python main.py simulate --controller all --duration 3600
```

### Train and run the RL controller
```bash
python main.py simulate --controller rl --rl-episodes 5 --ensemble-size 3
```

### All-in-one command (create, generate, simulate)
```bash
python main.py fourlane --gui --variations 15
```

### Analyze results
```bash
python main.py analyze
```

## Command Line Options

### Network Generation
- `--lanes`: Number of lanes per direction (1-4, default: 4)
- `--speed`: Speed limit in m/s (default: 13.89 m/s = 50 km/h)
- `--output-dir`: Output directory for network files

### Traffic Generation
- `--variations`: Number of variations per vehicle type (default: 10)
- `--duration`: Simulation duration in seconds (default: 3600)

### Simulation
- `--gui`: Run with SUMO GUI
- `--controller`: Controller to run (all, fixed, density, rl)
- `--no-train`: Skip RL controller training
- `--duration`: Simulation duration in seconds
- `--rl-episodes`: Number of episodes for RL training (default: 5)
- `--ensemble-size`: Number of models in the RL ensemble (default: 3)

### Analysis
- `--results`: Path to results file (.pkl)
- `--output-dir`: Directory to save analysis results

## Controller Details

### Fixed-Time Controller
- Uses pre-defined phase durations (31 seconds per phase)
- Simple but inefficient during varying traffic conditions

### Density-Based Controller
- Adapts phase durations based on traffic density
- Implements strategic sub-optimal behavior to model real-world controllers
- Uses lane density ratio to determine phase duration
- Phase durations range from 10 to 60 seconds

### RL-Based Controller (Ensemble DQN)
- Uses multiple DQN models in an ensemble for improved stability
- Models are dynamically weighted based on performance
- Comprehensive reward function based on:
  - Queue reduction (35%)
  - Queue length penalty (15%)
  - Throughput (25%)
  - Average speed (15%)
  - Queue balance between directions (10%)
- Supports different ensemble sizes for tuning exploration/exploitation

## Traffic Patterns

The simulation includes three distinct traffic stages:

1. **Stage 1 (0-1200s)**: Heavy traffic from west (600 veh/h), moderate on others
2. **Stage 2 (1200-2400s)**: Low density on all directions
3. **Stage 3 (2400-3600s)**: Heavy traffic from east (600 veh/h), north, and south, low from west

## Vehicle Types

- **Passenger Cars**: High maneuverability, medium gap requirements
- **Buses**: Limited maneuverability, large gap requirements
- **Trucks**: Very limited maneuverability, largest gap requirements
- **Motorcycles**: High maneuverability, smallest gap requirements

Each vehicle type includes multiple variations with different characteristics for realistic traffic simulation.

## Performance Metrics

- **Throughput**: Vehicles completed per hour
- **Average Waiting Time**: Time spent waiting at traffic lights
- **Average Speed**: Overall vehicle speed during simulation
- **Queue Lengths**: By direction and controller

## Visualization and Analysis

The project generates various plots to help analyze controller performance:
- Throughput comparison
- Waiting time analysis
- Speed distribution
- Comprehensive normalized comparison
- RL training progress and reward components

## Known Issues

- Ensure proper SUMO installation and environment variables
- PyTorch is required for the RL controller
- The RL controller requires more computational resources than other controllers

## Collision Avoidance Guidelines

For best results with collision avoidance:
1. Use a lower traffic volume (vehicles per hour)
2. Ensure adequate minimum gaps between vehicles
3. Set reasonable deceleration and emergency deceleration values
4. Use the `--variations` parameter to create diverse vehicle behavior

```

rm network/*

python main.py create_network --lanes 4

python main.py generate --lanes 4 --variations 15

python main.py simulate --gui
```
test

python -m venv venv

source venv/bin/activate

python main.py simulate --controller all --duration 3600

python main.py simulate --controller rl --rl-episodes 5 --ensemble-size 3