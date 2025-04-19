#!/usr/bin/env python

import os
import sys
import traci
import numpy as np
import pandas as pd
import random
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import time

# Silence PyTorch warnings if needed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DQNModel(nn.Module):
    """PyTorch deep Q-network model"""
    
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class EnsembleDQNAgent:
    """
    Ensemble of multiple DQN agents for improved performance 
    (inspired by boosting techniques)
    """
    
    def __init__(self, state_size, action_size, num_models=3, batch_size=32, memory_size=2000, gamma=0.95, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.num_models = num_models
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Use GPU if available
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        print(f"Creating ensemble of {num_models} DQN models")
        
        # Create multiple models with different initializations
        self.models = []
        for i in range(num_models):
            model = DQNModel(state_size, action_size).to(self.device)
            target_model = DQNModel(state_size, action_size).to(self.device)
            target_model.load_state_dict(model.state_dict())
            
            self.models.append({
                'model': model,
                'target': target_model,
                'optimizer': optim.Adam(model.parameters(), lr=0.001),
                'weight': 1.0 / num_models  # Equal weighting initially
            })
        
        self.criterion = nn.MSELoss()
        self.training_steps = 0
        
        # Track model performance
        self.model_performances = [[] for _ in range(num_models)]
    
    def update_target_model(self):
        """Update target networks for all models"""
        for model in self.models:
            model['target'].load_state_dict(model['model'].state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Use ensemble to determine action:
        - During training: randomly select one model using epsilon-greedy
        - During evaluation: weighted vote among all models
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            if training:
                # Randomly select one model during training for diversity
                model_idx = random.randrange(self.num_models)
                q_values = self.models[model_idx]['model'](state_tensor)
                return torch.argmax(q_values).item()
            else:
                # Use weighted voting during evaluation
                weighted_q_values = torch.zeros(self.action_size).to(self.device)
                
                for model in self.models:
                    q_values = model['model'](state_tensor)
                    weighted_q_values += model['weight'] * q_values
                
                return torch.argmax(weighted_q_values).item()
    
    def replay(self):
        """Train each model in the ensemble with slightly different batches"""
        if len(self.memory) < self.batch_size:
            return 0
        
        total_loss = 0
        losses = []
        
        # Create a reusable target Q-value for all models
        # This encourages diversity while maintaining a common goal
        common_indices = np.random.choice(len(self.memory), self.batch_size, replace=True)
        common_batch = [self.memory[i] for i in common_indices]
        common_states = torch.FloatTensor([item[0] for item in common_batch]).to(self.device)
        common_actions = torch.LongTensor([item[1] for item in common_batch]).to(self.device)
        common_rewards = torch.FloatTensor([item[2] for item in common_batch]).to(self.device)
        common_next_states = torch.FloatTensor([item[3] for item in common_batch]).to(self.device)
        common_dones = torch.FloatTensor([item[4] for item in common_batch]).to(self.device)
        
        # Calculate ensemble target Q-values once
        with torch.no_grad():
            ensemble_next_q = torch.zeros((self.batch_size, self.action_size)).to(self.device)
            for model in self.models:
                ensemble_next_q += model['weight'] * model['target'](common_next_states)
            next_action = ensemble_next_q.max(1)[1].unsqueeze(1)
            target_q_values = common_rewards + (1 - common_dones) * self.gamma * ensemble_next_q.gather(1, next_action).squeeze()
        
        # Train each model with a different subset of data
        for model_idx, model_dict in enumerate(self.models):
            model = model_dict['model']
            optimizer = model_dict['optimizer']
            
            # Bootstrap sampling - sample with replacement
            # This introduces diversity in training, similar to boosting
            indices = np.random.choice(len(self.memory), self.batch_size, replace=True)
            minibatch = [self.memory[i] for i in indices]
            
            states = torch.FloatTensor([item[0] for item in minibatch]).to(self.device)
            actions = torch.LongTensor([item[1] for item in minibatch]).to(self.device)
            
            # Current Q values from this model
            current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Mix model-specific targets with ensemble targets for better diversity
            # This is a boosting-like technique where models specialize while sharing knowledge
            model_target_q = model_dict['target'](common_next_states).max(1)[0]
            model_target_values = common_rewards + (1 - common_dones) * self.gamma * model_target_q
            
            # Use a weighted combination of model-specific targets and ensemble targets
            alpha = 0.7  # Weight for ensemble targets vs. model-specific targets
            combined_targets = alpha * target_q_values + (1 - alpha) * model_target_values
            
            # Compute loss and update model
            loss = self.criterion(current_q, combined_targets)
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track model performance (lower loss = better performance)
            loss_value = loss.item()
            losses.append(loss_value)
            self.model_performances[model_idx].append(loss_value)
            
            total_loss += loss_value
        
        # Update model weights based on recent performance (exponential moving average of losses)
        window_size = 10
        if self.training_steps > window_size:
            recent_performances = []
            for model_idx in range(self.num_models):
                recent_loss = np.mean(self.model_performances[model_idx][-window_size:])
                # Convert loss to weight (lower loss = higher weight)
                inv_loss = 1.0 / (recent_loss + 1e-5)
                recent_performances.append(inv_loss)
            
            # Normalize weights
            total_perf = sum(recent_performances)
            for model_idx, perf in enumerate(recent_performances):
                # Smooth weight updates (70% new, 30% old)
                new_weight = perf / total_perf
                old_weight = self.models[model_idx]['weight']
                self.models[model_idx]['weight'] = 0.7 * new_weight + 0.3 * old_weight
        
        # Renormalize weights
        total_weight = sum(model['weight'] for model in self.models)
        for model in self.models:
            model['weight'] /= total_weight
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Log model weights every 100 steps
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            weights = [f"{model['weight']:.3f}" for model in self.models]
            print(f"Model weights after {self.training_steps} steps: {weights}")
        
        # Return average loss across all models
        return total_loss / self.num_models
    
    def replay_with_loss(self):
        """Alias for the replay method that returns loss"""
        return self.replay()
    
    def load(self, name):
        """
        Load model weights from file with robust error handling.
        Attempts to load weights for all models in the ensemble.
        """
        try:
            saved_state = torch.load(name)
            
            # Check if saved state is for an ensemble
            if isinstance(saved_state, list) and len(saved_state) == self.num_models:
                # Load each model in the ensemble
                for i, model_state in enumerate(saved_state):
                    self.models[i]['model'].load_state_dict(model_state['model'])
                    self.models[i]['target'].load_state_dict(model_state['model'])  # Initialize target with model
                    self.models[i]['weight'] = model_state['weight']
                print(f"Successfully loaded ensemble of {self.num_models} models from {name}")
            else:
                # Try to adapt a single model to the first model in the ensemble
                print(f"Saved state doesn't match ensemble. Adapting to first model only.")
                
                # Check if we need to reshape for different architecture
                if 'network.0.weight' in saved_state:
                    saved_input_size = saved_state['network.0.weight'].shape[1]
                    current_input_size = self.state_size
                    
                    if saved_input_size != current_input_size:
                        print(f"Input size mismatch: saved={saved_input_size}, current={current_input_size}")
                        print("Creating a new ensemble with the current architecture")
                        return
                    
                    # Load only the first model, initialize others randomly
                    self.models[0]['model'].load_state_dict(saved_state)
                    self.models[0]['target'].load_state_dict(saved_state)
                    print(f"Loaded first model from {name}, others initialized randomly")
                else:
                    print(f"Incompatible saved state format in {name}")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Using newly initialized ensemble")
    
    def save(self, name):
        """Save all models in the ensemble to a file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(name), exist_ok=True)
            
            # Save states of all models in the ensemble
            ensemble_state = []
            for model_dict in self.models:
                ensemble_state.append({
                    'model': model_dict['model'].state_dict(),
                    'weight': model_dict['weight']
                })
            
            torch.save(ensemble_state, name)
            print(f"Saved ensemble of {self.num_models} models to {name}")
        except Exception as e:
            print(f"Error saving models: {e}")
            # Try saving to a different location
            try:
                torch.save(ensemble_state, 'models/backup_ensemble.pt')
                print("Saved backup to models/backup_ensemble.pt")
            except:
                print("Failed to save backup model")

class RLController:
    """
    Reinforcement Learning-based traffic light controller with Ensemble DQN
    """
    
    def __init__(self, intersection_id="center", min_green_time=5, max_green_time=30, yellow_time=4):
        self.intersection_id = intersection_id
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.lane_groups = {"WE": [], "NS": []}
        self.state_size = None
        self.action_size = 5
        self.action_values = [self.min_green_time + 5*i for i in range(self.action_size)]
        self.agent = None
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self._total_completed = 0  # New counter for reliable tracking
        self.prev_arrived = 0
        self.prev_completed_vehicles = 0  # Track previous completion count for reward
        self.vehicle_speeds = []
        self.queue_lengths = defaultdict(list)
        self.rewards = []
        self.avg_rewards = []
        self.reward_components = []  # Track reward components for analysis
        self.current_episode = 1  # Default episode number
        
        # Ensemble configuration 
        self.ensemble_size = 3  # Number of models in the ensemble

    def _detect_lane_groups(self):
        """
        Detect lane groups with better logging and error handling
        """
        all_lanes = traci.lane.getIDList()
        incoming_lanes = [lane for lane in all_lanes if "to_center" in lane and lane.split("_")[-1].isdigit()]
        
        print("\nDetecting lane groups:")
        print(f"Total lanes found: {len(all_lanes)}")
        print(f"Incoming lanes found: {len(incoming_lanes)}")
        
        if not incoming_lanes:
            print("WARNING: No incoming lanes detected. Check network configuration.")
        
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
        
        # Debug output
        print("\nWE lanes:", self.lane_groups["WE"])
        print("NS lanes:", self.lane_groups["NS"])
        
        # Calculate state size based on detected lanes
        self.state_size = len(self.lane_groups["WE"]) + len(self.lane_groups["NS"]) + 1
        print(f"State size (input features): {self.state_size}")
        
        # Create ensemble agent with the right state size
        self.agent = EnsembleDQNAgent(
            self.state_size, 
            self.action_size,
            num_models=self.ensemble_size
        )

    def get_state(self, current_phase):
        """Get current state representation for RL agent"""
        state = []
        for lane_id in self.lane_groups["WE"] + self.lane_groups["NS"]:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            state.append(queue)
        state.append(current_phase)
        return np.array(state, dtype=np.float32)

    def get_reward(self, prev_state, current_state):
        """
        Calculate reward based on multiple factors with a balanced approach.
        Uses several components weighted appropriately to provide stable learning signals.
        """
        # Extract queue information from states
        prev_queues = prev_state[:-1]
        current_queues = current_state[:-1]
        
        # Get the phase from the state
        current_phase = int(current_state[-1])
        
        # 1. Queue reduction reward - positive when queues decrease
        queue_change = np.sum(prev_queues) - np.sum(current_queues)
        
        # Scale queue change reward based on magnitude, but cap it
        queue_change_reward = min(5.0, max(-5.0, queue_change * 1.5))
        
        # 2. Queue length penalty - smaller penalty to avoid extreme negative values
        total_queue = np.sum(current_queues)
        queue_penalty = -0.2 * min(total_queue, 20)  # Cap at equivalent of 20 vehicles
        
        # 3. Vehicle processing reward - track throughput
        vehicles_processed = self._total_completed - self.prev_completed_vehicles
        self.prev_completed_vehicles = self._total_completed
        
        throughput_reward = min(5.0, vehicles_processed * 1.0)
        
        # 4. Speed bonus - reward higher average speeds
        recent_speeds = self.vehicle_speeds[-30:] if len(self.vehicle_speeds) >= 30 else self.vehicle_speeds
        avg_speed = np.mean(recent_speeds) if recent_speeds else 0
        
        # Normalize speed to a 0-1 range (assuming 13.89 m/s = 50 km/h is max desired speed)
        speed_factor = min(avg_speed / 13.89, 1.0)
        speed_bonus = speed_factor * 2.0
        
        # 5. Queue balance component - reward balancing queues across directions
        we_queues = np.sum(current_queues[:len(self.lane_groups["WE"])])
        ns_queues = np.sum(current_queues[len(self.lane_groups["WE"]):])
        total_direction_queues = we_queues + ns_queues
        
        # Calculate imbalance (0 = perfectly balanced, 1 = completely imbalanced)
        if total_direction_queues > 0:
            imbalance = abs(we_queues - ns_queues) / total_direction_queues
            balance_reward = -2.0 * imbalance  # Penalty for imbalance
        else:
            balance_reward = 0  # No imbalance if no queues
        
        # Calculate final reward as weighted sum
        reward = (
            0.35 * queue_change_reward +  # 35% weight to queue reduction
            0.15 * queue_penalty +        # 15% weight to queue penalty (negative)
            0.25 * throughput_reward +    # 25% weight to throughput
            0.15 * speed_bonus +          # 15% weight to speed
            0.10 * balance_reward         # 10% weight to balance between directions
        )
        
        # Add a small positive bias to encourage exploration in early training
        reward += 0.5
        
        # Store reward components for analysis
        self.reward_components.append({
            'queue_change': queue_change_reward,
            'queue_penalty': queue_penalty,
            'throughput': throughput_reward,
            'speed_bonus': speed_bonus,
            'balance': balance_reward,
            'total': reward
        })
        
        # Every 1000 steps, save detailed reward analysis
        if len(self.reward_components) % 1000 == 0 and len(self.reward_components) > 0:
            self._analyze_rewards()
        
        return reward

    def _analyze_rewards(self):
        """Analyze and visualize reward components for debugging"""
        if not self.reward_components:
            return
            
        os.makedirs("plots/rewards", exist_ok=True)
        
        try:
            # Extract components
            components = pd.DataFrame(self.reward_components[-1000:])  # Last 1000 steps
            
            # Calculate statistics
            stats = {
                'mean': components.mean(),
                'min': components.min(),
                'max': components.max(),
                'std': components.std()
            }
            
            # Save statistics to file
            with open(f"plots/rewards/reward_stats_{len(self.reward_components)}.txt", 'w') as f:
                f.write("Reward Component Statistics:\n")
                f.write("--------------------------\n")
                for stat_name, values in stats.items():
                    f.write(f"\n{stat_name.capitalize()}:\n")
                    for component, value in values.items():
                        f.write(f"  {component}: {value:.4f}\n")
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Plot total reward
            plt.subplot(3, 1, 1)
            plt.plot(components.index, components['total'], 'k-')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Total Reward (mean: {components["total"].mean():.2f})')
            plt.ylabel('Reward')
            plt.grid(True)
            
            # Plot positive components
            plt.subplot(3, 1, 2)
            plt.plot(components.index, components['queue_change'].clip(lower=0), 'g-', label='Queue Reduction')
            plt.plot(components.index, components['throughput'], 'b-', label='Throughput')
            plt.plot(components.index, components['speed_bonus'], 'm-', label='Speed')
            plt.title('Positive Components')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            
            # Plot negative components
            plt.subplot(3, 1, 3)
            plt.plot(components.index, components['queue_change'].clip(upper=0), 'r-', label='Queue Increase')
            plt.plot(components.index, components['queue_penalty'], 'c-', label='Queue Penalty')
            plt.plot(components.index, components['balance'], 'y-', label='Balance Penalty')
            plt.title('Negative Components')
            plt.xlabel('Step')
            plt.ylabel('Penalty')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"plots/rewards/reward_analysis_{len(self.reward_components)}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error analyzing rewards: {e}")

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

    def train(self, sumo_cmd, episodes=500, sim_duration=3600):
        """
        Train the RL ensemble agent with epoch labeling and warning suppression.
        
        Args:
            sumo_cmd: SUMO command to run simulation
            episodes: Number of training episodes
            sim_duration: Duration of each simulation episode in seconds
        
        Returns:
            Trained ensemble agent
        """
        # Suppress SUMO warnings
        sumo_cmd = list(sumo_cmd)  # Make a copy to avoid modifying the original
        if "--no-warnings" not in sumo_cmd:
            sumo_cmd.append("--no-warnings")
        
        print(f"Training Ensemble RL agent ({self.ensemble_size} models) for {episodes} episodes (each {sim_duration}s)...")
        
        # Create directories for plots
        os.makedirs("plots", exist_ok=True)
        os.makedirs("plots/training", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Metrics to track
        all_rewards = []
        avg_episode_rewards = []
        all_waiting_times = []
        all_speeds = []
        all_throughputs = []
        all_epsilon = []
        all_losses = []
        
        # Ensure we only loop for exactly the specified number of episodes
        for episode in range(int(episodes)):
            print(f"\n====== Starting Episode {episode+1}/{episodes} ======")
            episode_rewards = []
            episode_losses = []
            self.total_waiting_time = 0
            self.total_vehicles = 0
            self.completed_vehicles = 0
            self._total_completed = 0  # Reset total completed counter
            self.prev_arrived = 0
            self.prev_completed_vehicles = 0
            self.vehicle_speeds = []
            self.queue_lengths = defaultdict(list)
            self.reward_components = []
            
            # Set episode number for logging
            self.current_episode = episode + 1
            
            traci.start(sumo_cmd)
            if episode == 0:
                self._detect_lane_groups()
            
            current_phase = 0
            phase_timer = 0
            is_yellow = False
            done = False
            state = self.get_state(current_phase)
            step = 0
            
            while step < sim_duration and not done:
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
                        
                        next_state = self.get_state(current_phase)
                        reward = self.get_reward(state, next_state)
                        episode_rewards.append(reward)
                        self.agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        action = self.agent.act(state)
                        green_duration = self.action_values[action]
                    else:
                        phase_timer += 1
                else:
                    if phase_timer == 0:
                        action = self.agent.act(state)
                        green_duration = self.action_values[action]
                    if phase_timer >= green_duration:
                        is_yellow = True
                        phase_timer = 0
                        if current_phase == 0:
                            traci.trafficlight.setPhase(self.intersection_id, 1)
                        else:
                            traci.trafficlight.setPhase(self.intersection_id, 3)
                    else:
                        phase_timer += 1
                
                step += 1
                if step >= sim_duration:
                    done = True
            
            # Train the agent
            loss = self.agent.replay()
            if loss > 0:
                episode_losses.append(loss)
            
            if episode % 5 == 0:
                self.agent.update_target_model()
            
            # Calculate episode metrics
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            avg_waiting_time = self.total_waiting_time / max(1, self.total_vehicles)
            avg_speed = np.mean(self.vehicle_speeds) if self.vehicle_speeds else 0
            throughput = (self._total_completed / sim_duration) * 3600
            
            # Store metrics
            avg_episode_rewards.append(avg_reward)
            all_rewards.extend(episode_rewards)
            all_waiting_times.append(avg_waiting_time)
            all_speeds.append(avg_speed)
            all_throughputs.append(throughput)
            all_epsilon.append(self.agent.epsilon)
            all_losses.append(avg_loss)
            
            # Print progress
            print(f"Episode {episode+1}/{episodes} completed, " + 
                 f"Avg Reward: {avg_reward:.2f}, " +
                 f"Loss: {avg_loss:.4f}, " +
                 f"Waiting Time: {avg_waiting_time:.2f}s, " +
                 f"Throughput: {throughput:.1f} veh/h, " +
                 f"Epsilon: {self.agent.epsilon:.3f}")
            
            traci.close()
            
            # Save model periodically
            if episode % max(1, episodes // 5) == 0 or episode == episodes-1:
                model_path = f"models/rl_controller_ep{episode}.pt"
                self.agent.save(model_path)
                
                # Generate training progress plots
                self._plot_training_progress(
                    avg_episode_rewards, all_waiting_times, 
                    all_speeds, all_throughputs, all_epsilon, all_losses, episode+1
                )
        
        print(f"Training completed after {episodes} episodes.")
        
        # Save final model
        self.agent.save("models/rl_controller_final.pt")
        
        # Create training metrics visualizations
        self._plot_training_summary(
            avg_episode_rewards, all_waiting_times, 
            all_speeds, all_throughputs, all_epsilon, all_losses
        )
        
        # Save training metrics for future reference
        training_data = {
            'episode_rewards': avg_episode_rewards,
            'waiting_times': all_waiting_times,
            'speeds': all_speeds,
            'throughputs': all_throughputs,
            'epsilon': all_epsilon,
            'losses': all_losses
        }
        with open('plots/training/training_data.pkl', 'wb') as f:
            pickle.dump(training_data, f)
            
        return self.agent

    def run(self, sumo_cmd, model_path=None, sim_duration=3600, epoch=1):
        """
        Run the trained RL controller with better error handling
        
        Args:
            sumo_cmd: SUMO command to run
            model_path: Path to model weights file
            sim_duration: Simulation duration in seconds
            epoch: Epoch number for logging
            
        Returns:
            Dictionary with performance metrics
        """
        # Store the epoch number for logging
        self.current_episode = epoch
        
        # Reset metrics for this run
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self._total_completed = 0
        self.prev_arrived = 0
        self.prev_completed_vehicles = 0
        self.vehicle_speeds = []
        self.queue_lengths = defaultdict(list)
        
        # Add warning suppression
        sumo_cmd = list(sumo_cmd)  # Make a copy
        if "--no-warnings" not in sumo_cmd:
            sumo_cmd.append("--no-warnings")
        
        # Check if model exists
        model_exists = model_path and os.path.exists(model_path)
        if model_path and not model_exists:
            print(f"Warning: Model file {model_path} not found!")
            print("Will run with a new ensemble (random policy)")
            model_path = None
        
        # Start simulation and detect lanes
        traci.start(sumo_cmd)
        self._detect_lane_groups()
        
        # Try to load model if it exists
        if model_exists:
            try:
                self.agent.load(model_path)
                print(f"Successfully loaded ensemble models from: {model_path}")
                
                # Print model weights
                weights = [f"{model['weight']:.3f}" for model in self.agent.models]
                print(f"Ensemble model weights: {weights}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Will run with a new ensemble (random policy)")
        else:
            print("Running with a new ensemble (random policy)")
        
        # Run simulation
        current_phase = 0
        phase_timer = 0
        is_yellow = False
        state = self.get_state(current_phase)
        step = 0
        
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
                    state = self.get_state(current_phase)
                    action = self.agent.act(state, training=False)
                    green_duration = self.action_values[action]
                else:
                    phase_timer += 1
            else:
                if phase_timer == 0:
                    action = self.agent.act(state, training=False)
                    green_duration = self.action_values[action]
                if phase_timer >= green_duration:
                    is_yellow = True
                    phase_timer = 0
                    if current_phase == 0:
                        traci.trafficlight.setPhase(self.intersection_id, 1)
                    else:
                        traci.trafficlight.setPhase(self.intersection_id, 3)
                else:
                    phase_timer += 1
            
            step += 1
        
        # Calculate and return metrics
        metrics = self._calculate_metrics(sim_duration)
        traci.close()
        return metrics

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
            "controller_type": "RL-Based",
            "avg_waiting_time": float(avg_waiting_time),
            "avg_speed": float(avg_speed),
            "total_vehicles": int(self.total_vehicles),
            "throughput": float(throughput),
            "we_avg_queue": float(we_avg_queue),
            "ns_avg_queue": float(ns_avg_queue),
            "completed_vehicles": int(completed),
            "duration": float(sim_duration)  # Add duration for recalculation if needed
        }

    def _plot_training_progress(self, rewards, waiting_times, speeds, throughputs, epsilon, losses, episode):
        """Plot training progress metrics with additional ensemble analysis"""
        try:
            plt.figure(figsize=(15, 12))
            
            # Plot rewards
            plt.subplot(3, 2, 1)
            plt.plot(rewards, 'b-')
            plt.title('Average Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            
            # Plot waiting times
            plt.subplot(3, 2, 2)
            plt.plot(waiting_times, 'r-')
            plt.title('Average Waiting Time')
            plt.xlabel('Episode')
            plt.ylabel('Time (s)')
            plt.grid(True)
            
            # Plot speeds
            plt.subplot(3, 2, 3)
            plt.plot(speeds, 'g-')
            plt.title('Average Vehicle Speed')
            plt.xlabel('Episode')
            plt.ylabel('Speed (m/s)')
            plt.grid(True)
            
            # Plot throughputs
            plt.subplot(3, 2, 4)
            plt.plot(throughputs, 'm-')
            plt.title('Vehicle Throughput')
            plt.xlabel('Episode')
            plt.ylabel('Throughput (veh/h)')
            plt.grid(True)
            
            # Plot epsilon decay
            plt.subplot(3, 2, 5)
            plt.plot(epsilon, 'k-')
            plt.title('Exploration Rate (Epsilon) Decay')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.grid(True)
            
            # Plot training loss
            if losses:
                plt.subplot(3, 2, 6)
                plt.plot(losses, 'c-')
                plt.title('Training Loss')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"plots/training/progress_ep{episode}.png", dpi=150)
            plt.close()
            
            # Plot model weights if available
            if hasattr(self.agent, 'models') and len(self.agent.models) > 1:
                plt.figure(figsize=(10, 6))
                model_weights = []
                for i, model in enumerate(self.agent.models):
                    model_weights.append(model['weight'])
                
                colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f'][:len(model_weights)]
                plt.bar(range(len(model_weights)), model_weights, color=colors)
                plt.title('Ensemble Model Weights')
                plt.xlabel('Model Index')
                plt.ylabel('Weight')
                plt.xticks(range(len(model_weights)))
                plt.ylim(0, 1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig(f"plots/training/model_weights_ep{episode}.png", dpi=150)
                plt.close()
                
        except Exception as e:
            print(f"Error plotting training progress: {e}")

    def _plot_training_summary(self, rewards, waiting_times, speeds, throughputs, epsilon, losses):
        """Create a comprehensive training summary visualization"""
        try:
            # Create a 3x2 subplot figure
            plt.figure(figsize=(15, 12))
            
            # Plot rewards with smoothed line
            plt.subplot(3, 2, 1)
            plt.plot(rewards, 'b-', alpha=0.5, label='Actual')
            # Add smoothed line using rolling average if enough data
            if len(rewards) >= 10:
                window = min(10, len(rewards)//5)
                smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                plt.plot(smoothed, 'r-', linewidth=2, label='Smoothed')
            plt.title('Average Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            
            # Plot waiting times
            plt.subplot(3, 2, 2)
            plt.plot(waiting_times, 'r-')
            plt.title('Average Waiting Time')
            plt.xlabel('Episode')
            plt.ylabel('Time (s)')
            plt.grid(True)
            
            # Plot speeds
            plt.subplot(3, 2, 3)
            plt.plot(speeds, 'g-')
            plt.title('Average Vehicle Speed')
            plt.xlabel('Episode')
            plt.ylabel('Speed (m/s)')
            plt.grid(True)
            
            # Plot throughputs
            plt.subplot(3, 2, 4)
            plt.plot(throughputs, 'm-')
            plt.title('Vehicle Throughput')
            plt.xlabel('Episode')
            plt.ylabel('Throughput (veh/h)')
            plt.grid(True)
            
            # Plot epsilon decay
            plt.subplot(3, 2, 5)
            plt.plot(epsilon, 'k-')
            plt.title('Exploration Rate (Epsilon) Decay')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.grid(True)
            
            # Plot training loss
            if losses:
                plt.subplot(3, 2, 6)
                plt.plot(losses, 'c-')
                plt.title('Training Loss')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("plots/training/training_summary.png", dpi=200)
            plt.close()
            
            # Plot final model weights if available
            if hasattr(self.agent, 'models') and len(self.agent.models) > 1:
                plt.figure(figsize=(10, 6))
                model_weights = []
                for i, model in enumerate(self.agent.models):
                    model_weights.append(model['weight'])
                
                colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f'][:len(model_weights)]
                bars = plt.bar(range(len(model_weights)), model_weights, color=colors)
                plt.title('Final Ensemble Model Weights')
                plt.xlabel('Model Index')
                plt.ylabel('Weight')
                plt.xticks(range(len(model_weights)))
                plt.ylim(0, 1)
                
                # Add weight values above bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
                
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig("plots/training/final_model_weights.png", dpi=200)
                plt.close()
            
            # Try to compare with other controllers
            self._compare_with_other_controllers(waiting_times, speeds, throughputs)
            
        except Exception as e:
            print(f"Error creating training summary: {e}")

    def _compare_with_other_controllers(self, waiting_times, speeds, throughputs):
        """Compare RL controller with fixed-time and density-based controllers"""
        # Check if we have baseline data from other controllers
        baseline_file = 'baseline_controllers.pkl'
        
        if not os.path.exists(baseline_file):
            # Create baseline file by running other controllers
            try:
                self._create_baseline_data(baseline_file)
            except Exception as e:
                print(f"Could not create baseline data: {e}")
                return
        
        try:
            # Load baseline data
            with open(baseline_file, 'rb') as f:
                baselines = pickle.load(f)
            
            # Create comparison plot
            plt.figure(figsize=(18, 6))
            
            # Get final values from RL training
            final_waiting = waiting_times[-1] if waiting_times else 0
            final_speed = speeds[-1] if speeds else 0
            final_throughput = throughputs[-1] if throughputs else 0
            
            # Plot waiting time comparison
            plt.subplot(1, 3, 1)
            controllers = ['Fixed-Time', 'Density-Based', 'RL Ensemble']
            values = [
                baselines.get('fixed_time', {}).get('avg_waiting_time', 0),
                baselines.get('density_based', {}).get('avg_waiting_time', 0),
                final_waiting
            ]
            bars = plt.bar(controllers, values, color=['#3498db', '#2ecc71', '#e74c3c'])
            plt.title('Average Waiting Time Comparison')
            plt.ylabel('Waiting Time (s)')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
            plt.annotate('Lower is better', xy=(0.5, 0.95), xycoords='axes fraction', 
                       ha='center', va='top', fontsize=10, style='italic')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Plot speed comparison
            plt.subplot(1, 3, 2)
            values = [
                baselines.get('fixed_time', {}).get('avg_speed', 0),
                baselines.get('density_based', {}).get('avg_speed', 0),
                final_speed
            ]
            bars = plt.bar(controllers, values, color=['#3498db', '#2ecc71', '#e74c3c'])
            plt.title('Average Speed Comparison')
            plt.ylabel('Speed (m/s)')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
            plt.annotate('Higher is better', xy=(0.5, 0.95), xycoords='axes fraction', 
                       ha='center', va='top', fontsize=10, style='italic')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Plot throughput comparison
            plt.subplot(1, 3, 3)
            values = [
                baselines.get('fixed_time', {}).get('throughput', 0),
                baselines.get('density_based', {}).get('throughput', 0),
                final_throughput
            ]
            bars = plt.bar(controllers, values, color=['#3498db', '#2ecc71', '#e74c3c'])
            plt.title('Throughput Comparison')
            plt.ylabel('Throughput (veh/h)')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
            plt.annotate('Higher is better', xy=(0.5, 0.95), xycoords='axes fraction', 
                       ha='center', va='top', fontsize=10, style='italic')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("plots/training/controller_comparison.png", dpi=200)
            plt.close()
            
        except Exception as e:
            print(f"Error creating controller comparison: {e}")

    def _create_baseline_data(self, output_file):
        """Run fixed-time and density-based controllers to create baseline data"""
        from controllers.fixed_time import FixedTimeController
        from controllers.density_based import DensityBasedController
        
        baseline_data = {}
        
        # Run fixed-time controller
        print("Running fixed-time controller to create baseline...")
        fixed_controller = FixedTimeController()
        fixed_metrics = fixed_controller.run(["sumo", "-c", "network/simulation.sumocfg", "--no-warnings"])
        baseline_data['fixed_time'] = fixed_metrics
        
        # Run density-based controller
        print("Running density-based controller to create baseline...")
        density_controller = DensityBasedController()
        density_metrics = density_controller.run(["sumo", "-c", "network/simulation.sumocfg", "--no-warnings"])
        baseline_data['density_based'] = density_metrics
        
        # Save baseline data
        with open(output_file, 'wb') as f:
            pickle.dump(baseline_data, f)
        
        print(f"Baseline data saved to {output_file}")
        
        return baseline_data

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch is not installed. Please install it with: pip install torch")
        sys.exit(1)
    
    # Create controller and run
    controller = RLController()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run RL traffic light controller")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    parser.add_argument("--model", type=str, default="models/rl_controller_final.pt", help="Path to model file")
    parser.add_argument("--ensemble-size", type=int, default=3, help="Number of models in the ensemble")
    args = parser.parse_args()
    
    # Set ensemble size if specified
    if args.ensemble_size > 1:
        controller.ensemble_size = args.ensemble_size
        print(f"Using ensemble of {args.ensemble_size} models")
    
    # Run simulation
    sumo_cmd = ["sumo", "-c", "network/simulation.sumocfg", "--no-warnings"]
    
    if args.train:
        controller.train(sumo_cmd, episodes=args.episodes, sim_duration=args.duration)
    
    metrics = controller.run(sumo_cmd, model_path=args.model, sim_duration=args.duration)
    
    print("\nRL Ensemble Controller Results:")
    print(f"Average Waiting Time: {metrics['avg_waiting_time']:.2f} seconds")
    print(f"Average Speed: {metrics['avg_speed']:.2f} m/s")
    print(f"Throughput: {metrics['throughput']:.2f} vehicles/hour")