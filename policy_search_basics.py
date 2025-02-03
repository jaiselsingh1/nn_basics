import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class PointMassEnv:
    """Simple 1D point mass environment"""
    def __init__(self, mass=1.0, dt=0.1, target=10.0):
        self.mass = mass
        self.dt = dt
        self.target = target
        self.reset()
        
    def reset(self) -> np.ndarray:
        self.position = 0.0
        self.velocity = 0.0
        return np.array([self.position, self.velocity])
        
    def step(self, force: float) -> Tuple[np.ndarray, float]:
        acceleration = force / self.mass
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # Calculate reward (negative distance to target)
        reward = -abs(self.position - self.target)
        
        return np.array([self.position, self.velocity]), reward

def pid_controller(state: np.ndarray, target: float, kp=1.0, kd=0.5) -> float:
    position, velocity = state
    
    # Error terms
    position_error = target - position
    velocity_error = -velocity  # We want to stop at the target
    
    # Control law
    force = kp * position_error + kd * velocity_error
    return force

class PolicyNetwork:
    """Simple linear policy for policy search"""
    def __init__(self, input_dim=2, output_dim=1):
        # Initialize random weights
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        
    def __call__(self, state: np.ndarray) -> float:
        """Forward pass to get action"""
        return np.dot(state, self.weights)[0]
    
    def update(self, new_weights: np.ndarray):
        """Update policy parameters"""
        self.weights = new_weights

def simple_policy_search(env: PointMassEnv, episodes=100, population_size=10, elite_frac=0.2):
    """Simple policy search using a basic evolutionary strategy"""
    policy = PolicyNetwork()
    n_elite = int(population_size * elite_frac)
    
    for episode in range(episodes):
        # Generate population of policies
        noises = np.random.randn(population_size, 2, 1) * 0.1   # noises are how we're going to change the policy and vary it to see how to improve the rewards
        rewards = []
        
        # Evaluate each policy
        for noise in noises:
            policy.update(policy.weights + noise)
            state = env.reset()
            episode_reward = 0
            
            # Run episode
            for _ in range(50):  # Max steps per episode
                action = policy(state)
                state, reward = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        # Select elite policies and update
        elite_idxs = np.argsort(rewards)[-n_elite:]
        elite_noises = noises[elite_idxs]
        
        # Update policy weights using elite members
        policy.update(policy.weights + np.mean(elite_noises, axis=0) * 0.1)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Best Reward: {max(rewards)}")
    
    return policy

def compare_controllers():
    """Compare traditional PID vs learned policy"""
    env = PointMassEnv()
    
    # Train policy search controller
    learned_policy = simple_policy_search(env)
    
    # Run and compare both controllers
    pid_trajectory = []
    policy_trajectory = []
    
    # PID control
    state = env.reset()
    for _ in range(50):
        action = pid_controller(state, env.target)
        state, _ = env.step(action)
        pid_trajectory.append(state[0])
    
    # Policy search control
    state = env.reset()
    for _ in range(50):
        action = learned_policy(state)
        state, _ = env.step(action)
        policy_trajectory.append(state[0])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(pid_trajectory, label='PID Control')
    plt.plot(policy_trajectory, label='Policy Search')
    plt.axhline(y=env.target, color='r', linestyle='--', label='Target')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('PID vs Policy Search Control')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_controllers()












              

                        
                           









