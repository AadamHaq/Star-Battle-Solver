"""
Q-Learning implementation for the Queens puzzle solver.
This implementation learns from training data to solve queen placement puzzles
of various sizes while respecting color region constraints.
"""

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Optional, Dict
import copy
import os
import json

class LinkedInQueensEnvironment:
    def __init__(self, board: List[List[int]]):
        self.original_board = np.array(board)
        self.n = len(board)
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state"""
        self.board = self.original_board.copy()
        self.queens = set()  # Set of (row, col) positions
        self.current_step = 0
        self.violation_counts = defaultdict(int)  # Track rule violations for learning
        return self.get_state()
    
    def get_regions(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get all unique regions and their cells"""
        regions = defaultdict(list)
        for i in range(self.n):
            for j in range(self.n):
                region_id = self.board[i, j]
                regions[region_id].append((i, j))
        return regions
    
    def step(self, action: Tuple[int, int]) -> Tuple[str, float, bool, dict]:
        """Take an action and return (next_state, reward, done, info)"""
        row, col = action
        
        # Check if position is already occupied
        if (row, col) in self.queens:
            return self.get_state(), -5, False, {"already_occupied": True}
        
        # Place the queen (always allowed - let agent learn through rewards)
        self.queens.add((row, col))
        self.current_step += 1
        
        # Calculate reward based on rule violations
        reward, violations = self.calculate_reward_and_violations(row, col)
        
        # Update violation tracking for learning
        for violation_type in violations:
            self.violation_counts[violation_type] += 1
        
        # Check if done (either solved or max queens placed)
        done = len(self.queens) >= self.n or self.current_step >= self.n * 2
        solved = self.is_perfectly_solved()
        
        info = {
            "queens_placed": len(self.queens),
            "violations": violations,
            "solved": solved,
            "violation_counts": dict(self.violation_counts)
        }
        
        return self.get_state(), reward, done, info
    
    def calculate_reward_and_violations(self, new_row: int, new_col: int) -> Tuple[float, List[str]]:
        """Calculate reward and identify violations for the newly placed queen"""
        violations = []
        reward = 1.0  # Base reward for placing a queen
        
        # Get the region of the new queen
        new_region = self.board[new_row, new_col]
        
        # Check violations against all existing queens
        for q_row, q_col in self.queens:
            if q_row == new_row and q_col == new_col:
                continue  # Skip the newly placed queen
            
            # Same row violation
            if q_row == new_row:
                violations.append("same_row")
                reward -= 2.0
            
            # Same column violation
            if q_col == new_col:
                violations.append("same_column")
                reward -= 2.0
            
            # Same region violation
            if self.board[q_row, q_col] == new_region:
                violations.append("same_region")
                reward -= 2.0
            
            # Adjacent violation (including diagonals)
            if abs(q_row - new_row) <= 1 and abs(q_col - new_col) <= 1:
                violations.append("adjacent")
                reward -= 1.5
        
        # Bonus rewards for good placement
        if not violations:
            reward += 3.0  # Clean placement bonus
            
        # Progressive bonus for more queens without violations
        if len(self.queens) > 1 and not violations:
            reward += len(self.queens) * 0.5
        
        # Huge bonus for perfect solution
        if len(self.queens) == self.n and self.is_perfectly_solved():
            reward += 50.0
        
        # Penalty for too many violations
        total_violations = sum(self.violation_counts.values()) + len(violations)
        if total_violations > self.n:
            reward -= total_violations * 0.1
        
        return reward, violations
    
    def is_perfectly_solved(self) -> bool:
        """Check if the puzzle is perfectly solved"""
        if len(self.queens) != self.n:
            return False
        
        queens_list = list(self.queens)
        
        # Check one queen per row
        rows = set(q[0] for q in queens_list)
        if len(rows) != self.n:
            return False
        
        # Check one queen per column
        cols = set(q[1] for q in queens_list)
        if len(cols) != self.n:
            return False
        
        # Check one queen per region
        regions = set(self.board[q[0], q[1]] for q in queens_list)
        unique_regions = set(self.board.flatten())
        if len(regions) != len(unique_regions):
            return False
        
        # Check no adjacent queens
        for i, (r1, c1) in enumerate(queens_list):
            for j, (r2, c2) in enumerate(queens_list):
                if i != j and abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    return False
        
        return True
    
    def get_state(self) -> str:
        """Get current state as a string for Q-table indexing"""
        # Create a compact state representation
        state_info = {
            'queens': sorted(list(self.queens)),
            'step': self.current_step
        }
        return str(hash(str(state_info)))
    
    def get_state_features(self) -> Tuple:
        """Get state features for more sophisticated state representation"""
        if not self.queens:
            return (0, 0, 0, 0, 0)  # No queens placed
        
        # Features: num_queens, row_conflicts, col_conflicts, region_conflicts, adjacent_conflicts
        queens_list = list(self.queens)
        
        row_conflicts = len(queens_list) - len(set(q[0] for q in queens_list))
        col_conflicts = len(queens_list) - len(set(q[1] for q in queens_list))
        
        regions = [self.board[q[0], q[1]] for q in queens_list]
        region_conflicts = len(regions) - len(set(regions))
        
        adjacent_conflicts = 0
        for i, (r1, c1) in enumerate(queens_list):
            for j, (r2, c2) in enumerate(queens_list):
                if i < j and abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    adjacent_conflicts += 1
        
        return (len(queens_list), row_conflicts, col_conflicts, region_conflicts, adjacent_conflicts)
    
    def render(self):
        """Visualize the current board state"""
        display_board = self.board.copy().astype(str)
        for row, col in self.queens:
            display_board[row, col] = 'Q'
        
        print(f"Step {self.current_step}, Queens: {len(self.queens)}")
        for row in display_board:
            print(' '.join(f'{cell:>2}' for cell in row))
        print(f"Violations: {dict(self.violation_counts)}")
        print(f"Solved: {self.is_perfectly_solved()}")
        print()

class OrganicQLearningAgent:
    def __init__(self, board_size: int, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.board_size = board_size
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = [(i, j) for i in range(board_size) for j in range(board_size)]
        
        # Learning statistics
        self.rule_discovery = {
            'same_row': 0,
            'same_column': 0,
            'same_region': 0,
            'adjacent': 0
        }
        
    def get_action(self, state: str, occupied_positions: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Choose action using epsilon-greedy policy"""
        # Filter out already occupied positions
        available_actions = [action for action in self.action_space if action not in occupied_positions]
        
        if not available_actions:
            return random.choice(self.action_space)  # Fallback
        
        if random.random() < self.epsilon:
            # Explore: random action from available positions
            return random.choice(available_actions)
        else:
            # Exploit: best known action from available positions
            action_values = {action: self.q_table[state][str(action)] for action in available_actions}
            if not action_values:
                return random.choice(available_actions)
            return max(action_values.keys(), key=lambda k: action_values[k])
    
    def update_q_value(self, state: str, action: Tuple[int, int], reward: float, 
                      next_state: str, done: bool, info: dict):
        """Update Q-value using Q-learning update rule"""
        action_str = str(action)
        current_q = self.q_table[state][action_str]
        
        if done:
            max_next_q = 0
        else:
            # Estimate future value (simplified)
            next_state_values = list(self.q_table[next_state].values())
            max_next_q = max(next_state_values) if next_state_values else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_str] = new_q
        
        # Track rule discovery
        for violation in info.get('violations', []):
            if violation in self.rule_discovery:
                self.rule_discovery[violation] += 1
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def load_training_data(data_dir: str = "/RL_Resources/Training_Data/Metadata") -> List[Tuple[List[List[int]], List[Tuple[int, int]]]]:
    """Load all training boards and solutions from metadata files"""
    training_data = []
    
    # Try different possible paths relative to the current file location
    possible_paths = [
        data_dir,
        "../RL_Resources/Training_Data/Metadata",
        "../../RL_Resources/Training_Data/Metadata", 
        "RL_Resources/Training_Data/Metadata",
        "Training_Data/Metadata",
        "Metadata"
    ]
    
    metadata_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            metadata_dir = path
            print(f"Found metadata directory at: {os.path.abspath(path)}")
            break
    
    if metadata_dir is None:
        print(f"Warning: Could not find metadata directory. Tried paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        return []
    
    # Get all .txt files in the metadata directory
    txt_files = [f for f in os.listdir(metadata_dir) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} .txt files in {metadata_dir}")
    
    for filename in txt_files:
        filepath = os.path.join(metadata_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Parse the file content to extract queens_board and solved_board
            queens_board = None
            solved_board = None
            
            # Execute the file content to get the variables
            local_vars = {}
            exec(content, {}, local_vars)
            
            if 'queens_board' in local_vars:
                queens_board = local_vars['queens_board']
            
            if 'solved_board' in local_vars:
                solved_board = local_vars['solved_board']
            
            if queens_board is not None and solved_board is not None:
                training_data.append((queens_board, solved_board))
            else:
                print(f"Warning: Could not find queens_board or solved_board in {filename}")
                
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            continue
    
    print(f"Successfully loaded {len(training_data)} training boards with solutions")
    return training_data

def train_organic_agent(training_data: List[Tuple[List[List[int]], List[Tuple[int, int]]]], 
                       episodes_per_board: int = 20, verbose: bool = True) -> Tuple[OrganicQLearningAgent, List[float]]:
    """Train the agent organically across multiple boards"""
    if not training_data:
        raise ValueError("No training data provided")
    
    board_size = len(training_data[0][0])
    agent = OrganicQLearningAgent(board_size)
    
    total_episodes = 0
    solutions_found = 0
    episode_rewards = []
    perfect_solutions = 0
    
    for board_idx, (board, solution) in enumerate(training_data):
        if verbose and board_idx % 50 == 0:
            print(f"Training on board {board_idx + 1}/{len(training_data)}")
        
        env = LinkedInQueensEnvironment(board)
        
        for episode in range(episodes_per_board):
            state = env.reset()
            total_reward = 0
            steps = 0
            max_steps = board_size * 2
            
            while steps < max_steps:
                action = agent.get_action(state, env.queens)
                next_state, reward, done, info = env.step(action)
                
                # Additional reward shaping based on known solution
                if action in solution:
                    reward += 1.0  # Bonus for correct placement
                
                agent.update_q_value(state, action, reward, next_state, done, info)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    if info.get('solved', False):
                        solutions_found += 1
                        if set(env.queens) == set(solution):
                            perfect_solutions += 1
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            total_episodes += 1
    
    if verbose:
        print(f"\nTraining completed!")
        print(f"Total episodes: {total_episodes}")
        print(f"Solutions found: {solutions_found}")
        print(f"Perfect solutions: {perfect_solutions}")
        print(f"Success rate: {solutions_found/total_episodes*100:.2f}%")
        print(f"Perfect rate: {perfect_solutions/total_episodes*100:.2f}%")
        print(f"Rule discovery stats: {agent.rule_discovery}")
    
    return agent, episode_rewards

def test_agent_on_board(board: List[List[int]], solution: List[Tuple[int, int]], 
                       agent: OrganicQLearningAgent, render: bool = True) -> bool:
    """Test the trained agent on a specific board with known solution"""
    env = LinkedInQueensEnvironment(board)
    state = env.reset()
    
    if render:
        print("Testing agent on board:")
        print(f"Expected solution: {solution}")
        env.render()
    
    # Use greedy policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    steps = 0
    max_steps = len(board) * 2
    
    while steps < max_steps:
        action = agent.get_action(state, env.queens)
        state, reward, done, info = env.step(action)
        steps += 1
        
        if render:
            correct_move = "✓" if action in solution else "✗"
            print(f"Step {steps}: Placed queen at {action} {correct_move}, Reward: {reward:.2f}")
            if info.get('violations'):
                print(f"Violations: {info['violations']}")
            env.render()
        
        if done:
            success = info.get('solved', False)
            perfect_match = set(env.queens) == set(solution)
            
            if success and perfect_match:
                print("SUCCESS: Perfect solution found!")
            elif success:
                print("SUCCESS: Valid solution found (different from expected)")
            else:
                print(f"FAILED: Could not solve. Final state: {len(env.queens)} queens placed")
                print(f"Agent's solution: {sorted(list(env.queens))}")
                print(f"Expected solution: {sorted(solution)}")
            
            agent.epsilon = original_epsilon
            return success
    
    print("TIMEOUT: Max steps reached")
    agent.epsilon = original_epsilon
    return False

# Main execution
if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    training_data = load_training_data()
    
    if not training_data:
        print("No training data found. Using example board.")
        example_board = [
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 2, 2, 2, 2, 2, 2, 2, 1],
            [2, 2, 3, 2, 3, 2, 3, 2, 2],
            [2, 3, 3, 3, 3, 3, 3, 3, 4],
            [2, 2, 5, 3, 6, 6, 7, 4, 4],
            [2, 5, 5, 6, 6, 7, 7, 7, 4],
            [2, 2, 5, 8, 6, 8, 7, 8, 4],
            [8, 2, 2, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]
        example_solution = [(0, 7), (2, 4), (6, 6), (8, 3), (1, 0), (5, 2), (3, 8), (7, 1), (4, 5)]
        training_data = [(example_board, example_solution)]
    
    # Train the agent
    print("Training organic Q-learning agent...")
    num_boards_to_train = min(50, len(training_data))
    trained_agent, rewards = train_organic_agent(training_data[:num_boards_to_train], episodes_per_board=15)
    
    # Test on a few boards
    print("\nTesting trained agent:")
    test_data = training_data[-5:] if len(training_data) > 5 else training_data[:1]
    
    successes = 0
    for i, (test_board, test_solution) in enumerate(test_data):
        print(f"\n--- Test Board {i+1} ---")
        success = test_agent_on_board(test_board, test_solution, trained_agent, render=True)
        if success:
            successes += 1
    
    print(f"\nFinal Test Results: {successes}/{len(test_data)} boards solved")
    
    # Plot training progress
    if len(rewards) > 100:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 3, 2)
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg)
            plt.title(f'Moving Average Reward (window={window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
        
        plt.subplot(1, 3, 3)
        rule_types = list(trained_agent.rule_discovery.keys())
        rule_counts = list(trained_agent.rule_discovery.values())
        plt.bar(rule_types, rule_counts)
        plt.title('Rule Violations Discovered')
        plt.xlabel('Rule Type')
        plt.ylabel('Times Encountered')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()