import numpy as np
import random
import json
import time 

import tkinter as tk 

from collections import defaultdict
import matplotlib.pyplot as plt

from utils import get_config, debug, calculate_intervals
from dqn_plotter import NimTrainingVisualizer

# Load configuration settings
config = get_config()["game"]
dqn_agent = get_config()["dqn_agent"]

epsilon_pre = {"initial": dqn_agent["epsilon_initial"], "decay": dqn_agent["epsilon_decay"], "min": dqn_agent["epsilon_min"]}
discount_factor_pre = get_config()["dqn_agent"]["discount_factor"]
learning_rate_pre = get_config()["dqn_agent"]["learning_rate"]

def run_training_with_visualization():
    """
    Run training with proper GUI event handling.
    Call this function instead of train_nim_agent directly if you want visualization.
    """
    import threading
    
    # This will hold our training results
    results = {'agent': None, 'win_rates': None, 'completed': False}
    
    def training_thread():
        try:
            agent, win_rates = train_nim_agent()
            results['agent'] = agent
            results['win_rates'] = win_rates
            results['completed'] = True
            print("Training thread completed successfully")
        except Exception as e:
            print(f"Training error: {e}")
            results['completed'] = True
    
    # Start training in separate thread
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    # Keep main thread alive for GUI
    while not results['completed']:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("Training interrupted by user")
            break
    
    return results['agent'], results['win_rates']

class NimGame:
    """Nim Game Environment"""
    
    def __init__(self, initial_sticks=config["max_sticks"]):
        self.initial_sticks = initial_sticks
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        self.current_sticks = self.initial_sticks
        self.game_over = False
        self.winner = None
        return self.current_sticks
    
    def current_state(self):
        """Get current state of the game"""
        return self.current_sticks
    
    def step(self, action):
        """Make a move and return new state, reward, done"""
        if self.game_over:
            return self.current_sticks, 0, True
        
        # Validate action
        action = max(1, min(3, min(action, self.current_sticks)))
        
        # Apply action
        self.current_sticks -= action
        
        # Check if game is over
        if self.current_sticks <= 0:
            self.game_over = True
            # Player who took last stick loses (reward = -1)
            return 0, -1, True
        
        return self.current_sticks, 0, False
    
    def get_valid_actions(self):
        """Get list of valid actions"""
        if self.game_over:
            return []
        return list(range(1, min(4, self.current_sticks + 1)))

class QLearningAgent:
    """Q-Learning Agent for Nim"""
    
    def __init__(self, learning_rate=learning_rate_pre, discount_factor=discount_factor_pre  , epsilon=epsilon_pre["initial"]):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Statistics
        self.wins = 0
        self.losses = 0
        self.games_played = 0
    
    def get_action(self, state, valid_actions, training=True):
        """Choose action using epsilon-greedy policy"""
        if not valid_actions:
            return None
        
        # Exploration vs Exploitation
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Choose action with highest Q-value
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        if next_valid_actions:
            # Max Q-value for next state
            next_max_q = max([self.q_table[next_state][a] for a in next_valid_actions])
        else:
            next_max_q = 0
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def update_stats(self, won):
        """Update win/loss statistics"""
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_win_rate(self):
        """Get current win rate"""
        if self.games_played == 0:
            return 0
        return self.wins / self.games_played

class RandomAgent:
    """Random agent for training opponent"""
    
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.games_played = 0
    
    def get_action(self, state, valid_actions, training=True):
        if not valid_actions:
            return None
        return random.choice(valid_actions)
    
    def update_q_value(self, *args):
        pass  # Random agent doesn't learn
    
    def update_stats(self, won):
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_win_rate(self):
        if self.games_played == 0:
            return 0
        return self.wins / self.games_played

class OptimalAgent:
    """Mathematically optimal agent"""
    
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.games_played = 0
    
    def get_action(self, state, valid_actions, training=True):
        """Optimal strategy: leave opponent with (4n+1) sticks"""
        if not valid_actions:
            return None
        
        # Try to leave opponent with 1, 5, 9, 13, ... sticks
        for action in valid_actions:
            remaining = state - action
            if remaining % 4 == 1 or remaining == 0:
                return action
        
        # If no optimal move, take 1 stick
        return 1 if 1 in valid_actions else valid_actions[0]
    
    def update_q_value(self, *args):
        pass  # Optimal agent doesn't learn
    
    def update_stats(self, won):
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_win_rate(self):
        if self.games_played == 0:
            return 0
        return self.wins / self.games_played

def play_game(game, agent1, agent2, training=True):
    """Play one game between two agents"""
    state = game.reset()
    current_player = 0
    agents = [agent1, agent2]
    
    # Store experience for learning
    experiences = [[], []]  # [state, action, reward, next_state, next_valid_actions]
    
    while not game.game_over:
        current_agent = agents[current_player]
        valid_actions = game.get_valid_actions()
        
        # Get action
        action = current_agent.get_action(state, valid_actions, training)
        if action is None:
            break
        
        # Store current state and action
        experiences[current_player].append([state, action, 0, None, None])
        
        # Take action
        next_state, reward, done = game.step(action)
        
        # Update experience with next state
        experiences[current_player][-1][3] = next_state
        experiences[current_player][-1][4] = game.get_valid_actions()
        
        if done:
            # Verlierer bekommt -1
            experiences[current_player][-1][2] = -1
            # Gewinner bekommt +1
            experiences[1 - current_player][-1][2] = 1

            # Update Q-Werte
            for player_idx, player_exp in enumerate(experiences):
                agent = agents[player_idx]
                for exp in player_exp:
                    if len(exp) == 5:
                        agent.update_q_value(exp[0], exp[1], exp[2], exp[3], exp[4])

            # Update Stats
            loser = current_player
            winner = 1 - current_player
            agents[winner].update_stats(True)
            agents[loser].update_stats(False)

            return winner
        
        state = next_state
        current_player = 1 - current_player
    
    return None

def train_nim_agent(episodes=10000, initial_sticks=config["max_sticks"], save_path="nim_agent.json"):
    game = NimGame(initial_sticks)
    q_agent = QLearningAgent(learning_rate=0.1, epsilon=epsilon_pre["initial"])
    opponents = {
        'random': RandomAgent(),
        'optimal': OptimalAgent()
    }
    
    update_every, save_every = calculate_intervals(episodes)
    
    if debug():
        print(f"Training with {episodes} episodes, initial sticks: {initial_sticks}")
        print(f"Update every {update_every} episodes, save every {save_every} episodes")
    
    win_rates = []
    
    # Create visualizer if debug is enabled
    visualizer = None
    if debug():
        from dqn_plotter import create_training_visualizer
        visualizer = create_training_visualizer(total_episodes=episodes, max_sticks=initial_sticks)
        print("Visualizer created - training window should be visible")
    
    print("Training Q-Learning Agent...")
    
    # Keep track of recent wins for calculating win rate
    recent_games = []
    window_size = max(100, episodes // 50)  # Window for calculating win rate
    
    for episode in range(episodes):
        # Epsilon decay
        if episode > episodes / 4:
            progress = (episode - episodes / 4) / (episodes * 0.75)
            q_agent.epsilon = max(epsilon_pre['min'], 1.0 - progress * (1.0 - epsilon_pre['min']))
        
        # Opponent selection
        opponent_name = 'random' if random.random() < dqn_agent["random_train_opponent"] else 'optimal'
        opponent = opponents[opponent_name]
        
        # Who starts randomly?
        if random.random() < 0.5:
            winner = play_game(game, q_agent, opponent, training=True)
            q_agent_won = (winner == 0)
        else:
            winner = play_game(game, opponent, q_agent, training=True)
            q_agent_won = (winner == 1)
        
        # Track recent performance
        recent_games.append(1 if q_agent_won else 0)
        if len(recent_games) > window_size:
            recent_games.pop(0)
        
        # Update visualizer at specified intervals
        if episode % update_every == 0:
            # Calculate current win rate from recent games
            if recent_games:
                current_win_rate = sum(recent_games) / len(recent_games)
            else:
                current_win_rate = 0.0
            
            win_rates.append(current_win_rate)
            
            print(f"Episode {episode}: Win Rate = {current_win_rate:.3f}, Epsilon = {q_agent.epsilon:.3f}")
            
            # Update visualizer with real data
            if visualizer is not None:
                try:
                    visualizer.update(current_win_rate, q_agent.epsilon)
                except tk.TclError:
                    # Window was closed
                    print("Visualizer window closed, continuing training without visualization")
                    visualizer = None
        
        # Save progress periodically
        if episode % save_every == 0 and episode > 0:
            save_nim_agent(q_agent, save_path)
            if debug():
                print(f"Progress saved at episode {episode}")
    
    # Final save
    save_nim_agent(q_agent, save_path)
    print(f"Training completed! Agent saved to {save_path}")
    
    # Close visualizer
    if visualizer is not None:
        try:
            # Keep window open for a few seconds to show final results
            visualizer.after(3000, visualizer.close)
            print("Training complete - visualizer will close automatically in 3 seconds")
        except tk.TclError:
            pass
    
    return q_agent, win_rates


def evaluate_nim_agent(agent, episodes=1000, initial_sticks=15):
    """Evaluate trained agent against different opponents"""
    game = NimGame(initial_sticks)
    opponents = {
        'Random': RandomAgent(),
        'Optimal': OptimalAgent()
    }
    
    results = {}
    
    for opp_name, opponent in opponents.items():
        wins = 0
        for _ in range(episodes):
            # Play as first player
            winner = play_game(game, agent, opponent, training=False)
            if winner == 0:
                wins += 1
            
            # Play as second player  
            winner = play_game(game, opponent, agent, training=False)
            if winner == 1:
                wins += 1
        
        win_rate = wins / (episodes * 2)
        results[opp_name] = win_rate
        print(f"vs {opp_name}: {win_rate:.3f} win rate")
    
    return results

def save_nim_agent(agent, filename):
    """Save trained agent"""
    data = {
        'q_table': dict(agent.q_table),
        'stats': {
            'wins': agent.wins,
            'losses': agent.losses,
            'games_played': agent.games_played
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_nim_agent(filename):
    """Load trained agent"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    agent = QLearningAgent(epsilon=0.0)  # No exploration when loaded
    agent.q_table = defaultdict(lambda: defaultdict(float))
    
    for state, actions in data['q_table'].items():
        for action, q_value in actions.items():
            agent.q_table[int(state)][int(action)] = q_value
    
    agent.wins = data['stats']['wins']
    agent.losses = data['stats']['losses']
    agent.games_played = data['stats']['games_played']
    
    return agent
