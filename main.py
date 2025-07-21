import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageFont
import threading
import time

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class TetrisEnv:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width))
        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False

        # Tetris pieces (7 standard pieces)
        self.pieces = {
            "I": [
                [".....", "..#..", "..#..", "..#..", "..#.."],
                [".....", ".....", "####.", ".....", "....."],
            ],
            "O": [[".....", ".....", ".##..", ".##..", "....."]],
            "T": [
                [".....", ".....", ".#...", "###..", "....."],
                [".....", ".....", ".#...", ".##..", ".#..."],
                [".....", ".....", ".....", "###..", ".#..."],
                [".....", ".....", ".#...", "##...", ".#..."],
            ],
            "S": [
                [".....", ".....", ".##..", "##...", "....."],
                [".....", ".....", ".#...", ".##..", "..#.."],
            ],
            "Z": [
                [".....", ".....", "##...", ".##..", "....."],
                [".....", ".....", "..#..", ".##..", ".#..."],
            ],
            "J": [
                [".....", ".....", ".#...", ".#...", "##..."],
                [".....", ".....", "#....", "###..", "....."],
                [".....", ".....", ".##..", ".#...", ".#..."],
                [".....", ".....", ".....", "###..", "..#.."],
            ],
            "L": [
                [".....", ".....", ".#...", ".#...", ".##.."],
                [".....", ".....", ".....", "###..", "#...."],
                [".....", ".....", "##...", ".#...", ".#..."],
                [".....", ".....", "..#..", "###..", "....."],
            ],
        }

        self.piece_names = list(self.pieces.keys())
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width))
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.spawn_piece()
        return self.get_state()

    def spawn_piece(self):
        piece_name = random.choice(self.piece_names)
        self.current_piece = {
            "shape": piece_name,
            "rotation": 0,
            "grid": self.pieces[piece_name][0],
        }
        self.piece_x = self.width // 2 - 2
        self.piece_y = 0

        if not self.is_valid_position():
            self.game_over = True

    def is_valid_position(self, dx=0, dy=0, rotation=None):
        if rotation is None:
            rotation = self.current_piece["rotation"]

        shape = self.current_piece["shape"]
        grid = self.pieces[shape][rotation % len(self.pieces[shape])]

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    new_x = self.piece_x + j + dx
                    new_y = self.piece_y + i + dy

                    if (
                        new_x < 0
                        or new_x >= self.width
                        or new_y >= self.height
                        or (new_y >= 0 and self.board[new_y][new_x] == 1)
                    ):
                        return False
        return True

    def place_piece(self):
        shape = self.current_piece["shape"]
        rotation = self.current_piece["rotation"]
        grid = self.pieces[shape][rotation % len(self.pieces[shape])]

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    board_x = self.piece_x + j
                    board_y = self.piece_y + i
                    if board_y >= 0:
                        self.board[board_y][board_x] = 1

    def clear_lines(self):
        lines_to_clear = []
        for i in range(self.height):
            if np.all(self.board[i] == 1):
                lines_to_clear.append(i)

        for line in lines_to_clear:
            self.board = np.delete(self.board, line, axis=0)
            self.board = np.vstack([np.zeros((1, self.width)), self.board])

        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared

        # Scoring system
        if lines_cleared == 1:
            self.score += 40
        elif lines_cleared == 2:
            self.score += 100
        elif lines_cleared == 3:
            self.score += 300
        elif lines_cleared == 4:
            self.score += 1200

        return lines_cleared

    def step(self, action):
        # Actions: 0=left, 1=right, 2=rotate, 3=down, 4=drop
        reward = 0
        lines_cleared_before = self.lines_cleared

        if action == 0:  # Move left
            if self.is_valid_position(dx=-1):
                self.piece_x -= 1
        elif action == 1:  # Move right
            if self.is_valid_position(dx=1):
                self.piece_x += 1
        elif action == 2:  # Rotate
            new_rotation = (self.current_piece["rotation"] + 1) % len(
                self.pieces[self.current_piece["shape"]]
            )
            if self.is_valid_position(rotation=new_rotation):
                self.current_piece["rotation"] = new_rotation
        elif action == 3:  # Move down
            if self.is_valid_position(dy=1):
                self.piece_y += 1
                reward += 1  # Small reward for moving down
            else:
                self.place_piece()
                lines_cleared = self.clear_lines()
                reward += lines_cleared * 100  # Big reward for clearing lines
                self.spawn_piece()
        elif action == 4:  # Hard drop
            while self.is_valid_position(dy=1):
                self.piece_y += 1
                reward += 2
            self.place_piece()
            lines_cleared = self.clear_lines()
            reward += lines_cleared * 100
            self.spawn_piece()

        # Additional rewards based on board state
        height_penalty = self.get_height() * -0.51
        hole_penalty = self.get_holes() * -25
        bumpiness_penalty = self.get_bumpiness() * -10

        reward += height_penalty + hole_penalty + bumpiness_penalty

        return self.get_state(), reward, self.game_over

    def get_state(self):
        # Extract features for the neural network
        features = []

        # Board state (flattened)
        features.extend(self.board.flatten())

        # Height of each column
        heights = self.get_column_heights()
        features.extend(heights)

        # Number of holes
        features.append(self.get_holes())

        # Bumpiness
        features.append(self.get_bumpiness())

        # Lines cleared
        features.append(self.lines_cleared)

        # Current piece info (one-hot encoded)
        piece_encoding = [0] * len(self.piece_names)
        if self.current_piece:
            piece_idx = self.piece_names.index(self.current_piece["shape"])
            piece_encoding[piece_idx] = 1
        features.extend(piece_encoding)

        return np.array(features, dtype=np.float32)

    def get_column_heights(self):
        heights = []
        for col in range(self.width):
            height = 0
            for row in range(self.height):
                if self.board[row][col] == 1:
                    height = self.height - row
                    break
            heights.append(height)
        return heights

    def get_holes(self):
        holes = 0
        for col in range(self.width):
            block_found = False
            for row in range(self.height):
                if self.board[row][col] == 1:
                    block_found = True
                elif self.board[row][col] == 0 and block_found:
                    holes += 1
        return holes

    def get_bumpiness(self):
        heights = self.get_column_heights()
        bumpiness = sum(
            abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1)
        )
        return bumpiness

    def get_height(self):
        return max(self.get_column_heights())


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 5),  # 5 actions
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_size, action_size=5, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_size).to(self.device)
        self.target_network = DQN(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "epsilon": self.epsilon,
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.epsilon = checkpoint.get("epsilon", 0.01)
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class TetrisVisualizer:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Tetris")

        # Colors
        self.colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "cyan": (0, 255, 255),
            "blue": (0, 0, 255),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "green": (0, 255, 0),
            "purple": (128, 0, 128),
            "red": (255, 0, 0),
            "gray": (128, 128, 128),
        }

        self.block_size = 25
        self.board_x = 50
        self.board_y = 50

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def draw_board(self, env):
        # Draw border
        pygame.draw.rect(
            self.screen,
            self.colors["white"],
            (
                self.board_x - 2,
                self.board_y - 2,
                env.width * self.block_size + 4,
                env.height * self.block_size + 4,
            ),
            2,
        )

        # Draw placed blocks
        for row in range(env.height):
            for col in range(env.width):
                x = self.board_x + col * self.block_size
                y = self.board_y + row * self.block_size

                if env.board[row][col] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.colors["cyan"],
                        (x, y, self.block_size, self.block_size),
                    )
                    pygame.draw.rect(
                        self.screen,
                        self.colors["white"],
                        (x, y, self.block_size, self.block_size),
                        1,
                    )
                else:
                    pygame.draw.rect(
                        self.screen,
                        self.colors["black"],
                        (x, y, self.block_size, self.block_size),
                    )
                    pygame.draw.rect(
                        self.screen,
                        self.colors["gray"],
                        (x, y, self.block_size, self.block_size),
                        1,
                    )

        # Draw current piece
        if env.current_piece and not env.game_over:
            shape = env.current_piece["shape"]
            rotation = env.current_piece["rotation"]
            grid = env.pieces[shape][rotation % len(env.pieces[shape])]

            for i, row in enumerate(grid):
                for j, cell in enumerate(row):
                    if cell == "#":
                        x = self.board_x + (env.piece_x + j) * self.block_size
                        y = self.board_y + (env.piece_y + i) * self.block_size

                        if (
                            0 <= env.piece_y + i < env.height
                            and 0 <= env.piece_x + j < env.width
                        ):
                            pygame.draw.rect(
                                self.screen,
                                self.colors["yellow"],
                                (x, y, self.block_size, self.block_size),
                            )
                            pygame.draw.rect(
                                self.screen,
                                self.colors["white"],
                                (x, y, self.block_size, self.block_size),
                                2,
                            )

    def draw_info(self, env, episode, agent_level="Training"):
        info_x = self.board_x + env.width * self.block_size + 50
        info_y = self.board_y

        # Title
        title_text = self.font.render(
            f"RL Tetris - {agent_level}", True, self.colors["white"]
        )
        self.screen.blit(title_text, (info_x, info_y))

        # Stats
        stats = [
            f"Episode: {episode}",
            f"Score: {env.score}",
            f"Lines: {env.lines_cleared}",
            f"Height: {env.get_height()}",
            f"Holes: {env.get_holes()}",
            f"Bumpiness: {env.get_bumpiness()}",
        ]

        for i, stat in enumerate(stats):
            text = self.small_font.render(stat, True, self.colors["white"])
            self.screen.blit(text, (info_x, info_y + 50 + i * 30))

        if env.game_over:
            game_over_text = self.font.render("GAME OVER", True, self.colors["red"])
            self.screen.blit(game_over_text, (info_x, info_y + 250))

    def render(self, env, episode, agent_level="Training"):
        self.screen.fill(self.colors["black"])
        self.draw_board(env)
        self.draw_info(env, episode, agent_level)
        pygame.display.flip()

    def save_screenshot(self, filename):
        pygame.image.save(self.screen, filename)


def create_progress_image(scores_history, episode_ranges, model_names, filename):
    """Create a comprehensive progress visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("RL Tetris Training Progress", fontsize=16, fontweight="bold")

    # Plot 1: Score progression
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for i, (scores, episodes, name) in enumerate(
        zip(scores_history, episode_ranges, model_names)
    ):
        if len(scores) > 0:
            ax1.plot(
                episodes[: len(scores)],
                scores,
                color=colors[i],
                label=f"{name} (Max: {max(scores) if scores else 0})",
                linewidth=2,
                alpha=0.8,
            )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("Score Progression by Training Stage")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Moving average comparison
    window = 50
    for i, (scores, episodes, name) in enumerate(
        zip(scores_history, episode_ranges, model_names)
    ):
        if len(scores) >= window:
            moving_avg = [
                np.mean(scores[j : j + window]) for j in range(len(scores) - window + 1)
            ]
            ax2.plot(
                episodes[window - 1 : len(scores)],
                moving_avg,
                color=colors[i],
                label=f"{name} (Avg)",
                linewidth=3,
            )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Moving Average Score")
    ax2.set_title(f"Performance Trend (Moving Average - {window} episodes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Score distribution
    all_scores = [score for scores in scores_history for score in scores if score > 0]
    if all_scores:
        ax3.hist(all_scores, bins=30, alpha=0.7, color="#96CEB4", edgecolor="black")
        ax3.axvline(
            np.mean(all_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(all_scores):.1f}",
        )
        ax3.axvline(
            np.median(all_scores),
            color="blue",
            linestyle="--",
            label=f"Median: {np.median(all_scores):.1f}",
        )

    ax3.set_xlabel("Score")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Score Distribution Across All Training")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance metrics
    final_scores = [max(scores) if scores else 0 for scores in scores_history]
    avg_scores = [np.mean(scores) if scores else 0 for scores in scores_history]

    x = np.arange(len(model_names))
    width = 0.35

    ax4.bar(
        x - width / 2,
        final_scores,
        width,
        label="Best Score",
        color="#FFD93D",
        edgecolor="black",
        alpha=0.8,
    )
    ax4.bar(
        x + width / 2,
        avg_scores,
        width,
        label="Average Score",
        color="#6BCF7F",
        edgecolor="black",
        alpha=0.8,
    )

    ax4.set_xlabel("Training Stage")
    ax4.set_ylabel("Score")
    ax4.set_title("Performance Comparison by Stage")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text summary
    total_episodes = sum(len(scores) for scores in scores_history)
    best_overall = max(max(scores) if scores else 0 for scores in scores_history)

    fig.text(
        0.02,
        0.02,
        f"Total Episodes: {total_episodes} | Best Score: {best_overall} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def train_agent_stage(
    agent,
    env,
    visualizer,
    episodes,
    stage_name,
    episode_offset=0,
    save_screenshots=True,
    target_update_freq=100,
):
    """Train agent for a specific stage"""
    scores = []
    screenshot_episodes = [
        episodes // 4,
        episodes // 2,
        3 * episodes // 4,
        episodes - 1,
    ]

    print(f"\n=== Training Stage: {stage_name} ===")
    print(f"Episodes: {episodes}, Target Updates Every: {target_update_freq}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 1000

        while not env.game_over and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            # Update visualization every few steps for screenshots
            if save_screenshots and episode in screenshot_episodes and steps % 10 == 0:
                visualizer.render(env, episode_offset + episode + 1, stage_name)

        scores.append(env.score)

        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Progress reporting
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(
                f"Episode {episode_offset + episode + 1}, Avg Score (50): {avg_score:.1f}, "
                f"Epsilon: {agent.epsilon:.3f}, Best: {max(scores)}"
            )

        # Save screenshot at key episodes
        if save_screenshots and episode in screenshot_episodes:
            os.makedirs("training_progress", exist_ok=True)
            screenshot_path = f"training_progress/{stage_name.lower().replace(' ', '_')}_episode_{episode_offset + episode + 1}.png"
            visualizer.render(env, episode_offset + episode + 1, stage_name)
            visualizer.save_screenshot(screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")

    return scores


class TetrisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RL Tetris - Model Selection")
        self.root.geometry("600x400")
        self.root.configure(bg="#2c3e50")

        # Style configuration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.TButton", padding=(10, 5))

        self.setup_ui()
        self.env = None
        self.visualizer = None
        self.agent = None
        self.game_thread = None
        self.running = False

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="🎮 RL Tetris Demo",
            font=("Arial", 20, "bold"),
            fg="#ecf0f1",
            bg="#2c3e50",
        )
        title_label.pack(pady=20)

        # Description
        desc_text = (
            "Choose a trained model to watch the AI play Tetris!\n"
            "Each model represents a different stage of learning."
        )
        desc_label = tk.Label(
            self.root, text=desc_text, font=("Arial", 12), fg="#bdc3c7", bg="#2c3e50"
        )
        desc_label.pack(pady=10)

        # Model selection frame
        model_frame = tk.Frame(self.root, bg="#2c3e50")
        model_frame.pack(pady=20)

        self.model_var = tk.StringVar(value="beginner")

        models = [
            ("🐣 Beginner (Early Learning)", "beginner", "#e74c3c"),
            ("🎯 Intermediate (Getting Better)", "intermediate", "#f39c12"),
            ("🏆 Advanced (Well Trained)", "advanced", "#27ae60"),
        ]

        for text, value, color in models:
            rb = tk.Radiobutton(
                model_frame,
                text=text,
                variable=self.model_var,
                value=value,
                font=("Arial", 12),
                fg="#ecf0f1",
                bg="#2c3e50",
                selectcolor="#34495e",
                indicatoron=0,
                width=30,
                pady=5,
            )
            rb.pack(pady=5)

        # Control buttons
        button_frame = tk.Frame(self.root, bg="#2c3e50")
        button_frame.pack(pady=20)

        self.play_button = tk.Button(
            button_frame,
            text="▶ Start Demo",
            command=self.start_demo,
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
        )
        self.play_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(
            button_frame,
            text="⏹ Stop Demo",
            command=self.stop_demo,
            font=("Arial", 14, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to start demo",
            font=("Arial", 10),
            fg="#95a5a6",
            bg="#2c3e50",
        )
        self.status_label.pack(pady=10)

        # Info text
        info_text = (
            "Models will be loaded from the 'models/' directory.\n"
            "Make sure to run the training first to generate the models!"
        )
        info_label = tk.Label(
            self.root, text=info_text, font=("Arial", 9), fg="#7f8c8d", bg="#2c3e50"
        )
        info_label.pack(pady=(10, 20))

    def start_demo(self):
        if self.running:
            return

        model_name = self.model_var.get()
        model_path = f"models/tetris_agent_{model_name}.pth"

        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model Not Found",
                f"Model file '{model_path}' not found!\n"
                "Please run the training first to generate models.",
            )
            return

        self.running = True
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Loading {model_name} model...")

        # Start game in separate thread
        self.game_thread = threading.Thread(
            target=self.run_demo, args=(model_path, model_name)
        )
        self.game_thread.daemon = True
        self.game_thread.start()

    def stop_demo(self):
        self.running = False
        self.play_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Demo stopped")

        if self.visualizer:
            pygame.quit()

    def run_demo(self, model_path, model_name):
        try:
            # Initialize environment and visualizer
            self.env = TetrisEnv()
            self.visualizer = TetrisVisualizer()

            # Load the trained agent
            state_size = len(self.env.get_state())
            self.agent = DQNAgent(state_size)
            self.agent.load(model_path)
            self.agent.epsilon = 0.0  # No exploration during demo

            self.root.after(
                0,
                lambda: self.status_label.config(text=f"Running {model_name} demo..."),
            )

            episode = 1
            while self.running:
                state = self.env.reset()

                while not self.env.game_over and self.running:
                    # Agent plays
                    action = self.agent.act(state)
                    state, reward, done = self.env.step(action)

                    # Render the game
                    self.visualizer.render(
                        self.env, episode, f"{model_name.title()} AI"
                    )

                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            break

                    time.sleep(0.1)  # Control game speed

                if self.running:
                    time.sleep(2)  # Pause between games
                    episode += 1

        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("Error", f"Demo error: {str(e)}")
            )
        finally:
            self.running = False
            self.root.after(0, lambda: self.play_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.status_label.config(text="Demo finished"))
            if self.visualizer:
                pygame.quit()

    def run(self):
        self.root.mainloop()


def main():
    """Main function to handle training and GUI"""
    print("🎮 RL Tetris Project - Amazing Learning Journey! 🎮")
    print("=" * 60)

    choice = input(
        "\nWhat would you like to do?\n"
        "1. Train new models (full training pipeline)\n"
        "2. Run demo with existing models\n"
        "3. Train + Demo (complete experience)\n"
        "Enter choice (1-3): "
    ).strip()

    if choice in ["1", "3"]:
        print("\n🚀 Starting the incredible training journey!")
        print("This will create 3 different AI models showing learning progress...")

        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("training_progress", exist_ok=True)

        # Initialize environment and visualizer
        env = TetrisEnv()
        visualizer = TetrisVisualizer()

        # Get state size for neural network
        state_size = len(env.get_state())
        print(f"State space size: {state_size}")

        # Training configuration for three stages
        training_stages = [
            {
                "name": "Beginner",
                "episodes": 2,
                "lr": 0.001,
                "target_update": 50,
                "description": "Learning basic moves and piece placement",
            },
            {
                "name": "Intermediate",
                "episodes": 200,
                "lr": 0.0005,
                "target_update": 75,
                "description": "Improving strategy and line clearing",
            },
            {
                "name": "Advanced",
                "episodes": 500,
                "lr": 0.0001,
                "target_update": 100,
                "description": "Mastering advanced techniques",
            },
        ]

        all_scores = []
        episode_ranges = []
        model_names = []
        episode_offset = 0

        # Train each stage
        for i, stage in enumerate(training_stages):
            print(f"\n{'=' * 20} Stage {i + 1}: {stage['name']} {'=' * 20}")
            print(f"📝 {stage['description']}")
            print(f"Episodes: {stage['episodes']}, Learning Rate: {stage['lr']}")

            # Create agent for this stage
            if i == 0:
                agent = DQNAgent(state_size, lr=stage["lr"])
            else:
                # Load previous model and continue training
                prev_model = (
                    f"models/tetris_agent_{training_stages[i - 1]['name'].lower()}.pth"
                )
                agent = DQNAgent(state_size, lr=stage["lr"])
                if os.path.exists(prev_model):
                    agent.load(prev_model)
                    print(f"✅ Loaded previous model: {prev_model}")

            # Train this stage
            scores = train_agent_stage(
                agent,
                env,
                visualizer,
                episodes=stage["episodes"],
                stage_name=stage["name"],
                episode_offset=episode_offset,
                save_screenshots=True,
                target_update_freq=stage["target_update"],
            )

            # Save model
            model_path = f"models/tetris_agent_{stage['name'].lower()}.pth"
            agent.save(model_path)
            print(f"💾 Model saved: {model_path}")

            # Store results
            all_scores.append(scores)
            episode_ranges.append(
                list(range(episode_offset + 1, episode_offset + len(scores) + 1))
            )
            model_names.append(stage["name"])
            episode_offset += len(scores)

            # Print stage summary
            if scores:
                print(
                    f"📊 Stage Summary - Best: {max(scores)}, Avg: {np.mean(scores):.1f}, "
                    f"Final Epsilon: {agent.epsilon:.3f}"
                )

        # Create comprehensive progress visualization
        print("\n📈 Creating amazing progress visualization...")
        progress_image_path = "training_progress/complete_learning_journey.png"
        create_progress_image(
            all_scores, episode_ranges, model_names, progress_image_path
        )
        print(f"✨ Progress visualization saved: {progress_image_path}")

        # Create summary report
        summary_path = "training_progress/training_summary.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_episodes": sum(len(scores) for scores in all_scores),
            "stages": [
                {
                    "name": name,
                    "episodes": len(scores),
                    "best_score": max(scores) if scores else 0,
                    "average_score": float(np.mean(scores)) if scores else 0,
                    "final_scores": scores[-10:] if len(scores) >= 10 else scores,
                }
                for name, scores in zip(model_names, all_scores)
            ],
            "overall_best": max(max(scores) if scores else 0 for scores in all_scores),
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"📋 Training summary saved: {summary_path}")

        pygame.quit()

        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE! 🎉")
        print(f"Total Episodes: {summary_data['total_episodes']}")
        print(f"Best Score Achieved: {summary_data['overall_best']}")
        print("\n📁 Generated Files:")
        print("   • 3 Trained Models (models/)")
        print("   • Training Screenshots (training_progress/)")
        print(
            "   • Complete Progress Chart (training_progress/complete_learning_journey.png)"
        )
        print("   • Training Summary (training_progress/training_summary.json)")
        print("=" * 60)

        if choice == "3":
            print("\n🎮 Launching demo interface...")
            time.sleep(2)

    if choice in ["2", "3"]:
        # Check if models exist
        model_files = [
            "models/tetris_agent_beginner.pth",
            "models/tetris_agent_intermediate.pth",
            "models/tetris_agent_advanced.pth",
        ]

        missing_models = [f for f in model_files if not os.path.exists(f)]
        if missing_models:
            print(f"\n❌ Missing model files: {missing_models}")
            print("Please run training first (option 1) to generate models.")
            return

        print("\n🎮 Starting interactive demo...")
        app = TetrisGUI()
        app.run()

    else:
        print("\n❌ Invalid choice. Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Training interrupted by user. Progress has been saved!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check that all required packages are installed:")
        print("pip install torch pygame matplotlib seaborn numpy pillow")
    finally:
        try:
            pygame.quit()
        except:
            pass
