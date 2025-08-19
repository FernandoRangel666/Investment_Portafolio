# scripts/train.py
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import os

class PortfolioEnv(gym.Env):
    def __init__(self, close_df, initial_balance=10000, max_position_per_stock=1.0):
        super(PortfolioEnv, self).__init__()
        self.close_df = close_df
        self.tickers = close_df.columns.tolist()
        self.n_stocks = len(self.tickers)
        self.initial_balance = initial_balance
        self.max_position_per_stock = max_position_per_stock
        self.window_size = 10

        self.current_step = None
        self.portfolio_value = None
        self.net_worth_history = []
        self.portfolio_weights = None
        self.cost_history = []

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n_stocks + 1,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * self.window_size * self.n_stocks,),  # 20n
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        self.portfolio_weights = np.zeros(self.n_stocks + 1)
        self.cost_history = []
        return self._get_observation(), {}

    def _get_observation(self):
        window = self.close_df.iloc[self.current_step - self.window_size:self.current_step]
        prices = window.values  # (10, n)
        returns = window.pct_change().fillna(0.0).values  # Fill NaN with 0 → now (10, n)
        obs = np.concatenate([prices.flatten(), returns.flatten()])  # → (20n,)
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)

        # Split stock vs cash
        weights = np.clip(action[:-1], 0, self.max_position_per_stock)  # stock weights
        cash_weight = np.clip(action[-1], 0, 1)

        stock_weights = weights  # no extra slicing!

        current_prices = self.close_df.iloc[self.current_step].values
        prev_prices = self.close_df.iloc[self.current_step - 1].values
        returns = (current_prices - prev_prices) / prev_prices

        portfolio_return = np.dot(stock_weights, returns)

        turnover = np.sum(np.abs(stock_weights - self.portfolio_weights[:-1]))
        transaction_cost = 0.001 * turnover
        self.portfolio_value *= (1 - transaction_cost)

        self.net_worth_history.append(self.portfolio_value)
        self.portfolio_weights = np.append(stock_weights, cash_weight)

        risk = np.std(returns)
        # reward = self.portfolio_value / self.initial_balance - 1
        reward = portfolio_return - 0.5 * risk

        self.current_step += 1
        terminated = self.current_step >= len(self.close_df) - 1
        truncated = False

        info = {
            "net_worth": self.net_worth_history[-1],
            "return": portfolio_return,
            "weights": self.portfolio_weights,
            "cost": transaction_cost,
            "turnover": turnover
        }

        return self._get_observation(), reward, terminated, truncated, info


def linear_schedule(initial_value):
    return lambda progress: initial_value * (1 - progress)

def make_env(close_data):
    def _init():
        return PortfolioEnv(close_data)
    return _init

def train():
    print("Loading stock data...")
    close_data = pd.read_pickle("data/stock_data.pkl")

    print("Creating environment...")
    env = PortfolioEnv(close_data)
    env = DummyVecEnv([lambda: env])
    # env = SubprocVecEnv([make_env(close_data) for _ in range(4)])

    model = PPO(
        'MlpPolicy',  # Policy type: uses neural net to learn trading signals from price data
        env,  # Environment: market rules, assets, costs, and risk constraints
        verbose=1,  # Logging: 1 = show training progress (like daily P&L reports)

        # 🔹 learning_rate: "Adaptability to new market regimes"
        # - ↑ Higher (e.g., 0.001): Learns fast, but may overreact to noise (like a novice trader)
        # - ↓ Lower (e.g., 0.0001): Slow, stable learning; trusts past experience more
        # → Range: [0.0001–0.001] → Think: speed of updating trading strategy beliefs
        learning_rate=linear_schedule(3e-4),

        # 🔹 n_steps: "Evaluation window length"
        # - ↑ Longer (e.g., 2048): Uses more data per update → smoother but slower learning
        # - ↓ Shorter (e.g., 512): Faster updates, more responsive to recent performance
        # → Range: [512–4096] → Analogous to review cycle (quarterly vs annual)
        n_steps=64,

        # 🔹 batch_size: "Sample size per strategy review"
        # - ↑ Larger (e.g., 128): Stable updates, but may overfit to recent period
        # - ↓ Smaller (e.g., 32): Noisier updates → more robust, less overconfident
        # → Range: [32–256] → Like basing review on 32 trades instead of 256
        batch_size=32,

        # 🔹 n_epochs: "Number of times to refine strategy per review"
        # - ↑ More (e.g., 20): Learns deeply from same data → better utilization
        # - ↓ Less (e.g., 5): Risks underlearning from valuable feedback
        # → Range: [5–20] → Like re-analyzing last quarter’s trades multiple times
        n_epochs=2,

        # 🔹 gamma: "Patience / time horizon"
        # - ↑ High (0.99): Values long-term gains → patient, buy-and-hold style
        # - ↓ Low (0.90): Focuses on short-term returns → trader mindset
        # → Range: [0.90–0.99] → 0.97 = moderately short-term (1–2 year focus)
        gamma=0.9,

        # 🔹 gae_lambda: "Credit assignment smoothness"
        # - ↑ High (0.98): Spreads credit/ blame over many past actions
        # - ↓ Low (0.80): Blames/rewards only very recent decisions
        # → Range: [0.90–0.99] → 0.95 = balanced: recent actions matter most
        gae_lambda=0.95,

        # 🔹 ent_coef: "Exploration discipline"
        # - ↑ High (0.1+): Tries more random strategies → like testing new sectors
        # - ↓ Low (0.01): Sticks to known patterns → avoids reckless bets
        # → Range: [0.01–0.1] → 0.05 = moderate exploration (safe innovation)
        ent_coef=0.1,

        # 🔹 clip_range: "Policy change limit / risk control"
        # - ↑ High (0.2): Allows big strategy shifts → aggressive adaptation
        # - ↓ Low (0.05): Very conservative updates → avoids overcorrection
        # → Range: [0.05–0.2] → 0.1 = firm risk rule: no drastic pivots
        clip_range=0.2,

        # 🔹 vf_coef: "Value vs action focus"
        # - ↑ High (1.0): Overvalues P&L estimation, may neglect strategy
        # - ↓ Low (0.1): Undervalues risk assessment
        # → Range: [0.5–1.0] → 0.5 = equal weight: good balance
        vf_coef=0.5,

        # 🔹 max_grad_norm: "Risk circuit breaker"
        # - ↑ High (1.0): Allows large updates → volatile strategy changes
        # - ↓ Low (0.1): Very tight control → ultra-conservative learning
        # → Range: [0.1–1.0] → 0.5 = moderate cap: prevents blowups
        max_grad_norm=0.5,

        # 🔹 tensorboard_log: "Performance audit trail"
        # → Logs training metrics (P&L, drawdown, entropy) for monitoring & compliance
        # tensorboard_log="./models/ppo_portfolio/"
        tensorboard_log=None
    )

    print("🚀 Training model...")
    model.learn(total_timesteps=1000)
    model.save("./models/ppo_portfolio_trading")
    print("✅ Model saved to models/ppo_portfolio_trading.zip")

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    train()