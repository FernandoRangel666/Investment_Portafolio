# scripts/04_test.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import os

from train import PortfolioEnv

def load_tickers():
    with open("./data/tickers.json") as f:
        return json.load(f)

def test():
    print("Loading data and model...")
    close_data = pd.read_pickle("./data/stock_data.pkl")
    model = PPO.load("./models/ppo_portfolio_trading")

    env = DummyVecEnv([lambda: PortfolioEnv(close_data)])
    obs = env.reset()

    net_worths = [env.envs[0].initial_balance]
    all_weights = []

    for _ in range(len(close_data) - env.envs[0].window_size - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        net_worths.append(infos[0]['net_worth'])
        all_weights.append(infos[0]['weights'])
        if dones[0]:
            break

    # Analyze results
    weights_df = pd.DataFrame(
        all_weights,
        columns=env.envs[0].tickers + ["CASH"]  # add the extra slot
    )
    avg_allocation = weights_df.mean().sort_values(ascending=False).head(10)

    # ticker_names = {
    #     'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet (Google)',
    #     'AMZN': 'Amazon', 'META': 'Meta (Facebook)', 'NVDA': 'NVIDIA',
    #     'TSLA': 'Tesla', 'JPM': 'JPMorgan Chase', 'V': 'Visa', 'MA': 'Mastercard'
    # }
    # ticker_names = load_tickers()


    recommendations = []
    for ticker in avg_allocation.index:
        recommendations.append({
            "ticker": ticker,
            "name": ticker,
            # "name": ticker_names.get(ticker, "Unknown"),
            "avg_allocation": round(float(avg_allocation[ticker]) * 100, 2)
        })

    # Final portfolio value
    final_rl_value = round(net_worths[-1], 2)
    final_bh_value = round(calculate_buy_hold(close_data), 2)

    results = {
        "top_10_recommendations": recommendations,
        "profit": {
            "rl_agent_final": final_rl_value,
            "buy_and_hold_final": final_bh_value,
            "outperformance": round(final_rl_value - final_bh_value, 2)
        },
        "as_of": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to JSON
    with open("./data/recommendations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Test complete. Results saved to data/recommendations.json")
    print(f"📈 RL Final Value: ${final_rl_value}")
    print(f"🏦 Buy & Hold: ${final_bh_value}")

def calculate_buy_hold(close_data):
    initial_capital = 10000
    weights = np.ones(len(close_data.columns)) / len(close_data.columns)
    portfolio_value = [initial_capital]
    daily_returns = close_data.pct_change().dropna()

    for _, ret in daily_returns.iterrows():
        portfolio_return = np.dot(weights, ret)
        portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))

    return portfolio_value[-1]

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    test()