# scripts/06_predict_single.py
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from tqdm import tqdm
import time

# Reuse your environment
from train import PortfolioEnv  # Make sure this file exists


def _ask_float(prompt: str, default: float):
    """Prompt user for a float; return default on EOF or invalid input."""
    try:
        raw = input(f"{prompt} (default: {default}): ").strip()
        if raw == "":
            return float(default)
        return float(raw)
    except (EOFError, KeyboardInterrupt):
        print(f"\n⚠️  No interactive input available — using default {default}.")
        return float(default)
    except Exception:
        print(f"⚠️  Invalid input — using default {default}.")
        return float(default)


def evaluate_single_stock_profit(
    ticker,
    data_path="./data/stock_data.pkl",
    model_path="./models/ppo_portfolio_trading",
    initial_capital=10000.0,
    planned_years=None,
):
    if not os.path.exists(data_path):
        print("❌ Data not found. Run data fetcher first.")
        return None
    if not os.path.exists(model_path + ".zip"):
        print("❌ Model not found. Train the model first.")
        return None

    # Load data
    start = time.time()
    close_data = pd.read_pickle(data_path)
    print(f"✅ Data loaded in {time.time() - start:.2f}s")

    # Keep at least one year if planned_years not provided; but we'll decide after we know available length
    all_days = len(close_data)
    trading_days_per_year = 252

    if ticker not in close_data.columns:
        print(f"❌ Ticker '{ticker}' not in data. Available: {list(close_data.columns)}")
        return None

    # Load model
    start = time.time()
    model = PPO.load(model_path)
    print(f"✅ Model loaded in {time.time() - start:.2f}s")

    # Create environment with full history (we will slice below to match planned_years)
    # But PortfolioEnv expects a DataFrame for the environment; we will slice latest N rows.
    # Default to using last 1 year if planned_years is None
    default_years = 1
    if planned_years is None:
        planned_years = default_years

    # Compute requested days (rounded to int)
    requested_days = int(planned_years * trading_days_per_year)
    # We need at least window_size + 2 days for the env to step; but we don't know window_size until env is created.
    # We'll create a temporary env with the full dataset to read window_size, then slice accordingly.
    temp_env = DummyVecEnv([lambda: PortfolioEnv(close_data)])
    window_size = temp_env.envs[0].window_size
    temp_env.close()

    min_required_days = window_size + 2  # safe minimum so the RL loop can run at least once

    max_available_days = all_days
    # If user requested more days than available, use max available and compute used_years accordingly
    if requested_days > max_available_days:
        used_days = max_available_days
        used_years = max_available_days / trading_days_per_year
        print(f"⚠️ Requested {requested_days} days (~{planned_years:.2f} years) but only {max_available_days} days available. Using all available data ({used_years:.2f} years).")
    else:
        # but ensure we keep at least min_required_days
        used_days = max(requested_days, min_required_days)
        # If used_days would exceed available, cap it
        used_days = min(used_days, max_available_days)
        used_years = used_days / trading_days_per_year

    # Slice the latest used_days rows from close_data
    close_data_sliced = close_data.tail(used_days).copy()
    print(f"🔎 Using {used_days} trading days (~{used_years:.2f} years) for simulation.")

    # Create environment with the sliced data
    env = DummyVecEnv([lambda: PortfolioEnv(close_data_sliced)])
    obs = env.reset()

    net_worths = [env.envs[0].initial_balance]
    all_weights = []
    stock_returns = []

    # Run full test for the sliced data
    steps = len(close_data_sliced) - env.envs[0].window_size - 1
    if steps <= 0:
        print("❌ Not enough data rows after slicing to run the environment. Try more historical data or smaller window.")
        env.close()
        return None

    for _ in tqdm(range(steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        net_worths.append(infos[0]['net_worth'])
        all_weights.append(infos[0]['weights'])
        if dones[0]:
            break

    # Convert to DataFrame
    # weights_df = pd.DataFrame(all_weights, columns=list(close_data_sliced.columns) + ["CASH"])
    weights_df = pd.DataFrame(
        all_weights,
        columns=list(close_data_sliced.columns) + ["CASH"]
    )[[ticker, "CASH"]]  # <-- solo el ticker pedido + cash

    # Get average allocation to this stock
    avg_allocation = weights_df[ticker].mean()

    # Simulate: allocate exactly agent's weight to the ticker and rest to cash
    initial_capital = float(initial_capital)
    portfolio_value = [initial_capital]
    prices = close_data_sliced[ticker].values
    daily_returns = np.diff(prices, axis=0) / prices[:-1]  # daily % change

    agent_weights_on_ticker = weights_df[ticker].values  # agent’s allocation to this stock

    # Align simulation steps carefully: original code used range(len(daily_returns) - window_size)
    sim_steps = min(len(agent_weights_on_ticker), len(daily_returns) - env.envs[0].window_size + 1)
    if sim_steps <= 0:
        print("❌ Not enough aligned steps to simulate returns. Aborting simulation.")
        env.close()
        return None

    for i in range(sim_steps):
        w = agent_weights_on_ticker[i]  # how much agent allocated to this stock
        # Align the returns with the agent's decision window
        ret_index = i + env.envs[0].window_size - 1
        if ret_index < 0 or ret_index >= len(daily_returns):
            # Safety check: skip if out of range
            continue
        ret = daily_returns[ret_index]
        portfolio_return = w * ret
        portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))

    final_value = portfolio_value[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Annualize based on actual used_years (if used_years == 0 avoid division)
    if used_years > 0:
        annualized_return = ((final_value / initial_capital) ** (1 / used_years) - 1) * 100
    else:
        annualized_return = 0.0

    print(f"\n📊 Profit Forecast for {ticker}")
    print(f"   Avg Allocation by Agent: {avg_allocation * 100:.1f}%")
    print(f"   Estimated Final Value (starting from ${initial_capital:,.2f}): ${final_value:,.2f}")
    print(f"   Total Return: {total_return:.1f}%")
    print(f"   Annualized Return: {annualized_return:.1f}% (based on {used_years:.2f} years simulated)")

    env.close()

    # Dates for alignment
    dates = close_data_sliced.index[-len(portfolio_value):].strftime("%Y-%m-%d").tolist()

    # Prepare JSON-friendly graph data
    # graph_data = {
    #     "dates": dates,
    #     "portfolio_value": portfolio_value,
    #     "allocations": weights_df.to_dict(orient="records"),  # list of daily dicts {TICKER: w, CASH: w}
    # }
    graph_data = {
        "dates": close_data_sliced.index[-len(portfolio_value):].strftime("%Y-%m-%d").tolist(),
        "portfolio_value": portfolio_value,
        "allocations": weights_df.to_dict(orient="records")  # ahora solo trae {ticker, CASH}
    }

    return {
        "ticker": ticker,
        "avg_allocation": round(float(avg_allocation), 3),
        "estimated_final_value": round(final_value, 2),
        "annualized_return": round(annualized_return, 1),
        "years_requested": round(float(planned_years), 2),
        "years_used": round(float(used_years), 2),
        "initial_balance": round(float(initial_capital), 2),
        "graph_data": graph_data,  # <-- new
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate profit for a single stock based on RL agent behavior.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., NVDA, AAPL)")
    parser.add_argument("--initial_balance", type=float, default=None, help="Initial capital to start the simulated portfolio (e.g., 10000)")
    parser.add_argument("--years", type=float, default=None, help="Number of years to simulate (e.g., 1.5)")
    parser.add_argument("--data_path", type=str, default="./data/stock_data.pkl", help="Path to pickled price DataFrame")
    parser.add_argument("--model_path", type=str, default="./models/ppo_portfolio_trading", help="Path prefix to saved PPO model (without .zip)")
    args = parser.parse_args()

    # Interactive prompt if not provided via CLI
    if args.initial_balance is None:
        args.initial_balance = _ask_float("Enter initial balance (USD)", 10000.0)

    if args.years is None:
        args.years = _ask_float("Enter planned years of investment (can be fractional)", 1.0)

    result = evaluate_single_stock_profit(
        args.ticker,
        data_path=args.data_path,
        model_path=args.model_path,
        initial_capital=args.initial_balance,
        planned_years=args.years,
    )
    if result:
        print(
            f"\n✅ Success! You could expect ~${result['estimated_final_value']:,} from ${result['initial_balance']:,} in ~{result['years_used']:.2f} years."
        )
