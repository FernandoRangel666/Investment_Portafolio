# app.py  -- merged predict + recommendations
import os
import importlib
from flask import Flask, request, jsonify

app = Flask(__name__)

# === Config (overridable via ENV) ===
DATA_PATH = os.environ.get("DATA_PATH", "./data/stock_data.pkl")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "./models/ppo_portfolio_trading")  # no .zip
MODEL_ZIP = MODEL_PREFIX + ".zip"
TEST_RESULTS_PATH = os.environ.get("TEST_RESULTS_PATH", "./data/recommendation.json")

# --- Dynamic import of your evaluation function ---
def _load_evaluator():
    candidates = ["predict_single", "scripts.predict_single", "API_predict", "predict"]  # try a few
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            fn = getattr(mod, "evaluate_single_stock_profit", None)
            if callable(fn):
                return fn
        except Exception:
            continue
    raise ImportError(
        "Could not import evaluate_single_stock_profit. Make sure predict_single.py (or scripts/predict_single.py) "
        "is on the PYTHONPATH and defines evaluate_single_stock_profit."
    )

try:
    evaluate_single_stock_profit = _load_evaluator()
    import_error_on_import = None
except ImportError as e:
    evaluate_single_stock_profit = None
    import_error_on_import = str(e)


def _parse_inputs(req):
    if req.method == "GET":
        ticker = req.args.get("ticker")
        initial = req.args.get("initial_balance") or req.args.get("initial")
        years = req.args.get("years")
    else:
        data = req.get_json(silent=True) or {}
        ticker = data.get("ticker") or data.get("symbol")
        initial = data.get("initial_balance") or data.get("initial")
        years = data.get("years") or data.get("planned_years")

    if not ticker:
        return None, None, None, "Ticker is required."

    initial_balance = None
    planned_years = None

    if initial is not None:
        try:
            initial_balance = float(initial)
            if initial_balance <= 0:
                return None, None, None, "initial_balance must be > 0."
        except (ValueError, TypeError):
            return None, None, None, "initial_balance must be a number."

    if years is not None:
        try:
            planned_years = float(years)
            if planned_years <= 0:
                return None, None, None, "years must be > 0."
        except (ValueError, TypeError):
            return None, None, None, "years must be a number."

    return ticker.upper(), initial_balance, planned_years, None


# --- Predict route ---
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if import_error_on_import:
        return jsonify({"error": "server_misconfigured", "message": import_error_on_import}), 500
    if evaluate_single_stock_profit is None:
        return jsonify({"error": "server_misconfigured", "message": "Evaluation function unavailable."}), 500

    ticker, initial_balance, planned_years, parse_err = _parse_inputs(request)
    if parse_err:
        return jsonify({"error": "bad_request", "message": parse_err}), 400

    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "data_unavailable", "message": f"Stock data not found at {DATA_PATH}"}), 503
    if not os.path.exists(MODEL_ZIP):
        return jsonify({"error": "model_unavailable", "message": f"Model not found at {MODEL_ZIP}"}), 503

    try:
        result = evaluate_single_stock_profit(
            ticker=ticker,
            data_path=DATA_PATH,
            model_path=MODEL_PREFIX,
            initial_capital=initial_balance if initial_balance is not None else 10000.0,
            planned_years=planned_years
        )
        if result is None:
            return jsonify({"error": "prediction_failed", "message": "Prediction returned None."}), 400

        result["_meta"] = {"data_path": DATA_PATH, "model_path": MODEL_ZIP}
        return jsonify(result), 200

    except Exception as e:
        app.logger.exception("Error during prediction")
        return jsonify({"error": "internal_error", "message": str(e)}), 500


# --- Recommendations route (from your API_ten_recmm) ---
@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    results_file = TEST_RESULTS_PATH
    if not os.path.exists(results_file):
        return jsonify({"error": "No results available. Run the test script first."}), 404
    import json
    with open(results_file, "r") as f:
        data = json.load(f)
    return jsonify(data)


# --- Health and a small in-browser tester ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK", "message": "Server is running"}), 200


@app.route("/test", methods=["GET"])
def test_page():
    html = """
<!doctype html><html><head><meta charset="utf-8"><title>API Test</title></head><body>
<h2>API Test</h2>
<ul>
  <li><a href="/health" target="_blank">/health</a></li>
  <li><a href="/recommendations" target="_blank">/recommendations</a></li>
  <li>/predict (use the sample fetch in the page)</li>
</ul>
<pre id="out">loading...</pre>
<script>
async function tryPredict(){
  try {
    const resp = await fetch('/predict?ticker=AAPL');
    const json = await resp.json();
    document.getElementById('out').textContent = JSON.stringify(json, null, 2);
  } catch(e){
    document.getElementById('out').textContent = 'Error: ' + e;
  }
}
tryPredict();
</script></body></html>
"""
    return html, 200, {"Content-Type": "text/html"}


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    print(f"Starting merged API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
