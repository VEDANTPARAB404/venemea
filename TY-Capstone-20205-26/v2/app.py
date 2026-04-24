import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, session
from mmash_circadian import predict_circadian_score, MODEL_PATH, FEATURE_COLUMNS_PATH, DATA_DIR, load_all_participants
import joblib

app = Flask(__name__)
app.secret_key = "circadian-dashboard-secret-2025"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/research")
def research():
    return render_template("research.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX endpoint for circadian score prediction."""
    try:
        data = request.get_json()
        sleep_hours = float(data["sleep_hours"])
        bed_time_hours = float(data["bed_time_hours"])
        avg_hr = float(data["avg_hr"])
        daily_steps = int(data["daily_steps"])

        # Validation
        if sleep_hours <= 0:
            return jsonify({"ok": False, "error": "Sleep hours must be positive"}), 400
        if bed_time_hours <= 0:
            return jsonify({"ok": False, "error": "Bed time must be positive"}), 400
        if sleep_hours > bed_time_hours:
            return jsonify({"ok": False, "error": "Sleep hours cannot exceed bed time"}), 400
        if avg_hr < 30 or avg_hr > 200:
            return jsonify({"ok": False, "error": "Heart rate must be 30-200 bpm"}), 400
        if daily_steps < 0:
            return jsonify({"ok": False, "error": "Steps cannot be negative"}), 400

        result = predict_circadian_score(
            sleep_hours=sleep_hours,
            bed_time_hours=bed_time_hours,
            avg_hr=avg_hr,
            daily_steps=daily_steps
        )

        # Build breakdown for radar chart
        sleep_eff = result["sleep_efficiency"]
        hrv = result["estimated_hrv"]
        psqi = result["estimated_psqi"]

        # Normalize metrics to 0-100 for radar
        hr_score = max(0, min(100, 100 - (avg_hr - 60) * 2.5))
        steps_score = min(100, (daily_steps / 10000) * 100)
        hrv_score = min(100, (hrv / 100) * 100)
        psqi_score = max(0, 100 - (psqi / 21) * 100)

        return jsonify({
            "ok": True,
            "circadian_score": round(result["circadian_score"], 2),
            "status": result["status"],
            "sleep_efficiency": round(sleep_eff, 1),
            "estimated_hrv": round(hrv, 1),
            "estimated_psqi": round(psqi, 1),
            "recommendations": result["recommendations"],
            "radar": {
                "labels": ["Sleep Efficiency", "Heart Rate", "Activity", "HRV", "Sleep Quality", "Overall"],
                "values": [
                    round(sleep_eff, 1),
                    round(hr_score, 1),
                    round(steps_score, 1),
                    round(hrv_score, 1),
                    round(psqi_score, 1),
                    round(result["circadian_score"], 1)
                ]
            },
            "input_data": {
                "sleep_hours": sleep_hours,
                "bed_time_hours": bed_time_hours,
                "avg_hr": avg_hr,
                "daily_steps": daily_steps
            }
        })
    except KeyError as e:
        return jsonify({"ok": False, "error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/feature-importance")
def api_feature_importance():
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return jsonify({"ok": False, "error": "Model has no feature_importances_"}), 400

        pairs = list(zip(feature_cols, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top10 = pairs[:10]
        return jsonify({
            "ok": True,
            "labels": [p[0] for p in top10],
            "values": [float(p[1]) for p in top10]
        })
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "Model or feature columns not found. Train the model first."}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/prediction-distribution")
def api_prediction_distribution():
    try:
        df = load_all_participants(DATA_DIR)
        if "circadian_score" not in df.columns or df.empty:
            return jsonify({"ok": False, "error": "No circadian scores available."}), 400
        scores = df["circadian_score"].dropna().astype(float).tolist()
        return jsonify({"ok": True, "scores": scores})
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/model-info")
def api_model_info():
    """Return model metadata for the dashboard."""
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
        return jsonify({
            "ok": True,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "n_features": len(feature_cols),
            "oob_score": round(model.oob_score_, 4) if hasattr(model, 'oob_score_') else None,
            "feature_names": feature_cols
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
