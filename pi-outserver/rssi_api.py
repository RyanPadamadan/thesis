from flask import Flask, jsonify
from udp_listener import start_udp_thread

def start_api(log):
    app = Flask(__name__)
    start_udp_thread(log)

    @app.route("/rssi")
    def rssi_all():
        print(f"[FLASK] /rssi hit | log size: {len(log)}")
        return jsonify(list(log))  # cast Manager.list to normal list

    @app.route("/latest")
    def get_latest():
        latest = log[-1] if log else None
        return jsonify(latest) if latest else jsonify({"error": "No data yet"}), 404

    print("[FLASK] API running on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, use_reloader=False)
