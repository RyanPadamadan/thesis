from flask import Flask, jsonify, request
from udp_listener import start_udp_thread
from read import save_all_logs

def start_api(log):
    app = Flask(__name__)
    start_udp_thread(log)
    coords_log = []
    device_log = []

    @app.route("/rssi")
    def rssi_all():
        print(f"[FLASK] /rssi hit | log size: {len(log)}")
        return jsonify(list(log))  # cast Manager.list to normal list

    @app.route("/latest")
    def get_latest():
        latest = log[-1] if log else None
        return jsonify(latest) if latest else jsonify({"error": "No data yet"}), 404

    @app.route("/coords", methods=["POST", "GET"])
    def coords():
        if request.method == "POST":
            data = request.json
            print(f"[FLASK] /coords POST: {data}")
            coords_log.append(data)
            return "OK"
        elif request.method == "GET":
            print(f"[FLASK] /coords GET | log size: {len(coords_log)}")
            return jsonify(coords_log)

    @app.route("/device", methods=["POST", "GET"])
    def device():
        if request.method == "POST":
            data = request.json
            print(f"[FLASK] /device POST: {data}")
            device_log.append(data)
            return "OK"
        elif request.method == "GET":
            print(f"[FLASK] /device GET | log size: {len(device_log)}")
            return jsonify(device_log)
        
    @app.route("/reset", methods=["POST"])
    def reset_logs():
        log[:] = [] 
        coords_log.clear()
        device_log.clear()
        print("[FLASK] All logs cleared")
        return jsonify({"status": "cleared all logs"})

    @app.route("/start_experiment", methods=["POST"])
    def start_experiment():
        global experiment_active
        log[:] = []
        coords_log.clear()
        device_log.clear()
        experiment_active = True
        return jsonify({"status": "experiment started"})


    @app.route("/stop_experiment", methods=["POST"])
    def stop_experiment():
        global experiment_active
        experiment_active = False
        exp_folder = save_all_logs()
        return jsonify({"status": "experiment stopped", "saved_to": exp_folder})

    print("[FLASK] API running on http://0.0.0.0:8081")
    app.run(host="0.0.0.0", port=8081, use_reloader=False)