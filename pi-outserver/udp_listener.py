import socket
import threading
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 9999

def udp_listener(log):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[UDP LISTENER] Listening on {UDP_IP}:{UDP_PORT}")
    print("Hi")
    while True:
        data, _ = sock.recvfrom(1024)
        decoded = data.decode()
        try:
            src, dst, rssi, timestamp = decoded.split(",")
            pkt = {
                "src": src,
                "dst": dst,
                "rssi": float(rssi),
                "timestamp": float(timestamp),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(timestamp)))
            }
            log.append(pkt)
            # print("l1: ", log)
        except Exception as e:
            print(f"[ERROR] Malformed packet: {decoded} ({e})")


def start_udp_thread(log):
    threading.Thread(target=udp_listener, args=(log,), daemon=True).start()
