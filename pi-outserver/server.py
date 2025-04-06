from scapy.all import *
from scapy.layers.dot11 import RadioTap, Dot11
from rssi_store import add_entry
from rssi_api import start_api
import queue
import threading
import time

# ----------------------------
# Configuration
# ----------------------------
target_mac = "f0:09:0d:71:6d:af".lower()
interface = "wlan1"  # Change to your monitor mode interface name

q = queue.Queue()
stop_event = threading.Event()
pause_event = threading.Event()  # You can toggle this if needed

# ----------------------------
# Sniffer Logic
# ----------------------------
def sniff_routine():
    def packet_handler(pkt):
        if pkt.haslayer(Dot11):
            src = pkt[Dot11].addr2
            dst = pkt[Dot11].addr1
            
            if src and src.lower() == target_mac:
                radiotap = pkt.getlayer(RadioTap)
                rssi = getattr(radiotap, 'dBm_AntSignal', None)
                if rssi is not None and rssi != 0:
                    timestamp = time.time()
                    print(f"[RSSI={rssi} dBm] SRC={src} --> DST={dst}")
                    q.put(rssi)
                    add_entry(src, dst, rssi, timestamp)

    print(f"[SNIFFER] Listening on {interface}...")
    sniff(iface=interface, prn=packet_handler, store=0, timeout=3)


if __name__ == "__main__":
    print("[MAIN] Starting sniffer and API...")
    sniffer_thread = threading.Thread(target=sniff_routine, daemon=True)
    sniffer_thread.start()
    time.sleep(5)
    start_api()
