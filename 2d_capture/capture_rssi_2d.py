from scapy.all import *
from scapy.layers.dot11 import RadioTap, Dot11
from helper import *
import queue
import threading
import sys

target_mac = "f0:09:0d:71:6d:af".lower()

q = queue.Queue()

coords = []
fname = ""
rssi_q = {}
pause_event = threading.Event()
stop_event = threading.Event()

def sniff_routine():
    def packet_handler(pkt):
        if pkt.haslayer(Dot11):
            src = pkt[Dot11].addr2
            dst = pkt[Dot11].addr1
            if src and src.lower() == target_mac:
                radiotap = pkt.getlayer(RadioTap)
                rssi = radiotap.dBm_AntSignal
                if rssi != 0:
                    print(f"[RSSI={rssi} dBm] SRC={src} --> DST={dst}")
                    q.put(rssi)
                    # q.put(pkt)

    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.3)
            continue
        sniff(iface="wlp114s0", prn=packet_handler, store=0, timeout=3)


def point_routine():
    curr = None
    """
    When using first hit new, wait a second, and then move to location before moving to 
    a new coordinate to capture values. 
    """
    while True:
        inp = input("COMMAND: ")
        pause_event.set()
        if curr is not None:
            temp = []
            while not q.empty():
            # Drain the queue into a temporary list
                temp.append(q.get())
            rssi_q[curr] = temp
        if inp == 'done':
            # Signal the sniffing thread to stop
            stop_event.set()
            print(rssi_q)
            sys.exit()
            break
        if inp == 'new':
            # Pause sniffing temporarily
            coord_inp = input("Enter coordinates (x y): ")
            x, y = map(float, coord_inp.split())
            curr = (x, y)
            pause_event.clear()

t1 = threading.Thread(target=point_routine)
t1.start()
threading.Thread(target=sniff_routine, daemon=True).start()
t1.join()
with open("output", "w") as f:
    for key, vals in rssi_q.items():
        line = f"{key} : {' '.join(str(v) for v in vals)}\n"
        f.write(line)

