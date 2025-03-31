from scapy.all import *
from scapy.layers.dot11 import RadioTap, Dot11
from helper import *
import queue
import threading

target_mac = "f0:09:0d:71:6e:7d".lower()

q = queue.Queue()

coords = []
fname = ""
rssi_q = []
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
            time.sleep(0.1)
            continue
        sniff(iface="wlp114s0", prn=packet_handler, store=0, timeout=3)


def point_routine():
    while True:
        inp = input("COMMAND: ")
        if inp == 'DONE':
            # Signal the sniffing thread to stop
            stop_event.set()
            break
        if inp == 'NEW':
            # Pause sniffing temporarily
            pause_event.set()
            coord_inp = input("After moving to new location enter new coordinates (x y): ")
            x, y = map(float, coord_inp.split())
            coords.append((x, y))

            # Drain the queue into a temporary list
            temp = []
            while not q.empty():
                temp.append(q.get())
            rssi_q.append(temp)
            pause_event.clear()

point_routine()
threading.Thread(target=sniff_routine, daemon=True).start()

