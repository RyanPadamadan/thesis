from scapy.all import *
from scapy.layers.dot11 import RadioTap, Dot11
import signal
"""Capture rssi values from particular mac address and writes to file from particular distance"""
# import signal
target_mac = "f0:09:0d:71:6e:7d".lower()
q = []
distance = 0
fname = ""
def handler(signum, frame):
    print("Saving RSSI values")
    with open(fname, "w") as f:
        for val in q:
            f.write(f"{val}\n")
    wrpcap(fname + ".pcap", packets)
    exit(0)
        


def packet_handler(pkt):
# Only process packets with 802.11 layer
    if pkt.haslayer(Dot11):
        src = pkt[Dot11].addr2
        dst = pkt[Dot11].addr1
        # Match source or destination MAC
        if (src and src.lower() == target_mac):
            #Extract RSSI from RadioTap if available
            radiotap = pkt.getlayer(RadioTap)
            rssi = radiotap.dBm_AntSignal
            if rssi != 0:
                print(f"[RSSI={rssi} dBm] SRC={src} --> DST={dst}")
                q.append(rssi) 
                packets.append(pkt)



distance= input("Distance:")
fname = "data_" + str(distance)
signal.signal(signal.SIGINT, handler)
sniff(iface="wlp114s0", prn=packet_handler, store=0)
