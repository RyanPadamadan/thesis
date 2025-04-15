import socket
from scapy.all import *
from scapy.layers.dot11 import RadioTap, Dot11
import time

target_mac = "04:99:bb:d8:6e:2e".lower()
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def packet_handler(pkt):
    if pkt.haslayer(Dot11):
        src = pkt.addr2
        dst = pkt.addr1
        if src and src.lower() == target_mac:
            rssi = pkt.getlayer(RadioTap).dBm_AntSignal
            if rssi != 0:
                timestamp = time.time()
                message = f"{src},{dst},{rssi},{timestamp}"
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                print(f"[SENT] {message}")


def start_sniffer():
    sniff(iface="wlan1", prn=packet_handler, store=0)

if __name__ == "__main__":
    start_sniffer()