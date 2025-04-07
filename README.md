## Localisation of IOT devices using Mixed Reality and Wireless Networks
This project uses various techniques taking in data from IOT devices and using the spatial information collected using the Quest 3s to be able to localise IOT devices
---
### Server Setup
* Sniffer Process: Runs Scapy and sends captured packets to **server** process via UDP socket
* Server Process: Captures UDP packets on one thread, and other thread runs a Flask server to allow send packet data over http
### Current Softwares
- scapy 2.5
- matplotlib
- python 3.11
- Flask
