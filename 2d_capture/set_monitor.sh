sudo ip link set wlp114s0 down
sudo iw dev wlp114s0 set type monitor
sudo ip link set wlp114s0 up
sudo iw dev wlp114s0 set channel 8
