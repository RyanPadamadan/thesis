import signal
import os

def handler(sig, frame):
    print("Ctrl + C")
    print(os.getpid())    
    exit(0)

def handle_stop(sig, frame):
    print('Ctrl + Z')


signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGUSR1, handle_stop)
print(os.getpid())
while True:
    pass