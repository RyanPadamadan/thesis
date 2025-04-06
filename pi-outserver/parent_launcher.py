from multiprocessing import Process, Manager
from rssi_api import start_api
from sniffer_process import start_sniffer

if __name__ == "__main__":
    with Manager() as manager:
        shared_log = manager.list()

        # Pass shared_log to both
        p1 = Process(target=start_api, args=(shared_log,))
        p2 = Process(target=start_sniffer)

        p1.start()
        p2.start()
        p1.join()
        p2.join()
