# rssi_store.py
import time
from collections import deque
from threading import Lock

MAX_ENTRIES = 1000
_log = deque(maxlen=MAX_ENTRIES)
_lock = Lock()

def add_entry(src, dst, rssi, packet_timestamp=None):
    entry_time = packet_timestamp or time.time()
    entry = {
        "src": src,
        "dst": dst,
        "rssi": rssi,
        "timestamp": entry_time,
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry_time))
    }
    with _lock:
        print("received and appending" + str(entry))
        _log.append(entry)

def get_latest():
    with _lock:
        return _log[-1] if _log else None

def get_all():
    with _lock:
        return list(_log)
