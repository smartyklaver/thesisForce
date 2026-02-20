import time
from UDPsender import UDPSender

sender = UDPSender(udp_ip="127.0.0.1", udp_port=5005)

try:
    print("Sending 0.5 every second... Press Ctrl+C to stop.")
    while True:
        sender.send_data("0.5")
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped sending.")