import time
from UDPsender import UDPSender

import random

sender = UDPSender(udp_ip="127.0.0.1", udp_port=5005)
change = 0.1

try:
    print("Sending random floats")
    while True:
        change = random.randrange(0,30,1) / 10
        sender.send_data(str(change))
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped sending.")