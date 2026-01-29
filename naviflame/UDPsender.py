import socket

class UDPSender:
    def __init__(self, udp_ip="127.0.0.1", udp_port=5005):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_data(self, message):
        """Sends a message via UDP."""
        self.sock.sendto(message.encode('utf-8'), (self.udp_ip, self.udp_port))
