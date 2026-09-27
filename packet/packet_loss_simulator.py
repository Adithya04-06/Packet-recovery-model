import random

class PacketLossSimulator:
    def __init__(self, drop_probability: float = 0.0):
        """
        :param drop_probability: The probability (0.0 to 1.0) of dropping a packet.
        """
        self.drop_probability = drop_probability

    def simulate(self, packets: list) -> list:
        """
        Takes a list of packets and randomly drops some based on drop_probability.
        Returns the received packets.
        """
        received_packets = []
        for pkt in packets:
            if random.random() >= self.drop_probability:
                received_packets.append(pkt)
        return received_packets
