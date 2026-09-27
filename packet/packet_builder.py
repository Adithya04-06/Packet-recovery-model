import struct
import binascii

# Conceptually a packet structure:
# HEADER (magic bytes) : 2 bytes
# SEQ_NUM              : 4 bytes (unsigned int)
# LENGTH               : 2 bytes (unsigned short, payload length)
# PAYLOAD              : N bytes
# CRC32                : 4 bytes

HEADER_MAGIC = b'\xAA\xBB'

class Packet:
    def __init__(self, seq_num: int, payload: bytes):
        self.seq_num = seq_num
        self.payload = payload
        self.length = len(payload)

    def serialize(self) -> bytes:
        # Pack header, seq_num, length
        header = struct.pack("!2sI H", HEADER_MAGIC, self.seq_num, self.length)
        data = header + self.payload
        crc = binascii.crc32(data) & 0xFFFFFFFF
        crc_bytes = struct.pack("!I", crc)
        return data + crc_bytes

    @classmethod
    def deserialize(cls, data: bytes):
        if len(data) < 12: # 2+4+2 + crc(4)
            raise ValueError("Data too short to be a valid packet.")
        
        crc_received = struct.unpack("!I", data[-4:])[0]
        packet_data = data[:-4]
        crc_calculated = binascii.crc32(packet_data) & 0xFFFFFFFF
        
        if crc_received != crc_calculated:
            raise ValueError("CRC mismatch - packet corrupted.")
            
        magic, seq_num, length = struct.unpack("!2sI H", packet_data[:8])
        if magic != HEADER_MAGIC:
            raise ValueError("Invalid packet header magic.")
            
        payload = packet_data[8:]
        if len(payload) != length:
            raise ValueError("Payload length mismatch.")
            
        return cls(seq_num, payload)

class PacketBuilder:
    def __init__(self, max_payload_size=16):
        self.max_payload_size = max_payload_size

    def build_packets(self, data: bytes) -> list:
        """
        Splits data into multiple packets with sequential numbers.
        """
        packets = []
        seq_num = 0
        for i in range(0, len(data), self.max_payload_size):
            chunk = data[i:i + self.max_payload_size]
            pkt = Packet(seq_num, chunk)
            packets.append(pkt)
            seq_num += 1
        return packets
