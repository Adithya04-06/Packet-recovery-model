from crypto.aes_gcm import AESGCMCrypto
from packet.packet_builder import PacketBuilder
from packet.packet_loss_simulator import PacketLossSimulator
from packet.packet_reassembler import PacketReassembler

class CommunicationPipeline:
    def __init__(self, aes_key: bytes, packet_payload_size: int = 16):
        self.crypto = AESGCMCrypto(aes_key)
        self.builder = PacketBuilder(max_payload_size=packet_payload_size)
        self.reassembler = PacketReassembler(missing_placeholder=b" <MISSING> ")

    def transmit(self, text: str, drop_probability: float = 0.0):
        # 1. Text to bytes
        data_bytes = text.encode('utf-8')
        
        # 2. Chunk data first (so we can encrypt chunks independently to survive packet loss)
        # Note: In a real system, splitting utf-8 arbitrarily might split a multibyte character.
        # We will decode with errors='ignore' on the receiver side.
        chunks = []
        for i in range(0, len(data_bytes), self.builder.max_payload_size):
            chunks.append(data_bytes[i:i + self.builder.max_payload_size])
            
        # 3. Encrypt chunks
        encrypted_chunks = [self.crypto.encrypt(chunk) for chunk in chunks]
        
        # 4. Build packets (the encrypted chunk is the payload of the packet)
        # Wait, the encrypted chunk will be larger than max_payload_size due to nonce+tag.
        # So we just pass the encrypted chunks to builder, treating each as one packet payload.
        packets = []
        for i, enc_chunk in enumerate(encrypted_chunks):
            from packet.packet_builder import Packet
            packets.append(Packet(seq_num=i, payload=enc_chunk))
            
        # 5. Simulate Wireless Channel
        simulator = PacketLossSimulator(drop_probability=drop_probability)
        received_packets = simulator.simulate(packets)
        
        # 6. Receiver: Sequence validation & Payload extraction
        # The reassembler needs to put <MISSING> if seq is gap.
        # But wait, we need to decrypt *before* reassembling the final bytes!
        
        total_expected = len(packets)
        received_seqs = {p.seq_num: p for p in received_packets}
        
        final_bytes = bytearray()
        for seq in range(total_expected):
            if seq in received_seqs:
                encrypted_payload = received_seqs[seq].payload
                try:
                    decrypted = self.crypto.decrypt(encrypted_payload)
                    final_bytes.extend(decrypted)
                except Exception:
                    # If decryption fails (e.g., corruption), treat as missing
                    final_bytes.extend(b" <MISSING> ")
            else:
                final_bytes.extend(b" <MISSING> ")
                
        # 7. Decode
        received_text = final_bytes.decode('utf-8', errors='ignore')
        
        # Clean up multiple missing tokens into one if desired, or keep them to signify multiple lost packets
        import re
        received_text = re.sub(r'( <MISSING> )+', ' <MISSING> ', received_text).strip()
        if received_text.startswith('<MISSING> '):
            received_text = received_text[10:]
        if received_text.endswith(' <MISSING>'):
            received_text = received_text[:-10]
            
        return {
            "original_text": text,
            "corrupted_text": received_text,
            "total_packets": len(packets),
            "received_packets": len(received_packets),
            "lost_packets": len(packets) - len(received_packets)
        }

if __name__ == "__main__":
    from config.settings import AES_KEY
    pipeline = CommunicationPipeline(AES_KEY, packet_payload_size=8)
    res = pipeline.transmit("I have a new blue pen for my college", drop_probability=0.2)
    print("Original:", res['original_text'])
    print("Received:", res['corrupted_text'])
    print("Packets:", res['total_packets'], "Lost:", res['lost_packets'])
