class PacketReassembler:
    def __init__(self, missing_placeholder=b" <MISSING> "):
        """
        :param missing_placeholder: The byte sequence to insert when a packet is missing.
        """
        self.missing_placeholder = missing_placeholder

    def reassemble(self, received_packets: list, expected_total: int) -> bytes:
        """
        Reassembles payloads from a list of packets.
        Expected total is the number of packets originally sent (can be inferred or passed out of band).
        If expected_total is unknown, we just use the max received seq_num.
        """
        if not received_packets:
            return b""
            
        # Sort by sequence number
        received_packets.sort(key=lambda p: p.seq_num)
        
        if expected_total is None:
            expected_total = received_packets[-1].seq_num + 1

        reassembled_data = bytearray()
        
        received_seqs = {p.seq_num: p for p in received_packets}
        
        for seq in range(expected_total):
            if seq in received_seqs:
                reassembled_data.extend(received_seqs[seq].payload)
            else:
                reassembled_data.extend(self.missing_placeholder)
                
        return bytes(reassembled_data)
