import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AESGCMCrypto:
    def __init__(self, key: bytes):
        """
        Initializes the AES-GCM crypto module.
        :param key: 32 bytes for AES-256
        """
        if len(key) not in (16, 24, 32):
            raise ValueError("Key must be 16, 24, or 32 bytes long.")
        self.aesgcm = AESGCM(key)

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypts plaintext and returns nonce + ciphertext + tag.
        The cryptography library's AESGCM encrypt appends the 16-byte tag to the ciphertext automatically.
        We prepend the 12-byte nonce.
        """
        nonce = os.urandom(12)
        ciphertext_with_tag = self.aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext_with_tag

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts the data.
        encrypted_data must be: nonce (12 bytes) + ciphertext + tag (16 bytes)
        """
        if len(encrypted_data) < 28:
            raise ValueError("Encrypted data is too short to contain nonce and tag.")
        nonce = encrypted_data[:12]
        ciphertext_with_tag = encrypted_data[12:]
        try:
            plaintext = self.aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            return plaintext
        except Exception as e:
            # If tag verification fails or data is corrupted
            raise ValueError(f"Decryption failed: {e}")

if __name__ == "__main__":
    from config.settings import AES_KEY
    crypto = AESGCMCrypto(AES_KEY)
    text = b"Hello world, testing offline encryption."
    enc = crypto.encrypt(text)
    dec = crypto.decrypt(enc)
    assert dec == text
    print("AES-GCM working correctly.")
