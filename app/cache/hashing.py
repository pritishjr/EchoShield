import hashlib

#we use this to convert a window of raw audio signal to its "hashed" form (garbage string value). this is useful since it helps us skip the ai translation phase if we already have the hashed value of a audio sample (eg: 1ms of .wav file)

def audio_hash(audio_bytes: bytes) -> str:
    """
    Generates a fast SHA-256 hash string from raw audio bytes.
    Used as the unique key for Tier 1 (RAM) and Tier 2 (Redis) cache lookups.
    
    Args:
        audio_bytes (bytes): The raw binary chunk of audio (e.g., 1 second of .wav)
        
    Returns:
        str: A 64-character hexadecimal string representing the unique audio fingerprint.
    """
    if not audio_bytes:
        raise ValueError("Cannot hash empty audio information.")
    
    hasher = hashlib.sha256(audio_bytes)
    
    #return the hexadecimal representation of the hashed value.
    return hasher.hexdigest()
