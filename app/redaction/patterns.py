import re #built in regex library

CC_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b") #matches the 13 or 19 cc digits
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b") #matches the SSN
API_KEY_PATTERN = re.compile(r"\b[A-Za-z0-9_-]{20,}\b") #matches api keys

#pre-compiling the regex functions at a global level so the CPU doesnt have to re-evaluate the regex rules everytime a new chunk arrives. thereby maximizing the throughput.
#purpose: input transcribed text should be properly sanitized (redacted while output) for sanity.

def redaction_regex_patterns(self, text: str) -> str:
    """
    Scans transcribed text for PII (SSN, Credit Cards, API Keys) 
    and replaces them with redaction markers.
    
    Args:
        text (str): The raw transcribed string.
        
    Returns:
        str: The sanitized string safe for storage and UI display.
    """
    
    if not text:
        redacted_text = text #return nothing basically
    
    #applying redaction patterns sequentially: (why sequentially?)
    redacted_text = SSN_PATTERN.sub("[REDACTED_SSN]", redacted_text)
    redacted_text = CC_PATTERN.sub("[REDACTED_CC]", redacted_text)
    redacted_text = API_KEY_PATTERN.sub("[REDACTED_API]", redacted_text)
    
    return redacted_text


    