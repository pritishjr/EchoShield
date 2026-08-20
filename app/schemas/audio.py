
"""
app/schemas/audio.py

Only job is to make sure and decide:
- whether the shape of the response sent from the server to the client is precise. to see what's allowed to pass over from the network.

The issue we are trying to solve:
- TranscriptionResult data object is rather internal (jumps across the worker boundary with variable shape as needs be.)
- Therefore, there is no divide between what the client recieves and the internal structure when manually edited, and so any change would automatically appear in the client-facing JSON whether desirable or not.

TranscriptResponse changes this.
- returns a message for a successful transcript or an error. basically relays what information the client can expect (Literal[...]).
- Pydantic's discriminated unions use that field to pick the right model
automatically.
- matters most when the input is more than just bytes (raw binary)- specifying a format/sampling rate, .etc. 
- uses the same shape already set. 

this file serves (or channels) the required output via the TranscriptResponse. we can also change it explicitly.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from app.workers.transcribe import TranscriptionResult

#schema1
class TranscriptResponse(BaseModel):

    #field selection could be arbitrary based on the client's expectation rather than internalized selection.
    
    type: Literal["transcript"] = "transcript"
    redacted_text: str
    redaction_count: int
    language: str | None
    duration_ms: float
    is_silent: bool

    @classmethod
    def from_result(cls, result: TranscriptionResult) -> "TranscriptResponse":
        """
        TranscriptionResult -> wire-format translation. The entire point of this
        class existing is that the internal and external shapes are
        ALLOWED to diverge; an automatic unpack would quietly undo
        that protection the moment the two shapes actually do drift.
        """
        return cls(
            redacted_text=result.redacted_text,
            language=result.language,
            duration_ms=result.duration_ms,
            is_silent=result.is_silent,
        )

#the connection must stay open even after a failed processing.
#schema2
class ErrorResponse(BaseModel):
    """
    Sent for a chunk that failed processing (see PipelineError in
    services/pipeline.py). The connection stays open after this — it's
    a per-chunk error, not a connection-level one.
    """

    type: Literal["error"] = "error"
    message: str  

    pass


AudioStreamMessage = Annotated[
    Union[TranscriptResponse, ErrorResponse], # a websocket response will eaither conform to a TranscriptResponse schmea or ErrorRespomns schema
    Field(discriminator="type"), 
]
