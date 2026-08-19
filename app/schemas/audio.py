
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
        The one place TranscriptionResult -> wire-format translation
        happens. Deliberately explicit, field by field — not
        cls(**dataclasses.asdict(result)). The entire point of this
        class existing is that the internal and external shapes are
        ALLOWED to diverge; an automatic unpack would quietly undo
        that protection the moment the two shapes actually do drift.
        """
        return cls(
            redacted_text=result.redacted_text,
            redaction_count=result.redaction_count,
            language=result.language,
            duration_ms=result.duration_ms,
            is_silent=result.is_silent,
        )


class ErrorResponse(BaseModel):
    """
    Sent for a chunk that failed processing (see PipelineError in
    services/pipeline.py). The connection stays open after this — it's
    a per-chunk error, not a connection-level one.

    `message` is TRUSTED to already be client-safe by the time it
    reaches this model — this class does not sanitize it. That
    responsibility lives in services/pipeline.py, which constructs
    PipelineError with descriptive-but-safe text (e.g. "could not
    decode audio chunk: ...") rather than forwarding a raw internal
    exception string. That's an assumption this file leans on, not one
    it enforces: if pipeline.py's error branches are ever changed to
    include raw internals "for debuggability," this model would
    faithfully — and wrongly — ship that straight to the client.
    """

    type: Literal["error"] = "error"
    message: str  

    pass

# Every message this endpoint can send, tagged for automatic
# discrimination on the `type` field. Used today mainly as a single
# source of truth for "what can this endpoint emit" — useful for
# tests, and for generating client-side types later. It becomes
# directly load-bearing (i.e. actually used to validate/parse,
# not just document) the moment this file gains an incoming-message
# model that needs the same discrimination on a read path.
AudioStreamMessage = Annotated[
    Union[TranscriptResponse, ErrorResponse],
    Field(discriminator="type"),
]
