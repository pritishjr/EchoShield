
        
"""
app/api/websocket.py

The WebSocket endpoint. Deliberately thin: this file's only job is
network framing — accept a connection, loop over incoming binary
frames, hand each one to services/pipeline.py, and send back whatever
comes out. It has no opinion about hashing, caching, models, or
redaction; process_chunk() is the entire interface it depends on.

Error handling philosophy, matching what pipeline.py already
established:
  - A single bad chunk (PipelineError) does not end the connection.
    The client keeps streaming; we send back an error message for
    that one chunk and keep looping.
  - A client disconnecting (WebSocketDisconnect) ends the loop
    cleanly — the expected, normal way a session ends, not a failure.
  - Anything else escaping process_chunk() is, by construction,
    unexpected: pipeline.py's entire job is to translate worker/cache
    failures into PipelineError. So anything else is treated as
    connection-level, not chunk-level — logged loudly and the
    connection is closed, rather than looping on a state we can no
    longer reason about.
"""

from __future__ import annotations

import dataclasses
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.pipeline import PipelineError, process_chunk
logger = logging.getLogger("api.websocket")

router = APIRouter()


@router.websocket("/ws/audio")
async def audio_websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("connection accepted")

    #proper error handling:
    try:
        while True:
            try:
                #input the audio bytes from the websocket.
                audio_bytes = await websocket.receive_bytes()
            except KeyError:
                logger.warning("received non-binary frame — ignoring")
                continue

            #no need of a sequence id (for chunks) since each chunk is processed independently. this guarantee disappears when we want to perform concurrent of multiple audio chunks (for lower per-chunk processing latency).
            try:
                #process the chunks (the pipeline)
                #TranscriptionResult object:
                result = await process_chunk(audio_bytes)
                
            except PipelineError as exc:
                logger.warning("chunk failed: %s", exc)
                
                #send error data back to the client:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            payload = {"type": "transcript", **dataclasses.asdict(result)}
            
            #send the redacted audio back.
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("client disconnected")

    except Exception:
        # Reaching here means something unaccounted-for happened — a bug, not a bad chunk. PipelineError is already assumed before.
        logger.exception("unexpected error on websocket connection — closing")
        try:
            await websocket.close(code=1011)  # 1011: internal error
        except Exception:
            logger.exception("failed to cleanly close websocket after prior error")