
#first we create our agent as we want-
#orchestration and config with Twilio take the main priority.

import os
import asyncio
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect
)
from loguru import logger

#workers, pipelines, frames, processors:
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import RunnerArguments

#development runner: for session initialisation
from pipecat.runner.types import RunnerArguments
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

#transport: (using LiveKit)
from pipecat.runner.utils import create_transport
from pipecat.runner.livekit import configure
from pipecat.transports.livekit.transport import (
    LiveKitTransport,
    LiveKitParams
)

#VAD:
from pipecat.audio.vad.silero import SileroVADAnalyzer #local VAD
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
    LLMContextAggregatorPair,
    LLMAssistantAggregatorParams,
)
#turn detection: (smart model - default)
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_start import VADUserTurnStartStrategy #interruption handling

#STT:
from pipecat.services.deepgram.stt import DeepgramSTTService

#LLM:
from pipecat.services.openai.llm import OpenAILLMService

#function calling:
from pipecat.services.llm_service import FunctionCallParams

#TTS:
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.utils.text.pattern_pair_aggregator import PatternPairAggregator, MatchAction
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor

#context management:
from pipecat.frames.frames import (
    LLMMessagesUpdateFrame, #replaces the context box.
    LLMMessagesAppendFrame #adds-on to the context box.
)

#telephony:
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport
)
from pipecat.serializers.twilio import TwilioFrameSerializer 

###

async def running_bot(
    websocket: WebSocket,
    stream_sid: str,
    call_sid: str
):
    
    #initializing the serializer: transcodes from the logarithmic u-law to linear PCM.
    serializer = TwilioFrameSerializer(stream_sid, call_sid)
    
    #configuring the transport:
    transport = FastAPIWebsocketTransport(
        websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            serializer=serializer,
        )
    )
    
    