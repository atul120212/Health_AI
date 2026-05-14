"""
Voice pipeline — LiveKit WebRTC transport + Pipecat orchestration + Sarvam AI.

Architecture per session:
  Browser (LiveKit JS SDK)
    ↕  WebRTC audio  (LiveKit Cloud / self-hosted)
  LiveKitTransport  (Pipecat)
    → SileroVAD  →  SarvamSTTService  →  OpenAILLMService (sarvam-30b)
    ← LiveKitTransport  ←  SarvamTTSService
"""
from .room_manager import room_manager
from .pipeline_builder import start_pipeline

__all__ = ["room_manager", "start_pipeline"]
