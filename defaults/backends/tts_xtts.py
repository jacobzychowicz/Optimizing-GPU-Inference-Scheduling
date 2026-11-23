"""Real voice-cloning text-to-speech on the GPU with Coqui XTTS-v2.

Used by defaults/models/xtts.json. On the GPU machine:

    pip install coqui-tts torch     (a CUDA build of torch)

XTTS-v2 is released under the Coqui Public Model License (non-commercial). Read it
at https://coqui.ai/cpml and set "agree_to_cpml": true in the model's config (or
COQUI_TOS_AGREED=1 in the environment) to confirm you accept it. The model
(about 2 GB) downloads on first use.

config:
  model          Coqui model name (default tts_models/multilingual/multi-dataset/xtts_v2)
  speaker_wav    path on the GPU machine to a 3-15 s reference clip to clone; if not set,
                 the built-in speaker below is used
  speaker        built-in XTTS speaker (default "Ana Florence")
  language       default "en"
  device         default "cuda"
"""

import os
import time


def load(config):
    if config.get("agree_to_cpml"):
        os.environ["COQUI_TOS_AGREED"] = "1"
    if os.environ.get("COQUI_TOS_AGREED") != "1":
        raise RuntimeError(
            "XTTS-v2 needs you to accept the Coqui Public Model License (https://coqui.ai/cpml). "
            'Set "agree_to_cpml": true in the model config, or COQUI_TOS_AGREED=1 on the GPU machine.'
        )
    import torch
    from TTS.api import TTS

    device = config.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch can't see a CUDA GPU on this machine")
    tts = TTS(config.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")).to(device)
    voice = {"language": config.get("language", "en")}
    if config.get("speaker_wav"):
        voice["speaker_wav"] = os.path.expanduser(config["speaker_wav"])
    else:
        voice["speaker"] = config.get("speaker", "Ana Florence")
    return {"torch": torch, "tts": tts, "voice": voice, "device": device}


def _sync(state):
    if state["device"].startswith("cuda"):
        state["torch"].cuda.synchronize()


def warmup(state):
    for text in ("Warming up.", "This is a slightly longer sentence to warm up the voice model."):
        state["tts"].tts(text=text, **state["voice"])
    _sync(state)


def infer(state, job):
    text = (job.get("input") or {}).get("text") or ""
    started = time.perf_counter()
    audio = state["tts"].tts(text=text, **state["voice"])
    _sync(state)
    return {"seconds": time.perf_counter() - started, "samples": len(audio)}


def close(state):
    if state["device"].startswith("cuda"):
        state["torch"].cuda.empty_cache()
