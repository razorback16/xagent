"""Local speech recognition: NVIDIA Nemotron 3.5 ASR, streaming, 8-bit.

The model is `nvidia/nemotron-3.5-asr-streaming-0.6b` -- a cache-aware
FastConformer-RNNT that transcribes 40 language-locales and was built to consume
audio in chunks rather than in one shot. That is why it is here: a clip on disk
and a speaker still playing are the same problem to it, so both reach the model
through one code path and the text comes back while the audio is still arriving.

Weights are loaded in 8 bits. Two routes exist and both are genuinely int8:

  transformers  600M params quantized on load (bitsandbytes LLM.int8, or torchao
                int8 weight-only when bitsandbytes is absent). Streams text token
                by token through `TextIteratorStreamer`, so this is the default.
  gguf          the `q8_0` file NVIDIA publishes in the same repo, run by the
                NeMo-Speech.cpp `nemo-speech` binary. No torch, no CUDA -- but it
                wants a finished file, so a live capture is transcribed when it
                ends rather than as it plays.

Nothing here is imported until something asks to transcribe: the harness must not
pay a torch import to answer a question about a text file.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xagent.audio import SAMPLE_RATE, Clip, to_wav

MODEL_ID = os.environ.get("XAGENT_ASR_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
GGUF_FILE = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"

# Right-context, in 80ms frames, from the set the model was trained to switch
# between: {0, 1, 3, 6, 13} -> chunks of 80ms .. 1.12s. 6 is the middle of the
# latency/accuracy curve at ~0.56s, which is the right default for a harness that
# wants to read the words as they are spoken and still be accurate.
DEFAULT_LOOKAHEAD = int(os.environ.get("XAGENT_ASR_LOOKAHEAD", "6"))
DEFAULT_QUANT = os.environ.get("XAGENT_ASR_QUANT", "int8")
DEFAULT_LANG = os.environ.get("XAGENT_ASR_LANG", "auto")

# In `auto` mode the model appends the locale it detected after the terminal
# punctuation, as a special token. Read it, then keep it out of the transcript.
LANG_TAG = re.compile(r"<([a-z]{2}-[A-Za-z]{2,4})>")
OTHER_SPECIAL = re.compile(r"<\|?[a-z_]+\|?>", re.IGNORECASE)

INSTALL_HINT = (
    "install the audio extra to transcribe locally:\n"
    "    uv sync --extra asr          # torch + transformers>=5.13 + bitsandbytes\n"
    "or install NeMo-Speech.cpp and let the q8_0 GGUF do it on the CPU:\n"
    "    https://github.com/NVIDIA/NeMo-Speech.cpp"
)


class AsrError(RuntimeError):
    """Transcription could not run, or ran and failed."""


@dataclass
class Heard:
    """What the model made of some audio."""

    text: str = ""
    languages: list[str] = field(default_factory=list)
    seconds: float = 0.0
    backend: str = ""
    model: str = MODEL_ID

    @property
    def words(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        head = self.text.strip()[:120]
        tag = f" [{', '.join(self.languages)}]" if self.languages else ""
        return f"<Heard {self.words} words{tag} via {self.backend}> {head}"


# ------------------------------------------------------------------- feeding


class SampleFeed:
    """A growing buffer of float samples, addressed absolutely.

    The model's chunker walks the audio by sample index and steps back by half an
    FFT window at every chunk, so it needs to read behind where it just was. A
    plain queue cannot answer that; this can, while still blocking when the audio
    it is asked for has not been played yet. A file arrives already closed, which
    makes the file case a live case that never has to wait.
    """

    def __init__(self, rate: int = SAMPLE_RATE):
        self.rate = rate
        self._buf: list[float] = []
        self._base = 0                      # absolute index of self._buf[0]
        self._closed = False
        self._cv = threading.Condition()

    def push(self, samples: Sequence[float]) -> None:
        if not len(samples):
            return
        with self._cv:
            if self._closed:
                raise AsrError("cannot push into a closed feed")
            self._buf.extend(samples)
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def total(self) -> int:
        with self._cv:
            return self._base + len(self._buf)

    def get(self, start: int, end: int, timeout: float | None = None):
        """Samples [start, end), blocking until they exist. None once they cannot.

        Returns a short block only at the very end of a closed stream, and None
        when the stream is closed and start is already past its end -- which is
        the generator's signal to stop rather than an error.
        """
        start = max(0, start)
        with self._cv:
            while True:
                have = self._base + len(self._buf)
                if have >= end or self._closed:
                    if start >= have:
                        return None if self._closed else []
                    lo = start - self._base
                    hi = min(len(self._buf), end - self._base)
                    if lo < 0:
                        raise AsrError(
                            f"sample {start} was already released (buffer starts at "
                            f"{self._base}); the reader fell behind the stream"
                        )
                    return self._buf[lo:hi]
                if not self._cv.wait(timeout):
                    raise AsrError(f"timed out waiting for audio samples {start}..{end}")

    def release(self, before: int) -> None:
        """Drop samples nobody can ask for again, so a long capture is bounded."""
        with self._cv:
            cut = before - self._base
            if cut > 0:
                del self._buf[:cut]
                self._base += cut


def feed_from(samples: Sequence[float], rate: int = SAMPLE_RATE) -> SampleFeed:
    feed = SampleFeed(rate)
    feed.push(samples)
    feed.close()
    return feed


def _clean(piece: str) -> tuple[str, list[str]]:
    """Split a streamed piece into displayable text and the locales it carried."""
    found = LANG_TAG.findall(piece)
    text = LANG_TAG.sub("", piece)
    text = OTHER_SPECIAL.sub("", text)
    return text, found


# ------------------------------------------------------- transformers, int8


def _quant_config(quant: str):
    """An 8-bit config, by whichever quantizer is installed."""
    if quant in ("off", "none", "", "fp16", "bf16"):
        return None
    if quant not in ("int8", "8bit", "8", "int4", "4bit", "4"):
        raise AsrError(f"unknown quantization {quant!r}; use int8, int4 or off")
    four = quant in ("int4", "4bit", "4")
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        return (BitsAndBytesConfig(load_in_4bit=True) if four
                else BitsAndBytesConfig(load_in_8bit=True))
    except ImportError:
        pass
    if four:
        raise AsrError("4-bit needs bitsandbytes: pip install bitsandbytes")
    try:
        from transformers import TorchAoConfig

        # Weight-only int8: the same 8-bit weights, without a CUDA-only kernel,
        # so this is also the route on a machine with no bitsandbytes wheel.
        return TorchAoConfig("int8_weight_only")
    except ImportError as e:
        raise AsrError(
            "8-bit loading needs bitsandbytes or torchao "
            "(pip install bitsandbytes, or pass quant='off' to run in fp16)"
        ) from e


class TransformersAsr:
    """Nemotron through 🤗 Transformers, quantized on load, streaming out."""

    name = "transformers"

    def __init__(self, model_id: str = MODEL_ID, *, quant: str = DEFAULT_QUANT,
                 device: str | None = None, lookahead: int = DEFAULT_LOOKAHEAD):
        self.model_id = model_id
        self.quant = quant
        self.device = device or os.environ.get("XAGENT_ASR_DEVICE") or "auto"
        self.lookahead = lookahead
        self.model = None
        self.processor = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        try:
            import transformers  # noqa: F401

            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self):
        if self.model is not None:
            return self.model, self.processor
        with self._lock:
            if self.model is not None:
                return self.model, self.processor
            try:
                from transformers import AutoModelForRNNT, AutoProcessor
            except ImportError as e:
                raise AsrError(f"transformers is not installed. {INSTALL_HINT}") from e
            kwargs = {"device_map": self.device}
            config = _quant_config(self.quant)
            if config is not None:
                kwargs["quantization_config"] = config
            try:
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForRNNT.from_pretrained(self.model_id, **kwargs)
            except Exception as e:
                raise AsrError(f"could not load {self.model_id}: {type(e).__name__}: {e}") from e
            processor.set_num_lookahead_tokens(self.lookahead)
            self.model, self.processor = model, processor
            return model, processor

    @property
    def latency_ms(self) -> float:
        _, processor = self.load()
        return float(getattr(processor, "streaming_latency_ms", 0.0))

    def stream(self, feed: SampleFeed, *, lang: str = DEFAULT_LANG, on_text=None) -> Heard:
        from transformers import TextIteratorStreamer

        model, proc = self.load()
        rate = proc.feature_extractor.sampling_rate
        first_n = proc.num_samples_first_audio_chunk
        head = feed.get(0, first_n)
        if not head:
            return Heard(seconds=0.0, backend=self.name, model=self.model_id)

        first = proc(_as_array(head), sampling_rate=rate, is_streaming=True,
                     is_first_audio_chunk=True, language=lang, return_tensors="pt")
        first = first.to(model.device, dtype=model.dtype)

        def features():
            yield first.input_features[:, : proc.num_mel_frames_first_audio_chunk, :]
            mel = proc.num_mel_frames_first_audio_chunk
            hop = proc.feature_extractor.hop_length
            n_fft = proc.feature_extractor.n_fft
            start = mel * hop - n_fft // 2
            while True:
                end = start + proc.num_samples_per_audio_chunk
                block = feed.get(start, end)
                # A short block is a partial final chunk: the encoder is
                # cache-aware and expects whole chunks, so the tail below one
                # chunk is dropped rather than padded into invented audio.
                if block is None or len(block) < (end - start):
                    return
                chunk = proc(_as_array(block), sampling_rate=rate, is_streaming=True,
                             is_first_audio_chunk=False, language=lang,
                             return_tensors="pt").to(model.device, dtype=model.dtype)
                yield chunk.input_features
                feed.release(start)
                mel += proc.num_mel_frames_per_audio_chunk
                start = mel * hop - n_fft // 2

        streamer = TextIteratorStreamer(proc.tokenizer, skip_special_tokens=False)
        failure: list[BaseException] = []

        def run():
            try:
                model.generate(**{**first, "input_features": features(),
                                  "streamer": streamer})
            except BaseException as e:                  # surfaced on the main thread
                failure.append(e)
                streamer.end()

        worker = threading.Thread(target=run, name="xagent-asr", daemon=True)
        worker.start()

        pieces, langs = [], []
        for piece in streamer:
            text, found = _clean(piece)
            for locale in found:
                if locale not in langs:
                    langs.append(locale)
            if text:
                pieces.append(text)
                if on_text:
                    on_text(text)
        worker.join()
        if failure:
            raise AsrError(f"transcription failed: {type(failure[0]).__name__}: {failure[0]}")
        return Heard(text="".join(pieces).strip(), languages=langs,
                     seconds=feed.total / feed.rate, backend=self.name, model=self.model_id)


def _as_array(samples):
    try:
        import numpy as np

        return np.asarray(samples, dtype=np.float32)
    except ImportError:
        return list(samples)


# ------------------------------------------------------------- gguf, q8_0


class NemoSpeechAsr:
    """The published q8_0 GGUF, run by the NeMo-Speech.cpp CLI."""

    name = "gguf"

    def __init__(self, model_id: str = MODEL_ID, *, gguf: str | None = None, **_):
        self.model_id = model_id
        self.gguf = gguf or os.environ.get("XAGENT_ASR_GGUF")
        self.quant = "q8_0"

    @staticmethod
    def available() -> bool:
        return shutil.which("nemo-speech") is not None

    def weights(self) -> str:
        if self.gguf and Path(self.gguf).is_file():
            return self.gguf
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise AsrError(
                f"set XAGENT_ASR_GGUF to a local {GGUF_FILE}, or install "
                f"huggingface_hub so it can be fetched"
            ) from e
        self.gguf = hf_hub_download(self.model_id, GGUF_FILE)
        return self.gguf

    def stream(self, feed: SampleFeed, *, lang: str = DEFAULT_LANG, on_text=None) -> Heard:
        """Drain the feed, then transcribe the file it made.

        The binary takes a path, so live audio is transcribed once it stops rather
        than while it plays. Streaming here is of the *output*: lines are relayed
        as the process prints them.
        """
        while feed.get(feed.total, feed.total + 1) is not None:
            pass
        samples = feed.get(0, feed.total) or []
        binary = shutil.which("nemo-speech")
        if binary is None:
            raise AsrError(f"nemo-speech is not on PATH. {INSTALL_HINT}")
        weights = self.weights()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fh:
            fh.write(to_wav(samples, feed.rate))
            fh.flush()
            cmd = [binary, "transcribe", fh.name, "--model", weights,
                   "--language", lang]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1)
            pieces, langs = [], []
            for line in proc.stdout:
                text, found = _clean(line)
                for locale in found:
                    if locale not in langs:
                        langs.append(locale)
                if text.strip():
                    pieces.append(text)
                    if on_text:
                        on_text(text)
            code = proc.wait()
            err = (proc.stderr.read() or "").strip()
        if code != 0:
            raise AsrError(f"nemo-speech exited {code}: {err[-400:]}")
        return Heard(text="".join(pieces).strip(), languages=langs,
                     seconds=len(samples) / feed.rate, backend=self.name,
                     model=f"{self.model_id}:{GGUF_FILE}")


BACKENDS = {"transformers": TransformersAsr, "gguf": NemoSpeechAsr}

_CACHE: dict[tuple, object] = {}


def pick_backend() -> str | None:
    """Which backend would run, or None if neither can."""
    choice = os.environ.get("XAGENT_ASR_BACKEND")
    if choice:
        if choice not in BACKENDS:
            raise AsrError(f"unknown XAGENT_ASR_BACKEND={choice!r}; "
                           f"use one of {', '.join(BACKENDS)}")
        return choice
    for name, cls in BACKENDS.items():
        if cls.available():
            return name
    return None


def transcriber(backend: str | None = None, *, model: str = MODEL_ID,
                quant: str = DEFAULT_QUANT, lookahead: int = DEFAULT_LOOKAHEAD,
                device: str | None = None):
    """A loaded transcriber, kept between calls.

    600M parameters take seconds to quantize and place, and a session that
    transcribes twice should pay that once.
    """
    name = backend or pick_backend()
    if name is None:
        raise AsrError(f"no local ASR backend is installed. {INSTALL_HINT}")
    key = (name, model, quant, lookahead, device)
    if key not in _CACHE:
        _CACHE[key] = BACKENDS[name](model, quant=quant, lookahead=lookahead,
                                     device=device)
    return _CACHE[key]


def warm(**kwargs) -> str:
    """Load the weights now, so the first transcription is not also the download."""
    asr = transcriber(**kwargs)
    if hasattr(asr, "load"):
        asr.load()
    return f"{asr.name}/{getattr(asr, 'quant', '?')}"


def transcribe_clip(clip: Clip, *, lang: str = DEFAULT_LANG, on_text=None,
                    backend: str | None = None, **kwargs) -> Heard:
    """Transcribe decoded audio that is already complete."""
    asr = transcriber(backend, **kwargs)
    return asr.stream(feed_from(clip.samples, clip.rate), lang=lang, on_text=on_text)


def transcribe_feed(feed: SampleFeed, *, lang: str = DEFAULT_LANG, on_text=None,
                    backend: str | None = None, **kwargs) -> Heard:
    """Transcribe audio that is still arriving."""
    asr = transcriber(backend, **kwargs)
    return asr.stream(feed, lang=lang, on_text=on_text)


def pump(chunks: Iterator[bytes], feed: SampleFeed, *, on_chunk=None) -> threading.Thread:
    """Push PCM16 chunks into a feed on a thread, closing it when they run out."""
    from xagent.audio import from_pcm16

    def run():
        try:
            for raw in chunks:
                samples = from_pcm16(raw)
                feed.push(samples)
                if on_chunk:
                    on_chunk(samples)
        finally:
            feed.close()

    thread = threading.Thread(target=run, name="xagent-audio-pump", daemon=True)
    thread.start()
    return thread


__all__ = [
    "AsrError", "Heard", "SampleFeed", "MODEL_ID", "GGUF_FILE", "BACKENDS",
    "pick_backend", "transcriber", "warm", "transcribe_clip", "transcribe_feed",
    "feed_from", "pump", "queue",
]
