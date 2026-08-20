"""xagent — an LLM harness whose context window is a Python REPL.

The interpreter namespace holds the data; the context window holds only the
narration of what was done to it.
"""

__version__ = "0.1.0"

__all__ = ["Runner", "Provider", "Kernel", "ContextStore", "Compressor",
           "ImageAttachment", "AudioAttachment", "Clip", "Sound", "main"]


def __getattr__(name):  # lazy, so `import xagent` inside a kernel stays cheap
    if name == "Runner":
        from xagent.runner import Runner

        return Runner
    if name == "Provider":
        from xagent.provider import Provider

        return Provider
    if name == "Kernel":
        from xagent.kernel import Kernel

        return Kernel
    if name == "ContextStore":
        from xagent.context import ContextStore

        return ContextStore
    if name == "Compressor":
        from xagent.compress import Compressor

        return Compressor
    if name == "ImageAttachment":
        from xagent.vision import ImageAttachment

        return ImageAttachment
    if name in ("AudioAttachment", "Clip", "Sound"):
        import xagent.audio

        return getattr(xagent.audio, name)
    if name == "main":
        from xagent.cli import main

        return main
    raise AttributeError(name)
