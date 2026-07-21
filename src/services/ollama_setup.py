"""Checks that the Ollama models this app needs are actually present before
anything tries to use them, instead of failing deep inside a request with a
raw ollama._types.ResponseError 404. Works the same on any OS Ollama runs on
(Windows/macOS/Linux) since it only talks to the local Ollama daemon's REST
API — the exact failure this addresses was reported happening identically on
both this machine and a teammate's Windows one.

Called once at startup by both entry points (api.py's startup event and
src/main.py's main()).
"""

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "armenia-lawyer-router"


def ensure_ollama_models():
    """Auto-pulls EMBEDDING_MODEL if it's missing — a real, publicly
    published model, so this always works the same way regardless of OS.

    LLM_MODEL can't be handled the same way: it's custom to this project,
    built locally via `ollama create -f Modelfile` from a GGUF file that
    isn't checked into git (multi-GB, doesn't belong in the repo), so
    there's nothing to auto-pull it from. If it's missing, this only prints
    the exact command to fix it rather than silently doing nothing or
    pretending it can be automated the same way.
    """
    import ollama

    try:
        local_models = {m.model.split(":")[0] for m in ollama.list().models}
    except Exception as exc:
        print(f"⚠️ Could not reach Ollama to check installed models: {exc}")
        print("   Is Ollama installed and running? See https://ollama.com/")
        return

    if EMBEDDING_MODEL not in local_models:
        print(f"📥 Pulling missing Ollama model '{EMBEDDING_MODEL}' (one-time, ~270MB)...")
        try:
            ollama.pull(EMBEDDING_MODEL)
            print(f"✅ Pulled '{EMBEDDING_MODEL}'")
        except Exception as exc:
            print(f"❌ Failed to auto-pull '{EMBEDDING_MODEL}': {exc}")
            print(f"   Run manually: ollama pull {EMBEDDING_MODEL}")

    if LLM_MODEL not in local_models:
        print(f"⚠️ Ollama model '{LLM_MODEL}' is missing. It's custom to this project "
              f"(built from this repo's Modelfile + a local GGUF file), not something "
              f"that can be downloaded automatically. From the repo root, run:")
        print(f"   ollama create {LLM_MODEL} -f Modelfile")
