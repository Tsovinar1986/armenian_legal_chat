# OLLAMA Setup

A custom Ollama model that acts as a senior Armenian legal consultant, giving plain-language explanations of Armenian law (Civil, Criminal, Labor Codes, and the Constitution) instead of raw article dumps.

This guide explains how to install Ollama, build the model from the included `Modelfile`, and use it.

---

## 1. Prerequisites

- **macOS, Linux, or Windows** machine
- **Ollama** desktop app or CLI installed
- The base model weights file: `armenia_lawyer_router_gguf.Q4_K_M.gguf` (a quantized GGUF model)
- The `Modelfile` from this project (defines the system prompt and behavior)

### Install Ollama

- **macOS / Windows**: download the app from [ollama.com/download](https://ollama.com/download) and install it like any other application.
- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Verify the install:
```bash
ollama --version
```

---

## 2. Project Files

Make sure both files are together in one folder, for example:

```
Armenian Chat part/
├── armenia_lawyer_router_gguf.Q4_K_M.gguf
└── Modelfile
```

The `Modelfile` currently points to an absolute path:
```
FROM /Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/armenia_lawyer_router_gguf.Q4_K_M.gguf
```

**If you are setting this up on a different machine or user account**, edit the first line of the `Modelfile` so the path matches where the `.gguf` file actually lives on *your* computer. For example:
```
FROM /Users/<your-username>/Desktop/Armenian Chat part/armenia_lawyer_router_gguf.Q4_K_M.gguf
```
Or better, place the `.gguf` file in the same folder as the `Modelfile` and reference it with a relative path:
```
FROM ./armenia_lawyer_router_gguf.Q4_K_M.gguf
```

---

## 3. What's Inside the Modelfile

| Section | Purpose |
|---|---|
| `FROM ...gguf` | Points to the base quantized model weights |
| `SYSTEM "..."` | Instructs the model to behave as a senior Armenian legal consultant, always explaining law in plain language before citing articles, using a structured format (Summary → Practical advice/Pros & Cons → Legal Basis) |
| `PARAMETER stop "Routing hint:"` / `PARAMETER stop "->"` | Prevents the model from reverting to old internal "routing" style output it may have picked up from training data |
| `PARAMETER temperature 0.7` | Controls creativity/randomness (0 = very deterministic, 1 = more varied) |
| `PARAMETER top_p 0.9` | Nucleus sampling — keeps responses focused while allowing some variety |

You can tweak `temperature` and `top_p` later if answers feel too repetitive or too random.

---

## 4. Build the Model

Open a terminal, `cd` into the folder containing the `Modelfile`, then run:

```bash
cd "/path/to/Armenian Chat part"
ollama create armenia-lawyer-router -f Modelfile
```

This registers a new local Ollama model called **`armenia-lawyer-router`**. You'll see Ollama read the weights and apply the system prompt/parameters.

Confirm it was created:
```bash
ollama list
```
You should see `armenia-lawyer-router` in the list.

---

## 5. Run the Model

### Command line
```bash
ollama run armenia-lawyer-router
```
Then type your question directly, for example:
```
What are my rights if my landlord wants to evict me without notice?
```

### Ollama Desktop App
1. Open the Ollama app.
2. In the model selector dropdown (bottom of the chat window), choose **armenia-lawyer-router**.
3. Type your message in "Send a message" and press enter.

---

## 6. Example Usage

**Prompt:**
> My employer hasn't paid my salary for two months. What can I do?

**Expected style of response:**
1. **Summary** — plain-language explanation of employee wage rights in Armenia.
2. **Practical advice / Pros and Cons** — options like filing a complaint, negotiating, or going to court, with trade-offs.
3. **Legal Basis** — relevant articles from the Labor Code of Armenia, explained rather than just listed.

---

## 7. Updating the Model

If you edit the `Modelfile` (e.g., change the system prompt or parameters), rebuild it with:
```bash
ollama create armenia-lawyer-router -f Modelfile
```
Ollama will overwrite the existing model with the new configuration.

To remove the model entirely:
```bash
ollama rm armenia-lawyer-router
```

---

## 8. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `Error: file does not exist` on `ollama create` | The `FROM` path in Modelfile doesn't match your file location | Update the path (see Section 2) |
| Model still outputs "Routing hint:" text | Old cached model version | Rebuild with `ollama create armenia-lawyer-router -f Modelfile` again |
| Responses too repetitive | Temperature too low | Raise `temperature` (e.g., to 0.8–0.9) and rebuild |
| Responses too random/off-topic | Temperature too high | Lower `temperature` (e.g., to 0.5–0.6) and rebuild |

---

## 9. Sharing This Project With Others

To let someone else run this model, share:
1. The `Modelfile`
2. The `.gguf` weights file
3. This README

They just need Ollama installed, then follow Sections 2–5 above.
