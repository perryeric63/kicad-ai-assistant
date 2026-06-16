# KiCad AI Assistant

Local LLM-powered design assistant for KiCad 10. Works in PCB Editor and Footprint Editor.

## Project Structure

```
KiCad_Aiassistant/
├── ai_core.py                 # Shared engine: UI, LLM backend, streaming, code sandbox
├── ai_assistant_plugin.py     # PCB/Footprint prompts, context builders, tool detection
├── test_hello.py              # Minimal test plugin
├── deploy.sh                  # Copies .py files to KiCad plugins directory
├── plan.md                    # Design notes
└── README.md
```

## Architecture

```
ai_core.py (1120 lines)
  └─ AiAssistantFrame, LLM worker, streaming, token mgmt, code execution
  
ai_assistant_plugin.py (403 lines) ─> imports ai_core
  └─ PCB & Footprint prompts, context builders, tool detection, ActionPlugin
```

The assistant knows about all 9 KiCad tools:
- **PCB Editor** & **Footprint Editor**: full code execution
- **Eeschema**, **Symbol Editor**, **GerbView**, **Drawing Sheet Editor**, **PCB Calculator**, **Bitmap2Component**, **PCM**: knowledge mode

## Deployment

KiCad loads plugins from `~/.local/share/kicad/10.0/scripting/plugins/`.  
This workspace is the dev copy. Deploy with:

```bash
./deploy.sh
```

Or manually:

```bash
cp *.py ~/.local/share/kicad/10.0/scripting/plugins/
```

After deploying, restart KiCad or Tools → External Plugins → Refresh to reload.

## Requirements

- KiCad 10
- `llama-cpp-python` (`pip install llama-cpp-python`)
- GGUF model files in `~/AI_Models/`
