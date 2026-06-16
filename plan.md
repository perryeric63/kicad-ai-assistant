# KiCad AI Addon Plan (iGPU Vulkan)

## 1. System & KiCad Update

### Current State
| Item | Value |
|------|-------|
| KiCad | ~~9.0.9~~ → **10.0.3** (2026-05-20) ✅ |
| iGPU | AMD Radeon Graphics (RADV RENOIR) — Vega 7nm, 7 CUs |
| Vulkan | 1.4 via Mesa RADV 26.0.3 |
| Python API | `pcbnew` module with `ActionPlugin` base class |
| PyTorch | 2.12.0+cpu (no CUDA) |
| RAM | UMA ~45 GB/s DDR4 (shared CPU/GPU — same bottleneck as Blender project) |

### Steps to update KiCad (manual, needs sudo) — done
```bash
sudo add-apt-repository --remove ppa:kicad/kicad-9.0-releases
sudo add-apt-repository ppa:kicad/kicad-10.0-releases
sudo apt update && sudo apt full-upgrade
```

## 2. Architecture

```
plugins/                          # ~/.local/share/kicad/10.0/scripting/plugins/
├── ai_assistant_plugin.py       # 🔌 Entry point — KiCad discovers this
├── ai_assistant/                # 📦 Backend package
│   ├── __init__.py              # ActionPlugin class (used by wrapper)
│   ├── panel.py                 # wxPython chat UI
│   ├── llm_backend.py           # llama.cpp + Vulkan GPU offload
│   ├── component_db.py          # PCB context builder
│   ├── pcb_assist.py            # PCB operations (routing, placement, DRC)
│   ├── sch_assist.py            # Schematic operations (netlist, symbols)
│   ├── footprint_gen.py         # Footprint generation from datasheet
│   ├── routing_ops.py           # Auto-routing hints / track manipulation
│   └── vulkan_ext/              # Custom Vulkan compute ops (if needed)
│       ├── vulkan_ops.cpp       # C++ extension (mirrors Blender pattern)
│       └── setup.py
│
└── (models stored at ~/AI_Models/)
```

**Note:** KiCad's plugin loader scans for `.py` files in the plugins root directory,
not subdirectories. `ai_assistant_plugin.py` is the flat entry point that KiCad
discovers; it delegates to the `ai_assistant/` package for implementation.

## 3. LLM Backend (Vulkan Accelerated)

Reuse the exact same pattern from `Blender_AIassistant/addon/llm_backend.py`:

| Component | Reference | KiCad Adaptation |
|-----------|-----------|-----------------|
| Model format | llama.cpp GGUF | Same — code-compatible |
| GPU offload | `n_gpu_layers=24` via Vulkan | Same — RADV works identically |
| Threading | Async queue + worker thread | Same — KiCAD needs non-blocking UI |
| System prompt | Blender scene context | PCB/schematic context instead |

**Context builder** (`component_db.py`) feeds the LLM:
- Current open schematic sheets
- Netlist connectivity summary
- Component list with values
- PCB layer stackup
- DRC errors
- Selected footprints/symbols

## 4. KiCad Python API — Key Bindings

```python
import pcbnew

# Board access
board = pcbnew.GetBoard()
board = pcbnew.LoadBoard("/path/to/board.kicad_pcb")

# Footprints
for fp in board.GetFootprints():
    fp.GetReference(), fp.GetValue(), fp.GetPosition()
    fp.GetBoundingBox(), fp.GetOrientationDegrees()

# Tracks / vias
for track in board.GetTracks():
    track.GetStart(), track.GetEnd(), track.GetWidth(), track.GetLayer()

# Schematic (via eeschema, limited — may need IPC/kiplot)
# Use kicad-cli for schematic queries in async mode:
#   kicad-cli sch export python-bom project.kicad_sch

# Action plugin base
class MyPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "AI Assistant"
        self.category = "AI"
        self.description = "AI-powered PCB design assistant"
        self.show_toolbar_button = True

    def Run(self):
        # Launch UI
        pass
```

## 5. UI Design

**Option A: Embedded Panel** (preferred — follows Blender addon pattern)
- Register as `ActionPlugin` with a toolbar button
- Opens a wxPython panel docked in KiCad's main window
- Chat input + response display + code execution buttons

**Option B: Standalone Window**
- Separate floating window with wxPython
- Less integration but simpler to implement

## 6. KiCad-Specific Features

| Feature | Description | Reference |
|---------|-------------|-----------|
| **Schematic Assistant** | Answer questions about schematic, suggest connections | LLM + netlist context |
| **Auto-component Placement** | Suggest component positions based on connectivity | `pcbnew.FOOTPRINT.SetPosition()` |
| **Track Routing Help** | Generate track routing suggestions | DRC rules + topology analysis |
| **Footprint Generator** | Generate footprints from text descriptions | `pcbnew.FootprintLoad/Save` |
| **DRC Explainer** | Explain DRC errors and suggest fixes | LLM context from DRC report |
| **BOM Assistant** | Help with component selection and alternatives | LLM + component DB |
| **Via Stitching** | Auto-generate via stitching for RF/EMI | Vulkan parallel via placement |
| **Differential Pair** | Auto-route / suggest differential pair tracks | Geometry + GPU compute |
| **Layout from Image** | Trace board outline from reference image (like FlexBV) | Vulkan image processing |

## 7. Hardware Considerations (iGPU Vulkan)

Same constraints as `Blender_AIassistant/plan.md`:

| Op | CPU (MKL) | iGPU Vulkan | Verdict |
|----|-----------|-------------|---------|
| LLM inference (GGUF) | ~8 tok/s | ~12 tok/s (partial offload) | ✅ Use `n_gpu_layers` |
| Dense GEMM | 319 GFLOPS | ~38 GFLOPS | ❌ Keep on CPU |
| Sparse ops (routing graphs) | N/A | 2.2x vs CPU | ✅ Vulkan wins |
| Parallel track checks | N/A | Good for batch DRC | ✅ Vulkan compute |

**Key Insight (from Blender project):** UMA iGPU only helps for sparse/irregular ops. Keep dense LLM matmuls on CPU (via MKL), offload attention layers to Vulkan via `llama.cpp`. For PCB operations (batch DRC, via placement), Vulkan compute shaders on sparse data give real speedup.

## 8. Implementation Phases

### Phase 0: Foundation ✅ (Completed 2026-06-07)

| Task | Status | Test Result |
|------|--------|-------------|
| Update KiCad to 10.0.3 | ✅ | `kicad` → 10.0.3~ubuntu26.04.1 |
| Plugin directory structure | ✅ | `~/.local/share/kicad/10.0/scripting/plugins/ai_assistant/` |
| Port `llm_backend.py` from Blender | ✅ | System prompt → KiCad/PCB, `_build_board_context()` via `pcbnew` |
| Download GGUF model | ✅ | `qwen2.5-1.5b-q4_k_m.gguf` (1.1 GB) |
| `load_model()` + GPU offload | ✅ | `n_gpu_layers=24`, Qwen2.5 1.5B fully on Vulkan |
| `ask("hello")` | ✅ | Returned PCB explanation in ~20s |
| Async queue processing | ✅ | `submit_request()` → worker thread → `poll_response()` |
| KiCad plugin discovery | ✅ | `AiAssistantPlugin` found as `ActionPlugin` subclass |
| wxPython UI | ✅ | `AiAssistantFrame` creates, `show_window()` ready |
| `component_db` context builder | ✅ | Gracefully handles no-board-open; full API tested with `pcbnew.CreateEmptyBoard`/`LoadBoard`/`SaveBoard` |
| PCB ops (SaveBoard/LoadBoard) | ✅ | Board created, footprint added, saved, reloaded, verified |

#### Test Log
```python
# Test 1: Plugin registration
[DISCOVERY] Plugin class: AiAssistantPlugin
  Name: AI Assistant
  Category: AI
  Description: AI-powered PCB design assistant with local LLM (Vulkan)
  Toolbar: True

# Test 2: Model inference
>>> llm_backend.ask("What is a PCB? Reply in one sentence.")
'A PCB, or Printed Circuit Board, is a flat sheet of insulating material
 with conductive pathways (tracks) attached to its surface.'

# Test 3: GPU offload
>>> llm.model_params.n_gpu_layers
24

# Test 4: Board ops
>>> board = pcbnew.CreateEmptyBoard()
>>> fp = pcbnew.FOOTPRINT(board)
>>> fp.SetReference('R1')
>>> board.Add(fp)
>>> pcbnew.SaveBoard('/tmp/test.kicad_pcb', board)
>>> board2 = pcbnew.LoadBoard('/tmp/test.kicad_pcb')
>>> len(board2.GetFootprints())
1
```

#### Files Created
```
~/.local/share/kicad/10.0/scripting/plugins/
├── ai_assistant_plugin.py       # 🔌 Entry point — KiCad discovers this
└── ai_assistant/                # 📦 Backend package
    ├── __init__.py              # ActionPlugin class (used by wrapper)
    ├── llm_backend.py           # llama.cpp + Vulkan (ported from Blender)
    ├── panel.py                 # wxPython chat UI
    └── component_db.py          # PCB context builder

~/AI_Models/ (shared with Blender_AIassistant)
└── qwen2.5-1.5b-q4_k_m.gguf  # 1.1 GB, 24 layers on Vulkan
```

### Phase 1: Basic UI (Week 2)
- [ ] Register `ActionPlugin` in `__init__.py`
- [ ] Build wxPython chat panel (mirror `chat_ui.py` pattern)
- [ ] Call `llm_backend.ask()` on user input
- [ ] Display response in chat area
- [ ] Extract and display Python code blocks separately

### Phase 2: Context & PCB Ops (Week 3)
- [ ] Build `component_db.py` — scrape current board state → text context
- [ ] Add system prompt: teach LLM about KiCad Python API
- [ ] Implement code execution: `exec()` footprint/track placement scripts
- [ ] Add safety: confirm before modifying board

### Phase 3: Schematic Assistant (Week 4)
- [ ] Export netlist from current schematic → context
- [ ] Read symbol library → LLM can suggest parts
- [ ] Help with wiring suggestions

### Phase 4: GPU Compute Ops (Week 5)
- [ ] Port `vulkan_ops.cpp` from Blender project
- [ ] Add compute shaders for:
  - Parallel DRC checks (clearance, via density)
  - Batch via stitching / ground plane via arrays
  - Track topology optimization (maze routing on sparse grid)
- [ ] Benchmark vs CPU-only

### Phase 5: Polish (Week 6)
- [ ] Undo/redo for AI-generated changes
- [ ] Progress reporting for long operations
- [ ] Model auto-download (from HuggingFace)
- [ ] Plugin packaging as `.zip`

## 9. Files

| File | Purpose |
|------|---------|
| `plan.md` | This document |
| `ai_assistant/__init__.py` | Plugin registration, metadata |
| `ai_assistant/panel.py` | UI panel (wxPython) |
| `ai_assistant/llm_backend.py` | llama.cpp + Vulkan LLM (port from Blender) |
| `ai_assistant/pcb_assist.py` | PCB read/write operations |
| `ai_assistant/sch_assist.py` | Schematic read/export |
| `ai_assistant/component_db.py` | Build context from board state |
| `ai_assistant/footprint_gen.py` | Footprint creation helpers |
| `ai_assistant/routing_ops.py` | Route/track suggestion algorithms |
| `ai_assistant/vulkan_ext/vulkan_ops.cpp` | C++ Vulkan compute extension (port) |
| `ai_assistant/vulkan_ext/setup.py` | Build with pybind11 + torch |
| `ai_assistant/vulkan_ext/shaders/drc_check.comp` | Parallel DRC check shader |
| `ai_assistant/vulkan_ext/shaders/via_stitch.comp` | Via stitching shader |
| `~/AI_Models/*.gguf` | Local LLM models (shared) |

## 10. Verification

```bash
# Test plugin imports
python3 -c "
import sys
sys.path.insert(0, '/home/ericperry/.local/share/kicad/10.0/scripting/plugins')
from ai_assistant.llm_backend import load_model, ask
load_model()
print(ask('Describe the KiCad ActionPlugin API'))
"

# Test plugin discovery (simulates KiCad scan)
python3 -c "
import sys, importlib, inspect
import pcbnew
sys.path.insert(0, '/home/ericperry/.local/share/kicad/10.0/scripting/plugins')
mod = importlib.import_module('ai_assistant.__init__')
for name, obj in inspect.getmembers(mod):
    if inspect.isclass(obj) and issubclass(obj, pcbnew.ActionPlugin) and obj is not pcbnew.ActionPlugin:
        inst = obj()
        inst.defaults()
        print(f'Found: {inst.name} — {inst.description}')
"

# Inside KiCad: Tools → External Plugins → AI Assistant
```

## 11. Phase 1 Next Steps

1. Open KiCad → verify plugin appears in Tools → External Plugins
2. Click AI Assistant → panel should open with chat window
3. Type a PCB question → model should respond with context from current board
4. Add code execution: extract Python code blocks, confirm, `exec()`
