"""Shared AI Assistant core — UI, LLM backend, streaming, token management, code execution.

Used by tool-specific plugins (PCB, Footprint, Schematic, etc.).
Set SYSTEM_PROMPT and CONTEXT_BUILDER via init() before creating the frame.
"""

import pcbnew
import wx
import threading
import re
import queue
import traceback
import os
from pathlib import Path
from collections import deque
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────

MODEL_DIR = Path.home() / "AI_Models"
FALLBACK_MODEL = MODEL_DIR / "qwen2.5-1.5b-q4_k_m.gguf"
if not FALLBACK_MODEL.exists():
    FALLBACK_MODEL = MODEL_DIR / "tinyllama-1.1b-q4_k_m.gguf"
MODEL_PATH = str(FALLBACK_MODEL)
GPU_LAYERS = 0
BACKEND_OPTIONS = {"CPU": 0, "Vulkan": 24}
TEMPERATURE = 0.3
TEMP_OPTIONS = {"Precision (0.1)": 0.1, "Code (0.3)": 0.3, "Mixed (0.5)": 0.5, "Creative (0.7)": 0.7}
CTX_SIZE = 4096
MAX_OUTPUT = 1024

SYSTEM_PROMPT = ""
CONTEXT_BUILDER = None
SYSPROMPT_TOKENS = 0

def init(system_prompt, context_builder):
    global SYSTEM_PROMPT, CONTEXT_BUILDER, SYSPROMPT_TOKENS
    SYSTEM_PROMPT = system_prompt
    CONTEXT_BUILDER = context_builder
    SYSPROMPT_TOKENS = estimate_tokens(SYSTEM_PROMPT)

def get_context():
    return CONTEXT_BUILDER() if CONTEXT_BUILDER else "No board is currently open."

# ── Token estimation & context trimming ───────────────────────────────────

def estimate_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 4)

def trim_context(context, user_prompt, max_ctx_tokens):
    available = max_ctx_tokens - estimate_tokens(user_prompt)
    if available <= 0:
        return "Context too large to fit."
    ctx_tokens = estimate_tokens(context)
    if ctx_tokens <= available:
        return context
    lines = context.split('\n')
    conv_start = None
    for i, line in enumerate(lines):
        if line.startswith('Recent conversation:'):
            conv_start = i
            break
    if conv_start is not None:
        for j in range(conv_start + 1, len(lines)):
            if ':' in lines[j] and len(lines[j]) > 200:
                prefix, rest = lines[j].split(':', 1)
                lines[j] = prefix + ':' + rest[:120]
    trimmed = '\n'.join(lines)
    if estimate_tokens(trimmed) <= available:
        return trimmed
    fp_section_start = None
    for i, line in enumerate(lines):
        if 'Footprints (max' in line or (line.startswith('  ') and 'at (' in line and 'mm' in line):
            if fp_section_start is None and 'Footprints' in line:
                fp_section_start = i
    if fp_section_start is not None:
        kept = []
        for i, line in enumerate(lines):
            if i == fp_section_start:
                kept.append(line.replace('max 25', 'max 10'))
            elif i > fp_section_start and line.startswith('  ') and 'at (' in line and 'mm' in line:
                if len(kept) - fp_section_start <= 11:
                    kept.append(line)
            else:
                kept.append(line)
        trimmed = '\n'.join(kept)
        if estimate_tokens(trimmed) <= available:
            return trimmed
    if conv_start is not None:
        trimmed_lines = lines[:conv_start]
        while trimmed_lines and not trimmed_lines[-1].strip():
            trimmed_lines.pop()
        trimmed = '\n'.join(trimmed_lines)
        if estimate_tokens(trimmed) <= available:
            return trimmed
    summary_lines = []
    for line in lines:
        if any(kw in line for kw in ['Board:', 'Board thickness:', 'Board outline',
                                       'Existing references:', 'Last action:',
                                       'Free space', 'Selected:', 'Nets:']):
            summary_lines.append(line)
    summary_lines.append(f"Footprints: {lines.count(' at (')} total")
    trimmed = '\n'.join(summary_lines)
    return trimmed if estimate_tokens(trimmed) <= available else trimmed[:available * 4]

# ── Model Discovery ───────────────────────────────────────────────────────

def scan_gguf_models():
    models = []
    if MODEL_DIR.is_dir():
        for f in sorted(MODEL_DIR.iterdir()):
            if f.suffix.lower() == ".gguf":
                size_gb = f.stat().st_size / 1e9
                name = f.stem
                models.append((str(f), name, size_gb))
    return models

def format_model_label(path, name, size_gb):
    return f"{name}  ({size_gb:.1f} GB)"

def model_basename(path):
    return Path(path).stem

# ── LLM Backend (thread-safe, lazy load) ──────────────────────────────────

_llm = None
_llm_lock = threading.Lock()
_worker_thread = None
_request_queue = queue.Queue()
_response_ready = threading.Event()
_current_response = None
_current_error = None
_request_active = False
_shutdown = threading.Event()
_model_loaded = False
_model_path = MODEL_PATH
_last_token_estimate = 0
_last_action = ""

def load_model(path=None, gpu_layers=None):
    global _llm, _model_loaded, _model_path
    if gpu_layers is None:
        gpu_layers = GPU_LAYERS
    with _llm_lock:
        if _model_loaded:
            return True
        if path:
            _model_path = path
        from llama_cpp import Llama
        try:
            _llm = Llama(
                model_path=_model_path,
                n_gpu_layers=gpu_layers,
                n_ctx=CTX_SIZE,
                n_threads=4,
                verbose=False,
            )
            _model_loaded = True
            return True
        except Exception:
            _model_loaded = False
            raise

def unload_model():
    global _llm, _model_loaded
    with _llm_lock:
        _llm = None
        _model_loaded = False

def is_model_loaded():
    return _model_loaded

def _worker_loop():
    global _current_response, _current_error, _request_active, _last_token_estimate
    while not _shutdown.is_set():
        try:
            item = _request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        prompt, context, callback, stream_cb = item
        _request_active = True
        try:
            with _llm_lock:
                if not _model_loaded:
                    load_model()
                max_ctx = CTX_SIZE - SYSPROMPT_TOKENS - MAX_OUTPUT
                if context:
                    context = trim_context(context, prompt, max_ctx)
                    full_prompt = SYSTEM_PROMPT + f"\n\nCurrent context:\n{context}"
                else:
                    full_prompt = SYSTEM_PROMPT
                full_prompt += f"\n\nUser: {prompt}\n\nAssistant:"
                _last_token_estimate = estimate_tokens(full_prompt)
                if stream_cb:
                    full_text = ""
                    for chunk in _llm(
                        full_prompt,
                        max_tokens=MAX_OUTPUT,
                        temperature=TEMPERATURE,
                        top_p=0.9,
                        stop=["User:"],
                        echo=False,
                        stream=True,
                    ):
                        token = chunk['choices'][0]['text']
                        full_text += token
                        stream_cb(token, False)
                    stream_cb("", True)
                    response_text = re.sub(r'\nUser:.*$', '', full_text, flags=re.DOTALL).strip()
                    _current_response = response_text
                else:
                    result = _llm(
                        full_prompt,
                        max_tokens=MAX_OUTPUT,
                        temperature=TEMPERATURE,
                        top_p=0.9,
                        stop=["User:"],
                        echo=False,
                    )
                    response_text = result["choices"][0]["text"].strip()
                    response_text = re.sub(r'\nUser:.*$', '', response_text, flags=re.DOTALL).strip()
                _current_response = response_text
            _current_error = None
            if callback:
                callback(response_text, None)
        except Exception as e:
            _current_response = None
            _current_error = str(e)
            if callback:
                callback(None, str(e))
        _request_active = False
        _response_ready.set()
    _request_active = False

def submit_request(prompt, context="", callback=None, stream_callback=None):
    _response_ready.clear()
    _request_queue.put((prompt, context, callback, stream_callback))

def poll_response():
    if _response_ready.is_set():
        return _current_response, _current_error
    return None, None

def is_busy():
    return _request_active

def start_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _shutdown.clear()
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()

def stop_worker():
    _shutdown.set()
    try:
        _request_queue.put(None)
    except Exception:
        pass

# ── Message Classifier ────────────────────────────────────────────────────

PCB_ACTION_KEYWORDS = {
    'add', 'create', 'place', 'route', 'footprint', 'connector', 'pad',
    'track', 'move', 'rotate', 'delete', 'remove', 'modify', 'change',
    'drc', 'net', 'zone', 'via', 'smd', 'tht', 'mm', 'pitch', 'pin',
    'orient', 'layout', 'pour', 'clearance', 'layer', 'stackup',
    'netclass', 'outline', 'edge', 'board', 'generate', 'build', 'make',
    'update', 'rename', 'reposition', 'align', 'space', 'spacing',
    'width', 'height', 'size', 'dimension', 'courtyard', 'silk',
    'copper', 'ground', 'power', 'signal', 'drill', 'hole', 'mount',
    'thermal', 'differential', 'length', 'match', 'tune', 'fanout',
}

def is_pcb_action(message):
    msg = message.lower().strip()
    if any(kw in msg for kw in PCB_ACTION_KEYWORDS):
        return True
    if len(msg.split()) <= 2:
        return False
    conversational = {
        'hello', 'hi', 'hey', 'thanks', 'thank', 'ok', 'okay', 'yes', 'no',
        'what can you do', 'who are you', 'how are you', 'help',
        'explain', 'tell me about', 'what is', 'why', 'how does',
    }
    for pattern in conversational:
        if msg.startswith(pattern):
            return False
    question_starts = {'what', 'why', 'how', 'when', 'where', 'who', 'is', 'can', 'do', 'does'}
    first_word = msg.split()[0] if msg.split() else ''
    if first_word in question_starts:
        return False
    return False

# ── Safe Code Execution ───────────────────────────────────────────────────

SAFE_BUILTINS = {
    'True': True, 'False': False, 'None': None,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'len': len, 'range': range, 'min': min, 'max': max,
    'abs': abs, 'sum': sum, 'sorted': sorted, 'reversed': reversed,
    'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
    'any': any, 'all': all, 'isinstance': isinstance, 'hasattr': hasattr,
    'getattr': getattr, 'setattr': setattr, 'type': type,
    'round': round, 'print': print,
}

def fix_generated_code(code):
    fixes = [
        (r'wxPoint\(', 'VECTOR2I('),
        (r'wxSize\(', 'VECTOR2I('),
        (r'PAD_SHAPE_RECTANGLE', 'PAD_SHAPE_ROUNDRECT'),
        (r'board\.AddFootprint\(', 'board.Add('),
        (r'(?m)^.*FindFootprintByReference.*$', ''),
        (r'\.SetStroke\(', '.SetWidth('),
        (r'pcbnew\.STROKE\(([^)]+)\)', r'pcbnew.FromMM(\1)'),
        (r'\.SetType\(', '.SetAttribute('),
        (r'PAD_SMD(?=[\)\s,.\]]|$)', 'PAD_ATTRIB_SMD'),
        (r'PAD_THT(?=[\)\s,.\]]|$)', 'PAD_ATTRIB_PTH'),
        (r'PAD_NPTH(?=[\)\s,.\]]|$)', 'PAD_ATTRIB_NPTH'),
        (r'FP_LINE\(', 'PCB_SHAPE('),
        (r'FP_SHAPE\(', 'PCB_SHAPE('),
        (r'\.SetPadName\(', '.SetNumber('),
    ]
    for pattern, replacement in fixes:
        code = re.sub(pattern, replacement, code)
    code = re.sub(
        r'(\w+)\s*=\s*pcbnew\.PCB_SHAPE\(([^)]+)\)',
        r'\1 = pcbnew.PCB_SHAPE(\2)\n\1.SetShape(pcbnew.SHAPE_T_SEGMENT)',
        code,
    )
    return code

def execute_pcb_code(code, board):
    code = fix_generated_code(code)
    cleaned = re.sub(r'^\s*(import\s+.*|from\s+\S+\s+import\s+.*)$', '', code, flags=re.MULTILINE)
    local_vars = {
        "board": board,
        "pcbnew": pcbnew,
        "PAD_ATTRIB_SMD": getattr(pcbnew, 'PAD_ATTRIB_SMD', None),
        "PAD_ATTRIB_PTH": getattr(pcbnew, 'PAD_ATTRIB_PTH', None),
        "SHAPE_T_SEGMENT": getattr(pcbnew, 'SHAPE_T_SEGMENT', None),
        "math": __import__('math'),
        "get_fp": lambda ref, b=board: next((fp for fp in b.GetFootprints() if fp.GetReference() == ref), None),
        "remove_fp": lambda ref, b=board: b.Remove(next((fp for fp in b.GetFootprints() if fp.GetReference() == ref), None)) or True if any(fp.GetReference() == ref for fp in b.GetFootprints()) else False,
    }
    try:
        compiled = compile(cleaned, "<ai_code>", "exec")
        exec(compiled, {"__builtins__": SAFE_BUILTINS}, local_vars)
    except Exception:
        return False, traceback.format_exc().strip()
    try:
        board.Save(board.GetFileName())
    except Exception:
        return False, traceback.format_exc().strip()
    global _last_action
    _last_action = extract_action_summary(code, True)
    return True, "Code executed successfully."

def check_duplicate_refs(code, board):
    refs = re.findall(r'SetReference\(\s*["\']([^"\']+)["\']\s*\)', code)
    existing = set()
    for fp in board.GetFootprints():
        existing.add(fp.GetReference().strip())
    conflicts = [r for r in refs if r in existing]
    return conflicts

def extract_action_summary(code, success):
    if not success:
        return ""
    refs = re.findall(r'SetReference\(\s*["\']([^"\']+)["\']\s*\)', code)
    pos_m = re.search(r'SetPosition\(pcbnew\.VECTOR2I_MM\(([^,]+),\s*([^)]+)\)', code)
    pins = len(re.findall(r'SetNumber\(', code))
    parts = []
    if refs:
        parts.append(f"Added {', '.join(refs)}")
    if pins:
        parts.append(f"{pins} pads")
    if pos_m:
        parts.append(f"at ({pos_m.group(1).strip()}, {pos_m.group(2).strip()})mm")
    return " ".join(parts) if parts else ""

# ── Chat Panel UI ─────────────────────────────────────────────────────────

CHAT_HISTORY_SIZE = 100

def safe_call_after(frame, func, *args, **kwargs):
    try:
        if frame and not frame._destroyed:
            wx.CallAfter(func, *args, **kwargs)
    except Exception:
        pass

class AiAssistantFrame(wx.Frame):
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_or_create(cls, parent=None):
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    if not cls._instance._destroyed:
                        cls._instance.Raise()
                        return cls._instance
                except Exception:
                    pass
            frame = cls(parent)
            cls._instance = frame
            return frame

    @classmethod
    def clear_instance(cls):
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, parent=None):
        style = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(
            self, parent, wx.ID_ANY,
            "AI Assistant",
            size=(950, 640),
            style=style,
        )
        self._destroyed = False
        self._model_ready = False
        self._pending_code = ""
        self._pending_blocks = []
        self._fix_mode = False
        self._retry_count = 0
        self._last_failed_code = ""
        self._last_error_msg = ""
        self._edit_mode = False
        self._stream_buffer = ""
        self._streaming = False
        self._streamed_response = False
        self._history = deque(maxlen=CHAT_HISTORY_SIZE)
        self._models = scan_gguf_models()
        self._current_model_path = MODEL_PATH
        self._current_gpu_layers = GPU_LAYERS

        self._init_ui()
        self._setup_layout()

        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(wx.EVT_WINDOW_DESTROY, self._on_destroy)

        if self._models:
            self._try_load_model()
        else:
            self._status.SetLabel("No .gguf models found")
            self._status.SetForegroundColour("#ef4444")

    def _init_ui(self):
        self._panel = wx.Panel(self)
        self._panel.SetBackgroundColour("#1e1e1e")

        self._chat_display = wx.TextCtrl(
            self._panel, wx.ID_ANY,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 350),
        )
        self._chat_display.SetBackgroundColour("#1e1e1e")
        self._chat_display.SetForegroundColour("#d4d4d4")
        self._chat_display.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.NORMAL, wx.NORMAL))

        self._input = wx.TextCtrl(
            self._panel, wx.ID_ANY,
            style=wx.TE_PROCESS_ENTER | wx.TE_MULTILINE,
            size=(-1, 60),
        )
        self._input.SetBackgroundColour("#2b2b2b")
        self._input.SetForegroundColour("#d4d4d4")
        self._input.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.NORMAL, wx.NORMAL))
        self._input.Bind(wx.EVT_TEXT_ENTER, self._on_send)
        self._input.Bind(wx.EVT_KEY_DOWN, self._on_input_key)
        self._input.Disable()

        self._send_btn = wx.Button(self._panel, wx.ID_ANY, "Send")
        self._send_btn.SetBackgroundColour("#7c3aed")
        self._send_btn.SetForegroundColour("#ffffff")
        self._send_btn.Bind(wx.EVT_BUTTON, self._on_send)
        self._send_btn.Disable()

        self._apply_btn = wx.Button(self._panel, wx.ID_ANY, "Apply Code")
        self._apply_btn.SetBackgroundColour("#22c55e")
        self._apply_btn.SetForegroundColour("#ffffff")
        self._apply_btn.Disable()
        self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply_code)

        self._edit_btn = wx.Button(self._panel, wx.ID_ANY, "Edit")
        self._edit_btn.SetBackgroundColour("#f59e0b")
        self._edit_btn.SetForegroundColour("#ffffff")
        self._edit_btn.Disable()
        self._edit_btn.Bind(wx.EVT_BUTTON, self._on_edit_code)

        self._clear_btn = wx.Button(self._panel, wx.ID_ANY, "Clear")
        self._clear_btn.Bind(wx.EVT_BUTTON, self._on_clear)

        self._analyze_btn = wx.Button(self._panel, wx.ID_ANY, "Analyze Board")
        self._analyze_btn.SetBackgroundColour("#3b82f6")
        self._analyze_btn.SetForegroundColour("#ffffff")
        self._analyze_btn.Bind(wx.EVT_BUTTON, self._on_analyze_board)

        self._save_btn = wx.Button(self._panel, wx.ID_ANY, "Save Chat")
        self._save_btn.SetBackgroundColour("#6366f1")
        self._save_btn.SetForegroundColour("#ffffff")
        self._save_btn.Bind(wx.EVT_BUTTON, self._on_save_chat)

        self._search_btn = wx.Button(self._panel, wx.ID_ANY, "Search Lib")
        self._search_btn.SetBackgroundColour("#a855f7")
        self._search_btn.SetForegroundColour("#ffffff")
        self._search_btn.Bind(wx.EVT_BUTTON, self._on_search_lib)

        self._model_combo = wx.Choice(self._panel, wx.ID_ANY, size=(280, -1))
        self._model_combo.SetBackgroundColour("#2b2b2b")
        self._model_combo.SetForegroundColour("#d4d4d4")
        self._model_combo.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.NORMAL, wx.NORMAL))
        self._model_combo.Bind(wx.EVT_CHOICE, self._on_model_select)
        self._populate_model_combo()

        self._browse_btn = wx.Button(self._panel, wx.ID_ANY, "Browse...", size=(70, -1))
        self._browse_btn.SetBackgroundColour("#404040")
        self._browse_btn.SetForegroundColour("#d4d4d4")
        self._browse_btn.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.NORMAL, wx.NORMAL))
        self._browse_btn.Bind(wx.EVT_BUTTON, self._on_browse_model)

        self._backend_combo = wx.Choice(self._panel, wx.ID_ANY, size=(100, -1))
        self._backend_combo.SetBackgroundColour("#2b2b2b")
        self._backend_combo.SetForegroundColour("#d4d4d4")
        self._backend_combo.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.NORMAL, wx.NORMAL))
        for label in BACKEND_OPTIONS:
            self._backend_combo.Append(label, BACKEND_OPTIONS[label])
        self._backend_combo.SetSelection(0)
        self._backend_combo.Bind(wx.EVT_CHOICE, self._on_backend_select)

        self._temp_combo = wx.Choice(self._panel, wx.ID_ANY, size=(105, -1))
        self._temp_combo.SetBackgroundColour("#2b2b2b")
        self._temp_combo.SetForegroundColour("#d4d4d4")
        self._temp_combo.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.NORMAL, wx.NORMAL))
        for label in TEMP_OPTIONS:
            self._temp_combo.Append(label, TEMP_OPTIONS[label])
        self._temp_combo.SetSelection(1)
        self._temp_combo.Bind(wx.EVT_CHOICE, self._on_temp_select)

        self._status = wx.StaticText(self._panel, wx.ID_ANY, "Loading model...")
        self._status.SetForegroundColour("#a0a0a0")
        self._status.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.NORMAL, wx.NORMAL))

    def _setup_layout(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._chat_display, 1, wx.EXPAND | wx.ALL, 5)

        model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_label = wx.StaticText(self._panel, wx.ID_ANY, " Model:")
        model_label.SetForegroundColour("#a0a0a0")
        model_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.NORMAL, wx.NORMAL))
        model_sizer.Add(model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_sizer.Add(self._model_combo, 0, wx.RIGHT, 5)
        model_sizer.Add(self._browse_btn, 0, wx.RIGHT, 15)

        backend_label = wx.StaticText(self._panel, wx.ID_ANY, "GPU:")
        backend_label.SetForegroundColour("#a0a0a0")
        backend_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.NORMAL, wx.NORMAL))
        model_sizer.Add(backend_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_sizer.Add(self._backend_combo, 0, wx.RIGHT, 15)

        temp_label = wx.StaticText(self._panel, wx.ID_ANY, "Temp:")
        temp_label.SetForegroundColour("#a0a0a0")
        temp_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.NORMAL, wx.NORMAL))
        model_sizer.Add(temp_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_sizer.Add(self._temp_combo, 0, wx.RIGHT, 5)

        model_sizer.AddStretchSpacer()
        model_sizer.Add(self._status, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(model_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        sizer.Add(self._input, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self._send_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self._apply_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self._edit_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self._clear_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self._analyze_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self._save_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self._search_btn, 0, wx.RIGHT, 5)
        btn_sizer.AddStretchSpacer()
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self._panel.SetSizer(sizer)

    def _on_close(self, event):
        self._destroyed = True
        self.Destroy()

    def _on_destroy(self, event):
        self._destroyed = True
        self.clear_instance()
        event.Skip()

    def _safe_call(self, func, *args, **kwargs):
        if self._destroyed:
            return
        try:
            wx.CallAfter(func, *args, **kwargs)
        except Exception:
            pass

    def _try_load_model(self):
        threading.Thread(target=self._load_model_async, daemon=True).start()

    def _load_model_async(self):
        try:
            load_model(path=self._current_model_path, gpu_layers=self._current_gpu_layers)
            self._safe_call(self._on_model_ready)
        except Exception as e:
            self._safe_call(self._on_model_error, str(e))

    def _on_model_ready(self):
        if self._destroyed:
            return
        self._model_ready = True
        self._input.Enable()
        self._send_btn.Enable()
        self._analyze_btn.Enable()
        name = model_basename(self._current_model_path)
        backend = "Vulkan" if self._current_gpu_layers > 0 else "CPU"
        self._status.SetLabel(f"{name} ({backend})")
        self._status.SetForegroundColour("#22c55e")
        self._append_chat("System", f"Model loaded: {name} ({backend}). Ready!", "assistant")

    def _on_model_error(self, error_msg):
        if self._destroyed:
            return
        self._status.SetLabel("Model failed to load")
        self._status.SetForegroundColour("#ef4444")
        self._append_chat("System", f"Failed to load model: {error_msg}\nCheck model path: {MODEL_PATH}", "error")

    def _format_status(self, base_label):
        return f"{base_label} (~{_last_token_estimate}+{MAX_OUTPUT}/{CTX_SIZE} tok)"

    def _on_send(self, event=None):
        if self._destroyed or not self._model_ready:
            return
        text = self._input.GetValue().strip()
        if not text:
            return
        self._fix_mode = False
        self._retry_count = 0
        self._last_failed_code = ""
        self._last_error_msg = ""
        self._edit_mode = False
        self._input.Clear()
        self._append_chat("You", text, "user")
        self._input.Disable()
        self._send_btn.Disable()
        self._analyze_btn.Disable()
        self._status.SetLabel(self._format_status("Thinking..."))
        self._status.SetForegroundColour("#a0a0a0")

        if is_pcb_action(text):
            context = get_context()
            if self._history:
                recent = list(self._history)[-8:]
                lines = ["\n\nRecent conversation:"]
                for sender, text_h, msg_type, ts in recent:
                    if msg_type in ('user', 'assistant'):
                        lines.append(f"{sender}: {text_h[:300]}")
                context += "\n".join(lines)
        else:
            context = "Board is open. (Conversational mode — board details omitted.)"
        start_worker()
        submit_request(text, context, self._on_response, stream_callback=self._on_stream_token)

    def _on_stream_token(self, token, done):
        self._safe_call(self._handle_stream_token, token, done)

    def _handle_stream_token(self, token, done):
        if self._destroyed:
            return
        if not self._streaming:
            self._streaming = True
            self._stream_buffer = ""
            self._streamed_response = False
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._chat_display.SetDefaultStyle(wx.TextAttr("#d4d4d4", "#1e1e1e"))
            self._chat_display.AppendText(f"[{timestamp}] Assistant:\n")
        if done:
            self._streaming = False
            self._streamed_response = True
            return
        self._stream_buffer += token
        self._chat_display.AppendText(token)
        self._chat_display.ShowPosition(self._chat_display.GetLastPosition())

    def _on_response(self, response, error):
        self._safe_call(self._handle_response, response, error)

    def _extract_code_blocks(self, text):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if not blocks:
            m = re.search(r"```(?:python)?\s*\n(.+?)(```|$)", text, re.DOTALL)
            if m:
                block = m.group(1).strip()
                if block:
                    blocks = [block]
        if blocks:
            return "\n\n".join(cb.strip() for cb in blocks if cb.strip())
        return ""

    def _extract_code_blocks_list(self, text):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if not blocks:
            m = re.search(r"```(?:python)?\s*\n(.+?)(```|$)", text, re.DOTALL)
            if m:
                block = m.group(1).strip()
                if block:
                    blocks = [block]
        return [cb.strip() for cb in blocks if cb.strip()]

    def _execute_block(self, code, board):
        return execute_pcb_code(code, board)

    def _handle_response(self, response, error):
        if self._destroyed:
            return
        if self._fix_mode:
            self._handle_fix_response(response, error)
            return
        self._input.Enable()
        self._send_btn.Enable()
        self._analyze_btn.Enable()
        if error:
            self._append_chat("Error", error, "error")
            self._status.SetLabel("Error generating response")
            self._status.SetForegroundColour("#ef4444")
            return
        if not self._streamed_response:
            self._append_chat("Assistant", response, "assistant")
        else:
            ts = datetime.now().strftime("%H:%M:%S")
            self._history.append(("Assistant", response, "assistant", ts))
        self._streamed_response = False
        self._status.SetLabel(self._format_status("Ready"))
        self._status.SetForegroundColour("#22c55e")

        self._pending_code = self._extract_code_blocks(response)
        self._pending_blocks = self._extract_code_blocks_list(response)
        if self._pending_code:
            self._apply_btn.Enable()
            self._edit_btn.Enable()
            self._append_chat("System", f"Code block extracted ({len(self._pending_code)} chars). Click 'Apply Code' to execute.", "info")
        else:
            self._apply_btn.Disable()
            self._edit_btn.Disable()

    def _request_fix(self, code, error_msg):
        self._fix_mode = True
        fix_prompt = (
            "Your previous Python code for KiCad failed with this error:\n"
            f"{error_msg}\n\n"
            "Faulty code:\n"
            f"```python\n{code}\n```\n\n"
            "Fix the bug. Output only the corrected Python code in a markdown code block. "
            "Do NOT use import statements."
        )
        self._input.Disable()
        self._send_btn.Disable()
        self._status.SetLabel(self._format_status("Fixing code..."))
        self._status.SetForegroundColour("#f59e0b")
        submit_request(fix_prompt, "", self._on_response, stream_callback=self._on_stream_token)

    def _handle_fix_response(self, response, error):
        if error:
            self._append_chat("Auto-Fix Error", error, "error")
            self._input.Enable()
            self._send_btn.Enable()
            self._status.SetLabel(self._format_status("Ready"))
            self._status.SetForegroundColour("#22c55e")
            self._retry_count = 0
            return
        if not self._streamed_response:
            self._append_chat("Assistant (Fixed)", response, "assistant")
        self._streamed_response = False
        self._pending_code = self._extract_code_blocks(response)
        self._pending_blocks = self._extract_code_blocks_list(response)
        if not self._pending_code:
            self._append_chat("Auto-Fix Failed", "Could not extract code from fix response", "error")
            self._input.Enable()
            self._send_btn.Enable()
            self._status.SetLabel(self._format_status("Ready"))
            self._status.SetForegroundColour("#22c55e")
            self._retry_count = 0
            return
        self._auto_apply_fix()

    def _auto_apply_fix(self):
        try:
            board = pcbnew.GetBoard()
        except Exception as e:
            self._append_chat("System", f"Cannot get board: {e}", "error")
            self._input.Enable()
            self._send_btn.Enable()
            self._retry_count = 0
            return
        success, msg = execute_pcb_code(self._pending_code, board)
        if success:
            self._append_chat("Auto-Fix Applied", self._pending_code + "\n\n" + msg, "success")
            self._pending_code = ""
            self._apply_btn.Disable()
            self._edit_btn.Disable()
            self._input.Enable()
            self._send_btn.Enable()
            self._status.SetLabel(self._format_status("Ready"))
            self._status.SetForegroundColour("#22c55e")
            self._retry_count = 0
            return
        self._retry_count += 1
        if self._retry_count < 2:
            self._last_failed_code = self._pending_code
            self._last_error_msg = msg
            self._pending_code = ""
            self._apply_btn.Disable()
            self._edit_btn.Disable()
            self._append_chat("System", f"Fix attempt {self._retry_count}/2 failed: {msg}. Retrying...", "info")
            self._request_fix(self._last_failed_code, msg)
        else:
            self._append_chat("Auto-Fix Error", msg, "error")
            self._append_chat("System", "Auto-fix exhausted after 2 attempts. Please rephrase your request.", "info")
            self._pending_code = ""
            self._apply_btn.Disable()
            self._edit_btn.Disable()
            self._input.Enable()
            self._send_btn.Enable()
            self._status.SetLabel(self._format_status("Ready"))
            self._status.SetForegroundColour("#22c55e")
            self._retry_count = 0

    def _on_apply_code(self, event=None):
        if self._destroyed:
            return
        code = self._pending_code
        if self._edit_mode:
            code = self._input.GetValue().strip()
            if not code:
                return
        if not code:
            return
        try:
            board = pcbnew.GetBoard()
        except Exception as e:
            self._append_chat("System", f"Cannot get board: {e}", "error")
            return
        conflicts = check_duplicate_refs(code, board)
        warn = ""
        if conflicts:
            warn = f"\n\nWARNING: Reference(s) already exist: {', '.join(conflicts)}"
        dlg = wx.MessageDialog(
            self,
            "Apply generated code to the board?" + warn + "\n\n" + code[:500],
            "Confirm Code Execution",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
        )
        if dlg.ShowModal() != wx.ID_YES:
            dlg.Destroy()
            return
        dlg.Destroy()

        blocks = self._pending_blocks
        if self._edit_mode:
            blocks = [code]

        for idx, block_code in enumerate(blocks):
            if len(blocks) > 1:
                self._append_chat("System", f"Running block {idx+1}/{len(blocks)}...", "info")
            success, msg = self._execute_block(block_code, board)
            if success:
                if len(blocks) > 1:
                    self._append_chat(f"Block {idx+1} OK", block_code[:200] + "..." if len(block_code) > 200 else block_code, "success")
                else:
                    self._append_chat("Code Executed", block_code + "\n\n" + msg, "success")
                continue
            self._retry_count += 1
            self._last_failed_code = block_code
            self._last_error_msg = msg
            self._apply_btn.Disable()
            self._edit_btn.Disable()
            self._edit_mode = False
            self._append_chat("System", f"Block {idx+1}/{len(blocks)} failed: {msg}. Auto-fixing ({self._retry_count}/2)...", "info")
            self._request_fix(block_code, msg)
            return

        self._pending_code = ""
        self._pending_blocks = []
        self._apply_btn.Disable()
        self._edit_btn.Disable()
        self._retry_count = 0
        self._edit_mode = False

    def _populate_model_combo(self):
        self._model_combo.Clear()
        for path, name, size_gb in self._models:
            self._model_combo.Append(format_model_label(path, name, size_gb), path)
        if self._model_combo.GetCount() > 0:
            current = model_basename(self._current_model_path)
            for i in range(self._model_combo.GetCount()):
                if current in self._model_combo.GetString(i):
                    self._model_combo.SetSelection(i)
                    break
            else:
                self._model_combo.SetSelection(0)

    def _on_model_select(self, event):
        idx = self._model_combo.GetSelection()
        if idx == wx.NOT_FOUND:
            return
        path = self._model_combo.GetClientData(idx)
        if path and path != self._current_model_path:
            self._switch_model(path=path)

    def _on_backend_select(self, event):
        idx = self._backend_combo.GetSelection()
        if idx == wx.NOT_FOUND:
            return
        gpu_layers = self._backend_combo.GetClientData(idx)
        if gpu_layers != self._current_gpu_layers:
            self._switch_model(gpu_layers=gpu_layers)

    def _on_temp_select(self, event):
        global TEMPERATURE
        idx = self._temp_combo.GetSelection()
        if idx == wx.NOT_FOUND:
            return
        TEMPERATURE = self._temp_combo.GetClientData(idx)
        self._append_chat("System", f"Temperature set to {TEMPERATURE}", "info")

    def _on_browse_model(self, event):
        dlg = wx.FileDialog(
            self, "Select a GGUF model file",
            defaultDir=str(MODEL_DIR),
            wildcard="GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_CANCEL:
            dlg.Destroy()
            return
        path = dlg.GetPath()
        dlg.Destroy()
        size_gb = os.path.getsize(path) / 1e9
        name = Path(path).stem
        self._models.append((path, name, size_gb))
        self._model_combo.Append(format_model_label(path, name, size_gb), path)
        self._model_combo.SetSelection(self._model_combo.GetCount() - 1)
        self._switch_model(path)

    def _switch_model(self, path=None, gpu_layers=None):
        self._model_ready = False
        if path is not None:
            self._current_model_path = path
        if gpu_layers is not None:
            self._current_gpu_layers = gpu_layers
        self._input.Disable()
        self._send_btn.Disable()
        self._apply_btn.Disable()
        self._edit_btn.Disable()
        self._analyze_btn.Disable()
        self._pending_code = ""
        self._status.SetLabel("Loading model...")
        self._status.SetForegroundColour("#a0a0a0")
        unload_model()
        threading.Thread(target=self._load_model_async, daemon=True).start()

    def _on_clear(self, event=None):
        if self._destroyed:
            return
        self._chat_display.Clear()
        self._history.clear()

    def _on_edit_code(self, event=None):
        if self._destroyed or not self._pending_code:
            return
        self._edit_mode = True
        self._input.SetValue(self._pending_code)
        self._input.Enable()
        self._send_btn.Enable()
        self._apply_btn.Enable()
        self._edit_btn.Disable()
        self._pending_code = ""
        self._status.SetLabel("Edit the code, then click Apply Code")
        self._status.SetForegroundColour("#f59e0b")

    def _on_analyze_board(self, event=None):
        if self._destroyed or not self._model_ready:
            return
        self._input.Disable()
        self._send_btn.Disable()
        self._status.SetLabel(self._format_status("Analyzing board..."))
        self._status.SetForegroundColour("#a0a0a0")
        context = get_context()
        prompt = (
            "Analyze the board. Report any issues, placement problems, "
            "routing suggestions, unconnected nets, DRC concerns, and "
            "optimization recommendations. Be concise and specific."
        )
        start_worker()
        submit_request(prompt, context, self._on_response, stream_callback=self._on_stream_token)

    def _on_save_chat(self, event=None):
        if self._destroyed or not self._history:
            return
        dlg = wx.FileDialog(
            self, "Save Chat History",
            defaultDir=str(Path.home()),
            wildcard="Text files (*.txt)|*.txt|Markdown (*.md)|*.md",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_CANCEL:
            dlg.Destroy()
            return
        path = dlg.GetPath()
        dlg.Destroy()
        try:
            with open(path, 'w') as f:
                f.write("=== KiCad AI Assistant Session ===\n\n")
                for sender, text, msg_type, ts in self._history:
                    f.write(f"[{ts}] {sender}:\n{text}\n\n")
                if self._pending_code:
                    f.write(f"\n--- PENDING CODE ---\n{self._pending_code}\n")
            self._append_chat("System", f"Chat saved to {path}", "info")
        except Exception as e:
            self._append_chat("System", f"Failed to save: {e}", "error")

    def _on_search_lib(self, event=None):
        if self._destroyed:
            return
        dlg = wx.TextEntryDialog(self, "Search footprint libraries:", "Library Search")
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        term = dlg.GetValue().strip().lower()
        dlg.Destroy()
        if not term:
            return
        self._append_chat("System", f"Searching libraries for '{term}'...", "info")
        results = []
        try:
            fptable = getattr(pcbnew, 'GetGlobalFootprintLibraryTable', None)
            if fptable:
                fptable = fptable()
                count = fptable.GetCount()
                for i in range(count):
                    lib_nick = fptable.GetNickname(i)
                    uri = fptable.GetURI(i)
                    if not uri:
                        continue
                    try:
                        names = pcbnew.FootprintEnumerate(uri)
                    except Exception:
                        continue
                    for fpname in names:
                        if term in fpname.lower():
                            results.append(f"{fpname}  [{lib_nick}]")
        except Exception:
            pass
        if not results and MODEL_DIR.parent.exists():
            try:
                import glob as _glob
                lib_dirs = [
                    str(MODEL_DIR.parent / ".local" / "share" / "kicad" / "10.0" / "footprints"),
                    "/usr/share/kicad/footprints",
                    str(Path.home() / ".local" / "share" / "kicad" / "10.0" / "footprints"),
                ]
                for lib_dir in lib_dirs:
                    if not Path(lib_dir).is_dir():
                        continue
                    for f in _glob.glob(f"{lib_dir}/**/*.kicad_mod", recursive=True):
                        name = Path(f).stem
                        if term in name.lower():
                            results.append(f"{name}  [{Path(lib_dir).name}]")
                    if results:
                        break
            except Exception:
                pass
        if results:
            out = f"Matches for '{term}' ({len(results)} found):\n"
            for r in results[:30]:
                out += f"  {r}\n"
            if len(results) > 30:
                out += f"  ... and {len(results) - 30} more\n"
            self._append_chat("Library", out, "assistant")
        else:
            self._append_chat("Library", f"No matches for '{term}'.", "info")

    def _on_input_key(self, event):
        if event.GetKeyCode() == wx.WXK_RETURN and not event.ShiftDown():
            self._on_send()
        else:
            event.Skip()

    def _append_chat(self, sender, text, msg_type="assistant"):
        if self._destroyed:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        if msg_type == "user":
            self._chat_display.SetDefaultStyle(wx.TextAttr("#e0e0e0", "#2b2b2b"))
        elif msg_type == "error":
            self._chat_display.SetDefaultStyle(wx.TextAttr("#ef4444", "#1e1e1e"))
        elif msg_type == "success":
            self._chat_display.SetDefaultStyle(wx.TextAttr("#22c55e", "#1e1e1e"))
        elif msg_type == "info":
            self._chat_display.SetDefaultStyle(wx.TextAttr("#7c3aed", "#1e1e1e"))
        else:
            self._chat_display.SetDefaultStyle(wx.TextAttr("#d4d4d4", "#1e1e1e"))
        self._chat_display.AppendText(f"[{timestamp}] {sender}:\n{text}\n\n")
        self._chat_display.ShowPosition(self._chat_display.GetLastPosition())
        self._history.append((sender, text, msg_type, timestamp))
