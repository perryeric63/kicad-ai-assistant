"""KiCad AI Assistant Plugin — PCB & Footprint Editor support.
Imports shared core from ai_core. Provides PCB/FP SYSTEM_PROMPT and context builders.
"""

import pcbnew
from pathlib import Path
import ai_core

# ── SYSTEM_PROMPTs ────────────────────────────────────────────────────────

PCB_SYSTEM_PROMPT = (
    "You are an AI assistant integrated into KiCad's PCB editor. "
    "Help the user with PCB design: place footprints, route tracks, explain DRC errors, "
    "suggest component placements, generate footprints, and answer electronics questions. "
    "When the user asks you to perform a PCB action, output Python code using the pcbnew API "
    "in a markdown code block.\n\n"
    "RULES:\n"
    "0. Think step-by-step before writing code. First reason about coordinates, pin layout, "
    "and pitch, then write the code. Double-check your work.\n"
    "1. Do NOT use `import` statements. The `pcbnew` module and `board` are already available.\n"
    "2. Use VECTOR2I_MM for millimeter positions (easier than raw nm).\n"
    "3. Only use the CORRECT API below.\n\n"
    "CORRECT KiCad 10 Python API:\n"
    "  fp = pcbnew.FOOTPRINT(board)                         # create footprint\n"
    "  fp.SetReference(\"REF\")\n"
    "  fp.SetValue(\"VAL\")\n"
    "  fp.SetPosition(pcbnew.VECTOR2I_MM(x_mm, y_mm))       # mm, not nm!\n"
    "  fp.SetOrientationDegrees(0)\n\n"
    "  pad = pcbnew.PAD(fp)                                  # create pad\n"
    "  pad.SetNumber(str(n))\n"
    "  pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)\n"
    "  pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)               # NOT SetType!\n"
    "  pad.SetSize(pcbnew.VECTOR2I_MM(w_mm, h_mm))           # mm!\n"
    "  pad.SetPosition(pcbnew.VECTOR2I_MM(x_mm, y_mm))       # mm!\n"
    "  pad.SetLayerSet(pad.SMDMask())                        # simpler than manual LSET\n"
    "  pad.SetRoundRectRadiusRatio(0.25)\n"
    "  fp.Add(pad)\n\n"
    "  board.Add(fp)                                         # add to board\n"
    "  # board.Save() is automatic\n\n"
    "  # Single-row connector pattern:\n"
    "  for i in range(pin_count):\n"
    "      pad = pcbnew.PAD(fp)\n"
    "      pad.SetNumber(str(i + 1))\n"
    "      pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)\n"
    "      pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)\n"
    "      pad.SetSize(pcbnew.VECTOR2I_MM(w_mm, h_mm))\n"
    "      pad.SetPosition(pcbnew.VECTOR2I_MM(i * pitch_mm, 0))\n"
    "      pad.SetLayerSet(pad.SMDMask())\n"
    "      pad.SetRoundRectRadiusRatio(0.25)\n"
    "      fp.Add(pad)\n\n"
    "  silk = pcbnew.PCB_SHAPE(fp)                           # NOT FP_LINE!\n"
    "  silk.SetShape(pcbnew.SHAPE_T_SEGMENT)\n"
    "  silk.SetStart(pcbnew.VECTOR2I_MM(x1, y1))\n"
    "  silk.SetEnd(pcbnew.VECTOR2I_MM(x2, y2))\n"
    "  silk.SetLayer(pcbnew.F_SilkS)\n"
    "  silk.SetWidth(pcbnew.FromMM(0.127))                   # 0.127mm\n"
    "  fp.Add(silk)\n\n"
    "  # ── THT (through-hole) pad ──\n"
    "  pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)               # NOT PAD_THT!\n"
    "  pad.SetDrillSize(pcbnew.VECTOR2I_MM(drill, drill))\n"
    "  layers = pcbnew.LSET()\n"
    "  layers.AddLayer(pcbnew.F_Cu)\n"
    "  layers.AddLayer(pcbnew.B_Cu)\n"
    "  pad.SetLayerSet(layers)\n\n"
    "  # ── Dual-row connector pattern ──\n"
    "  for row in range(2):\n"
    "      for col in range(cols):\n"
    "          i = row * cols + col\n"
    "          pad.SetNumber(str(i + 1))\n"
    "          x = col * pitch_mm\n"
    "          y = row * row_spacing_mm\n"
    "          pad.SetPosition(pcbnew.VECTOR2I_MM(x, y))\n\n"
    "  # ── Mounting hole (unplated) ──\n"
    "  pad.SetNumber(\"\")                                     # no pin number\n"
    "  pad.SetAttribute(pcbnew.PAD_ATTRIB_NPTH)              # unplated\n"
    "  pad.SetDrillSize(pcbnew.VECTOR2I_MM(3.0, 3.0))\n\n"
    "  # ── Net assignment ──\n"
    "  pad.SetNet(pcbnew.NET_INFO(board, \"GND\"))\n\n"
    "  # ── Courtyard layer (F_CrtYd) ──\n"
    "  shape = pcbnew.PCB_SHAPE(fp)\n"
    "  shape.SetShape(pcbnew.SHAPE_T_SEGMENT)\n"
    "  shape.SetLayer(pcbnew.F_CrtYd)\n"
    "  shape.SetWidth(pcbnew.FromMM(0.05))\n"
    "  fp.Add(shape)\n\n"
    "  # ── Modify existing footprint ──\n"
    "  fp = get_fp(\"J1\")                                  # find by reference\n"
    "  fp.SetPosition(pcbnew.VECTOR2I_MM(x, y))             # move it\n"
    "  fp.SetOrientationDegrees(90)                         # rotate\n"
    "  remove_fp(\"OLD_CONN\")                               # delete footprint\n\n"
    "NEVER USE: FP_LINE, FP_SHAPE, wxPoint, wxSize, FindFootprintByReference, PAD_SHAPE_RECTANGLE, board.AddFootprint, SetType, PAD_SMD, PAD_THT, PAD_NPTH, SetStroke, STROKE"
)

# ── Tool Knowledge (shared across all tools) ──────────────────────────────

TOOL_KNOWLEDGE = """
=== KiCad Tools Knowledge ===

--- Eeschema (Schematic Editor) ---
- Create and edit schematic sheets with hierarchical design support.
- Place symbols from installed libraries; use Place -> Symbol (A shortcut).
- Connect pins with wires (W), buses, and net labels.
- Assign footprints to symbols using CvPcb or symbol properties.
- Annotate schematic (Tools -> Annotate) to auto-assign reference designators.
- Run ERC (Electrical Rules Check) to find unconnected pins, conflicts, power issues.
- Generate netlist (Tools -> Generate Netlist) for PCB layout import.
- BOM (Bill of Materials) exported via File -> Fabrication Outputs -> BOM.
- BOM scripts are standalone Python files in /usr/share/kicad/plugins/ using kicad_netlist_reader.
- Common symbols: R (resistor), C (capacitor), L (inductor), D (diode), Q (transistor), U (IC), J (connector), SW (switch), TP (test point).
- Net classes: assign in Schematic Setup for power, signal, differential pairs.
- Hierarchical labels connect signals between sheet levels.

--- Symbol Editor ---
- Create custom schematic symbols: File -> New Library, then create new symbol.
- Draw body with lines, rectangles, circles, arcs, polygons.
- Add pins with electrical type (input, output, bidirectional, power, etc.).
- Hide power pins (VCC/GND) for ICs; set pin number/name.
- Pin table: View -> Pin Table for bulk editing.
- Offset/rotation: pins can be placed on any side with any rotation.
- Common pin lengths: 100mil or 150mil.
- Symbol unit concept: multiple interchangeable sections (e.g., 4 NAND gates in a 7400).

--- GerbView (Gerber Viewer) ---
- View and inspect Gerber (.gbr) and drill (.drl) files.
- Layer stack: load multiple Gerber files; toggle visibility per layer.
- D-code list: shows apertures used in each Gerber file.
- Measure tool for checking distances between features.
- Export to PDF/PNG for documentation.
- Cross-probe with PCB editor (when open).
- Gerber layers: F_Cu, B_Cu, inner layers, F_SilkS, B_SilkS, F_Mask, B_Mask, F_Paste, B_Paste, Edge_Cuts.
- Common Gerber file extensions: .GTL (top copper), .GBL (bottom), .GTO (top silk), .GBO (bottom), .GTS (top mask), .GBS (bottom), .GKO (outline).

--- Drawing Sheet Editor (PlEditor) ---
- Customize title blocks, borders, and drawing frames.
- Standard sizes: A4, A3, A2, A1, A0 (and US equivalents).
- Title block fields: title, date, revision, company, sheet number, total sheets.
- Text variables: ${TITLE}, ${REVISION}, ${DATE}, ${SHEETNAME}, ${FILENAME}, etc.
- Custom logos and graphics can be added.
- Templates: .kicad_wks files in template directory.

--- PCB Calculator ---
- Track width calculator: based on current and temperature rise (IPC-2221).
- Via size calculator: annular ring requirements.
- Differential pair calculator: trace width and spacing for controlled impedance.
- RF calculators for microstrip, stripline, coplanar waveguide.

--- Bitmap2Component (Image Converter) ---
- Converts BMP/PNG images to footprint silk screen or copper layer.
- Grayscale mapped to pad height or silk line thickness.
- Resolution (DPI) determines output quality.
- Output is a .kicad_mod footprint file.

--- Plugin and Content Manager (PCM) ---
- Built-in package manager for KiCad plugins, libraries, and themes.
- PCM stores downloaded content in user's KiCad data directory.
- Repositories: default KiCad repo + custom URLs.
- Installed plugins appear in Tools -> External Plugins or toolbar buttons."""

FP_SYSTEM_PROMPT = (
    "You are an AI assistant integrated into KiCad's Footprint Editor. "
    "Help the user create and edit footprints: place pads, draw silk screen, "
    "add courtyard outlines, set 3D models, and configure footprint properties. "
    "When the user asks you to perform a footprint action, output Python code "
    "using the pcbnew API in a markdown code block.\n\n"
    "RULES:\n"
    "0. Think step-by-step before writing code. First reason about pad layout "
    "and pitch, then write the code.\n"
    "1. Do NOT use `import` statements. `pcbnew` and `board` (the footprint) are available.\n"
    "2. Use VECTOR2I_MM for millimeter positions.\n"
    "3. In the Footprint Editor, `board` IS the footprint itself — "
    "do NOT call `board.Add(fp)`. Use the footprint directly.\n\n"
    "CORRECT KiCad 10 Footprint Editor API:\n"
    "  fp = board                                # board IS the footprint here\n"
    "  fp.SetReference(\"REF\")\n"
    "  fp.SetValue(\"VAL\")\n"
    "  pad = pcbnew.PAD(fp)\n"
    "  pad.SetNumber(str(n))\n"
    "  pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)\n"
    "  pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)\n"
    "  pad.SetSize(pcbnew.VECTOR2I_MM(w_mm, h_mm))\n"
    "  pad.SetPosition(pcbnew.VECTOR2I_MM(x_mm, y_mm))\n"
    "  pad.SetLayerSet(pad.SMDMask())\n"
    "  pad.SetRoundRectRadiusRatio(0.25)\n"
    "  fp.Add(pad)\n\n"
    "  # Silk, courtyard, THT pads — same API as PCB editor above.\n\n"
    "NEVER USE: FP_LINE, FP_SHAPE, wxPoint, FindFootprintByReference, board.AddFootprint, SetType, PAD_SMD, PAD_THT, PAD_NPTH, SetStroke, STROKE"
)

# ── Tool Detection ────────────────────────────────────────────────────────

CURRENT_TOOL = "pcb"

def detect_tool():
    global CURRENT_TOOL
    try:
        board = pcbnew.GetBoard()
        if board is None:
            CURRENT_TOOL = "unknown"
            return "unknown"
        fps = list(board.GetFootprints())
        if len(fps) > 1:
            CURRENT_TOOL = "pcb"
            return "pcb"
        elif len(fps) == 1:
            CURRENT_TOOL = "footprint"
            return "footprint"
        else:
            # Check if we're in footprint editor by trying to cast board to FOOTPRINT
            try:
                board.GetReference()
                CURRENT_TOOL = "footprint"
                return "footprint"
            except Exception:
                pass
            CURRENT_TOOL = "unknown"
            return "unknown"
    except Exception:
        CURRENT_TOOL = "unknown"
        return "unknown"

def get_system_prompt():
    tool = detect_tool()
    if tool == "footprint":
        return FP_SYSTEM_PROMPT + TOOL_KNOWLEDGE
    return PCB_SYSTEM_PROMPT + TOOL_KNOWLEDGE

# ── Context Builders ──────────────────────────────────────────────────────

def get_context_for_selection():
    lines = []
    try:
        board = pcbnew.GetBoard()
    except Exception:
        return "No board open."
    if board is None:
        return "No board open."
    try:
        for fp in board.GetFootprints():
            if fp.IsSelected():
                pos = fp.GetPosition()
                lines.append(f"Selected: {fp.GetReference()} ({fp.GetValue()}) at ({pos.x/1e6:.1f}, {pos.y/1e6:.1f})")
        for t in board.GetTracks():
            if t.IsSelected():
                s, e = t.GetStart(), t.GetEnd()
                lines.append(f"Selected track: ({s.x/1e6:.1f}, {s.y/1e6:.1f}) -> ({e.x/1e6:.1f}, {e.y/1e6:.1f})")
    except Exception:
        pass
    return "\n".join(lines) if lines else "Nothing selected."

def get_board_context():
    """PCB Editor context builder — full board information."""
    try:
        board = pcbnew.GetBoard()
    except Exception:
        return "No board is currently open."
    if board is None:
        return "No board is currently open."
    lines = []
    lines.append(f"Tool: PCB Editor | Board: {board.GetFileName()}")
    try:
        lines.append(f"Board thickness: {board.GetDesignSettings().GetBoardThickness()}")
    except Exception:
        pass
    try:
        fp_count = len(list(board.GetFootprints()))
        track_count = len(list(board.GetTracks()))
        bb = board.GetBoardBoundingBox()
        lines.append(f"Board outline (mm): ({bb.GetPosition().x/1e6:.1f}, {bb.GetPosition().y/1e6:.1f}) to ({bb.GetEnd().x/1e6:.1f}, {bb.GetEnd().y/1e6:.1f}) | "
                     f"Footprints: {fp_count} | Tracks: {track_count}")
        layers = board.GetDesignSettings().GetEnabledLayers()
        lines.append(f"Enabled layers: {list(layers.Seq())}")
        zones = [z for z in board.Zones()]
        lines.append(f"Zones: {len(zones)}")

        refs = []
        for fp in board.GetFootprints():
            refs.append(fp.GetReference())
        if refs:
            lines.append(f"\nExisting references: {', '.join(sorted(set(refs)))}")

        try:
            net_info = board.GetNetInfo()
            net_names = []
            for ni in net_info.NetsByName():
                if hasattr(ni, 'GetNetname'):
                    net_names.append(ni.GetNetname())
            if net_names:
                lines.append(f"Nets: {', '.join(sorted(set(net_names)))}")
        except Exception:
            pass

        if fp_count > 0:
            selection = get_context_for_selection()
            if selection and "Nothing selected" not in selection:
                lines.append(f"\n{selection}")
            lines.append("\nFootprints (max 25):")
            fps = list(board.GetFootprints())
            for i, fp in enumerate(fps):
                if i >= 25:
                    lines.append(f"  ... and {fp_count - 25} more")
                    break
                pos = fp.GetPosition()
                ref = fp.GetReference()
                val = fp.GetValue() if fp.GetValue() != ref else ""
                orient = fp.GetOrientationDegrees() if hasattr(fp, 'GetOrientationDegrees') else fp.GetOrientation() / 10.0
                pads = len(list(fp.Pads()))
                label = f"{ref} {val}".strip()
                lines.append(f"  {label} at ({pos.x/1e6:.1f}, {pos.y/1e6:.1f})mm, {orient:.0f} deg, {pads} pads")

        if len(list(board.GetFootprints())) > 0:
            fps = list(board.GetFootprints())
            max_x = max(fp.GetPosition().x for fp in fps) / 1e6
            max_y = max(fp.GetPosition().y for fp in fps) / 1e6
            board_w = bb.GetWidth() / 1e6
            hints = []
            if board_w - max_x > 20:
                hints.append(f"right (X > {max_x+10:.0f} mm)")
            if max_y - bb.GetPosition().y/1e6 > 20:
                hints.append("top")
            if bb.GetPosition().y/1e6 < -10:
                hints.append("bottom")
            if hints:
                lines.append(f"\nFree space hints: {', '.join(hints)}")

        if track_count > 0:
            lines.append("\nTrack list (first 10):")
            for i, t in enumerate(board.GetTracks()):
                if i >= 10:
                    break
                if hasattr(t, 'GetStart'):
                    s, e = t.GetStart(), t.GetEnd()
                    lines.append(f"  Track: ({s.x/1e6:.1f}, {s.y/1e6:.1f}) -> ({e.x/1e6:.1f}, {e.y/1e6:.1f}) w={t.GetWidth()/1e6:.2f}mm")
    except Exception:
        pass

    if ai_core._last_action:
        lines.append(f"\nLast action: {ai_core._last_action}")
    return "\n".join(lines)

def get_footprint_context():
    """Footprint Editor context builder — single footprint detail."""
    try:
        board = pcbnew.GetBoard()
    except Exception:
        return "No footprint open."
    if board is None:
        return "No footprint open."
    lines = []
    lines.append(f"Tool: Footprint Editor | File: {board.GetFileName()}")
    try:
        ref = board.GetReference()
        val = board.GetValue()
        pos = board.GetPosition()
        pads = len(list(board.Pads()))
        lines.append(f"Footprint: {ref} ({val})")
        lines.append(f"Position: ({pos.x/1e6:.1f}, {pos.y/1e6:.1f})mm | Pads: {pads}")
        lines.append("\nPad list:")
        for pad in board.Pads():
            ppos = pad.GetPosition()
            pnum = pad.GetNumber()
            pshape = str(pad.GetShape()).split('.')[-1] if hasattr(pad, 'GetShape') else "?"
            lines.append(f"  Pad {pnum}: ({ppos.x/1e6:.3f}, {ppos.y/1e6:.3f})mm shape={pshape}")
        silk_items = [s for s in board.GraphicalItems() if hasattr(s, 'GetLayer') and 'SilkS' in str(s.GetLayer())]
        lines.append(f"\nSilk items: {len(silk_items)}")
    except Exception:
        pass
    return "\n".join(lines)

def get_context():
    """Auto-detect tool and return appropriate context."""
    tool = detect_tool()
    if tool == "footprint":
        return get_footprint_context()
    return get_board_context()

# ── Plugin Registration ───────────────────────────────────────────────────

# Set up shared core with auto-detecting context and prompt
ai_core.init(get_system_prompt(), get_context)
# Override context builder to use our auto-detecting version
ai_core.CONTEXT_BUILDER = get_context

class AiAssistantPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "AI Assistant"
        self.category = "AI"
        self.description = "AI-powered design assistant for PCB & Footprint Editor"
        self.show_toolbar_button = True
        self.icon_file_name = ""

    def Run(self):
        # Re-init with current tool's system prompt
        ai_core.init(get_system_prompt(), get_context)
        ai_core.CONTEXT_BUILDER = get_context
        frame = ai_core.AiAssistantFrame.get_or_create()
        if frame:
            frame.Show()
            frame.Raise()

plugin = AiAssistantPlugin()
try:
    plugin.register()
except Exception:
    pass
