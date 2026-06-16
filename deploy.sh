#!/bin/bash
# deploy.sh — copies workspace .py files to KiCad plugins directory
set -e
WORKSPACE="$(dirname "$(readlink -f "$0")")"
PLUGIN_DIR="$HOME/.local/share/kicad/10.0/scripting/plugins"

for f in "$WORKSPACE"/*.py; do
    name="$(basename "$f")"
    cp "$f" "$PLUGIN_DIR/$name"
    echo "  $name → $PLUGIN_DIR/"
done
echo "Done. Restart KiCad or reopen the assistant to reload."
