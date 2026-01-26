# Scanbot Simple GUI

Minimal `omni.ui` extension that creates a small window with a slider and a button.

## Quick start
- Launch Isaac Lab (or Kit) with this repo as an extension folder:
  - `./isaaclab.sh --ext-folder /home/scanbot/dev/isaac_lab_scanbot/scanbot/source/extensions --enable scanbot.simple_gui`
- The "Scanbot Simple GUI" window appears on startup. Move the slider to update the value label and press the button to log the current value; the status text in the window updates and the message is written to the console.

Use this as a starting point for Scanbot-specific tools; the code lives in `source/extensions/scanbot.simple_gui/scanbot/simple_gui/__init__.py`.
