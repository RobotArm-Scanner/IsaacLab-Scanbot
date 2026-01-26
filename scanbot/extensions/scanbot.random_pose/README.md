# Scanbot Random Pose

Tiny `omni.ui` panel with a single **Random!** button that generates a random action for the active gym env (via `scanbot_context`) and applies it on the next update tick (`env.step`). No periodic stepping is done unless a button-triggered action is pending.

## Quick start
- Launch Isaac Lab (or Kit) with this repo as an extension folder:
  - `./isaaclab.sh --ext-folder /home/scanbot/dev/isaac_lab_scanbot/scanbot/source/extensions --enable scanbot.random_pose`
- Run through `scanbot_launcher.py` (registers the env into `scanbot_context`).
- Click **Random!**. A random action is applied on the next update tick. If you need continuous stepping, restore it in the launcher loop.

Code lives in `scanbot/source/extensions/scanbot.random_pose/scanbot/random_pose/__init__.py`.
