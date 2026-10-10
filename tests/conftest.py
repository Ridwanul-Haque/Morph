import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import morph.contact_plan  # noqa: E402

# grasp plans the tests make stay out of the app's own cache (runs/programs/grasps)
morph.contact_plan.GRASP_CACHE = Path(tempfile.mkdtemp(prefix="morph_grasps_"))
