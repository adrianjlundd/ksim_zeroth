"""Simple robot pose viewer without timeout issues."""
import mujoco
import mujoco.viewer
import numpy as np

# Joint biases - CORRECT ORDER (arms first, then legs)
JOINT_BIASES = [
    # Arms first (qposadr 7-12)
    ("right_shoulder_pitch", 0.3),   # arms forward symmetric
    ("right_shoulder_yaw", 0.0),
    ("right_elbow_yaw", 0.0),
    ("left_shoulder_pitch", 0.3),    # arms forward symmetric
    ("left_shoulder_yaw", 0.0),
    ("left_elbow_yaw", 0.0),
    # Legs second (qposadr 13-22)
    ("right_hip_pitch", 0.2),      # slight forward lean for balance
    ("right_hip_yaw", 0.0),
    ("right_hip_roll", -0.785),    # push outward (-45 degrees) - CORRECT
    ("right_knee_pitch", -3.14),   # PERFECT - DO NOT TOUCH
    ("right_ankle_pitch", 0.4),    # ~23 degrees up to flatten foot
    ("left_hip_pitch", 0.2),       # slight forward lean for balance
    ("left_hip_yaw", -0.2),        # NEGATIVE to rotate outward
    ("left_hip_roll", 0.785),      # push outward (+45 degrees) - CORRECT
    ("left_knee_pitch", 0.0),      # PERFECT - DO NOT TOUCH
    ("left_ankle_pitch", 0.0),
]

# Load model with floor scene
import mujoco_scenes
import mujoco_scenes.mjcf

model = mujoco_scenes.mjcf.load_mjmodel("meshes/robot.mjcf", scene="smooth")
data = mujoco.MjData(model)

# Set joint positions to reference pose
print("\n" + "="*60)
print("[INSPECT MODE - Standalone Viewer]")

# First, print actual joint names from the model
print(f"Model has {model.njnt} joints:")
for i in range(model.njnt):
    jnt_id = i
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    jnt_type = model.jnt_type[i]
    qposadr = model.jnt_qposadr[i]
    print(f"  Joint {i}: {jnt_name} (type={jnt_type}, qposadr={qposadr})")
print("="*60)

print("\nSetting robot to reference pose:")

print("\n*** VERIFYING JOINT ORDER ***")
print("Joint 7 (qposadr 7):", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 1))
print("Joint 8 (qposadr 8):", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 2))
print("Joint 13 (qposadr 13):", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 7))
print("Joint 14 (qposadr 14):", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 8))
print("Joint 19 (qposadr 19):", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 13))
print("="*60 + "\n")

# Set base position (x, y, z)
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.45  # higher for more upright stance

# Set base orientation quaternion (w, x, y, z) - upright
data.qpos[3] = 1.0  # w
data.qpos[4] = 0.0  # x
data.qpos[5] = 0.0  # y
data.qpos[6] = 0.0  # z

# Set joint positions (skip 7 DoFs: 3 pos + 4 quat for floating base)
for i, (name, value) in enumerate(JOINT_BIASES):
    data.qpos[7 + i] = value
    print(f"  qpos[{7+i}] = {name:25s}: {value:7.4f} rad ({np.degrees(value):7.2f}°)")
print("="*60 + "\n")

# Forward kinematics
mujoco.mj_forward(model, data)

# Reset velocities to zero
data.qvel[:] = 0.0

# Launch viewer
print("Launching MuJoCo viewer...")
print("Controls:")
print("  - Right-click + drag: Rotate camera")
print("  - Shift + Right-click + drag: Pan camera")
print("  - Scroll wheel: Zoom")
print("  - Double-click: Select body")
print("  - Ctrl+P: Screenshot")
print("  - ESC: Close viewer\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)  # Physics enabled
        viewer.sync()
