# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import mediapy as media
import mujoco
import predictive_sampling
import numpy as np


# %%
xml = '/home/max/workspace/research_template/code/gym/mjpc_tests/xmls/particle.xml'

# create simulation model + data
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
#renderer = mujoco.Renderer(model)


# %%
# reward
def reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  # position
  goal = data.mocap_pos[0, :2]
  pos_error = data.qpos - goal
  r0 = -np.dot(pos_error, pos_error)

  # velocity
  r1 = -np.dot(data.qvel, data.qvel)

  # effort
  r2 = -np.dot(data.ctrl, data.ctrl)

  return 5.0 * r0 + 0.1 * r1 + 0.1 * r2


# %%
# planner
horizon = 0.5
splinestep = 0.1
planstep = 0.025
nimprove = 4
nsample = 4
noise_scale = 0.01
interp = "zero"
planner = predictive_sampling.Planner(
    model,
    reward,
    horizon,
    splinestep,
    planstep,
    nsample,
    noise_scale,
    nimprove,
    interp=interp,
)
# %%
# simulate
mujoco.mj_resetData(model, data)
steps = 301

# set goal position
data.mocap_pos[0, :2] = np.array([0.3, 0.0])

# history
qpos = [data.qpos]
qvel = [data.qvel]
act = [data.act]
ctrl = []
rewards = []

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# verbose
VERBOSE = True

for _ in range(steps):
  ## predictive sampling

  # improve policy
  planner.improve_policy(
      data.qpos, data.qvel, data.act, data.time, data.mocap_pos, data.mocap_quat
  )

  # get action from policy
  data.ctrl = planner.action_from_policy(data.time)

  # reward
  rewards.append(reward(model, data))

  if VERBOSE:
    print("time  : ", data.time)
    print(" qpos  : ", data.qpos)
    print(" qvel  : ", data.qvel)
    print(" act   : ", data.act)
    print(" action: ", data.ctrl)
    print(" reward: ", rewards[-1])

  # step
  mujoco.mj_step(model, data)

  # history
  qpos.append(data.qpos)
  qvel.append(data.qvel)
  act.append(data.act)
  ctrl.append(ctrl)

  ## render and save frames
  #renderer.update_scene(data)
  #pixels = renderer.render()
  #frames.append(pixels)

if VERBOSE:
  print("\nfinal qpos: ", qpos[-1])
  print("goal state : ", data.mocap_pos[0, 0:2])
  print("state error: ", np.linalg.norm(qpos[-1][0:2] - data.mocap_pos[0, 0:2]))
# %%
#media.show_video(frames, fps=FPS)