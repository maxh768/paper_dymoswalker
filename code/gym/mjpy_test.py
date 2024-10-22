import mujoco
import glfw
import numpy as np
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                       title='Demo', monitor=None,
                                       share=None)
    glfw.make_context_current(window)
    return window

window = init_window(2400, 1800)
width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

xml = "/home/max/workspace/research_template/code/gym/inverted_pendulum.xml"

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

scene = mujoco.MjvScene(model, 6000)
camera = mujoco.MjvCamera()
camera.trackbodyid = 2
camera.distance = 3
camera.azimuth = 90
camera.elevation = -20
mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

Kp = 5
Kd = 100
data.qpos = np.array([-1, np.deg2rad(3)])

dummy_state = np.zeros((1,4))

sensor_states = np.zeros((1,4))
sensor_states[0,:] = (data.sensordata)

real_states = np.zeros((1,4))
real_states[0,:] = np.block([[data.qpos, data.qvel]])

time = np.zeros((1,1))
time[0,0] = data.time
dummy_time = np.zeros((1,1))

target_x = 0
target_theta = np.deg2rad(0)


while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)

    data.ctrl = np.zeros(1) + Kd*(np.zeros(1) + data.qvel[1]) + Kp*(target_theta + data.qpos[1])
    # data.ctrl = np.zeros(2) + Kd*(np.zeros(2) - data.sensordata[2:3]) + Kp*(np.zeros(2) - data.sensordata[:2])
    print(data.ctrl)
    mujoco.mj_step2(model, data)

    dummy_state[0,:] = (data.sensordata)
    sensor_states = np.append(sensor_states, dummy_state, axis=0)
    
    dummy_state[0,:] = np.block([[data.qpos, data.qvel]])
    real_states = np.append(real_states, dummy_state, axis=0)
    
    dummy_time[0,0] = data.time
    time = np.append(time, dummy_time, axis=0)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

plt.plot(time[:,0], sensor_states[:,1], color='b', label='sensor')
plt.plot(time[:,0], real_states[:,1], color='r', label='real')
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Sine and Cosine functions")
plt.legend()
plt.show()