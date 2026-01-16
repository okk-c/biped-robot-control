import numpy as np
from isaacgym import gymapi

class KeyboardCmd:
    def __init__(self, env):
        self.env = env
        self.gym = env.gym
        self.viewer = env.viewer

        self.command = np.zeros(3)

        # 订阅键盘事件
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "turn_left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "turn_right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "stop")

        print("✅ Keyboard control enabled:")
        print("   W/S → 前进 / 后退")
        print("   A/D → 左右侧移")
        print("   Q/E → 左右转向")
        print("   SPACE → 停止")

    def get_command(self):
        # 每帧清零
        self.command[:] = 0.0

        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0:
                if evt.action == "forward":
                    self.command[0] = 1.0
                elif evt.action == "backward":
                    self.command[0] = -1.0
                elif evt.action == "left":
                    self.command[1] = 1.0
                elif evt.action == "right":
                    self.command[1] = -1.0
                elif evt.action == "turn_left":
                    self.command[2] = 1.0
                elif evt.action == "turn_right":
                    self.command[2] = -1.0
                elif evt.action == "stop":
                    self.command[:] = 0.0

        return self.command
