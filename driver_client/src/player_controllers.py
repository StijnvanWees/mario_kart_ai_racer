import pydirectinput
from time import sleep
import numpy as np


MENU_KEY_TIMEOUT = 0.5


class PlayerKeyMap:
    def __init__(self, player_id, a, b, x, down, left, right, trigger_l):
        '''
        :param player_id: id int
        :other params: pydirectinput keys as configured in Dolphin emulator
        '''
        self.player_id = player_id
        self.key_map = {}
        self.key_map["a"] = a
        self.key_map["b"] = b
        self.key_map["x"] = x
        self.key_map["l"] = left
        self.key_map["r"] = right
        self.key_map["d"] = down
        self.key_map["tl"] = trigger_l

    def get_player_id(self):
        return self.player_id

    def get_key(self, gc_key):
        if type(gc_key) == str:
            return self.key_map[gc_key]
        else:
            return [self.key_map[key] for key in gc_key]

    def get_key_map(self):
        return self.key_map


class KeyboardActions:
    def __init__(self):
        pass

    def key_tab(self, btn):
        if type(btn) == str:
            pydirectinput.keyDown(btn)
            pydirectinput.keyUp(btn)
        else:
            for bt in btn:
                pydirectinput.keyDown(bt)
            for bt in btn:
                pydirectinput.keyUp(bt)

    def key_press(self, btn):
        pydirectinput.keyDown(btn)

    def key_release(self, btn):
        pydirectinput.keyUp(btn)


class PlayerController(PlayerKeyMap, KeyboardActions):
    def __init__(self, player_id, a, b, x, down, left, right, trigger_l):
        PlayerKeyMap.__init__(self, player_id, a, b, x, down, left, right, trigger_l)
        KeyboardActions.__init__(self)

        self.keys_held = []
        self.last_action = [0, 0]

    def get_keys_held(self):
        return self.keys_held

    def key_tab(self, gc_key):
        KeyboardActions.key_tab(self, self.get_key(gc_key))

    def key_press(self, gc_key):
        if gc_key in self.keys_held:
            return

        self.keys_held.append(gc_key)
        KeyboardActions.key_press(self, self.get_key(gc_key))

    def key_release(self, gc_key):
        if gc_key not in self.keys_held:
            return

        KeyboardActions.key_release(self, self.get_key(gc_key))
        self.keys_held.remove(gc_key)

    def keys_release(self, gc_keys):
        [self.key_release(gc_key) for gc_key in gc_keys]

    def menu_tab(self, gc_key, time_out=None):
        self.key_tab(gc_key)
        if not time_out:
            sleep(MENU_KEY_TIMEOUT)
        else:
            sleep(time_out)

    def reset(self):
        [KeyboardActions.key_release(self, k) for k in self.get_key_map().values()]

        self.keys_held = []
        self.last_action = [0, 0]

    def perform_action(self, action):
        self.key_press("a")

        if np.argmax(self.last_action[0]) != np.argmax(action[0]):
            self.keys_release(["l", "r"])
            if action[0][0] == 1:
                self.key_press("l")
            elif action[0][2] == 1:
                self.key_press("r")

        if action[1][1] == 1:
            self.key_tab(["x", "d"])
        if action[1][2] == 1:
            self.key_tab("x")

        self.last_action = action


player_controllers = {}
player_controllers["1"] = PlayerController(1, 'x', 'z', 'c', 'down', 'left', 'right', 'q')
#player_controllers["2"] = PlayerController(2, 'r', 't', 'y', 'u', 'i', 'o', 'p')
#player_controllers["3"] = PlayerController(3, 'f', 'g', 'h', 'j', 'k', 'l', ';')
#player_controllers["4"] = PlayerController(4, 'v', 'b', 'n', 'm', ',', '.', '/')
