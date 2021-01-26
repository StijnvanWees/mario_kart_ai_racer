import os
import subprocess
import pydirectinput
from time import sleep
from timeit import default_timer
from random import randint
import numpy as np
import tensorflow as tf

from src.config import *
from src.player_controllers import player_controllers
from src.screen_recorder import make_screen_shot
from src.policies import policy_manager
from src.detection_models import detection_models
from src.replay_buffer import store_race_in_replay_buffer


def key_tab(btn):
    if type(btn) == str:
        pydirectinput.keyDown(btn)
        pydirectinput.keyUp(btn)
    else:
        for bt in btn:
            pydirectinput.keyDown(bt)
        for bt in btn:
            pydirectinput.keyUp(bt)


def run_mario_kart():
    subprocess.Popen([EMULATOR_PATH + EMULATOR_FILENAME, '-e=' + MARIO_KART_EMULATOR_PATH])
    [player_controller.reset() for player_controller in player_controllers.values()]


def kill_mario_kart():
    [player_controller.reset() for player_controller in player_controllers.values()]
    key_tab('esc')
    key_tab('enter')
    os.system("taskkill /f /im " + EMULATOR_FILENAME)


def start_mario_kart():
    run_mario_kart()
    sleep(17.0)
    player_controllers["1"].menu_tab('a')


def restart_race(previous_race_game_info):
    if previous_race_game_info["has_finished"] == 1:
        sleep(10.0)
        player_controllers["1"].menu_tab('a')
        sleep(2.0)
        player_controllers["1"].menu_tab('a')
        sleep(1.0)
        player_controllers["1"].menu_tab('d')
        sleep(1.0)
        player_controllers["1"].menu_tab('d')
        player_controllers["1"].menu_tab('a')
        player_controllers["1"].menu_tab('l')
        player_controllers["1"].menu_tab('a')
    else:
        key_tab('a')
        sleep(1.0)
        player_controllers["1"].menu_tab('d')
        player_controllers["1"].menu_tab('a')
        player_controllers["1"].menu_tab('l')
        player_controllers["1"].menu_tab('a')


def start_mario_kart_race():
    sleep(5.0)
    player_controllers["1"].menu_tab('a')
    player_controllers["1"].menu_tab('a')
    player_controllers["1"].menu_tab('a')
    player_controllers["1"].menu_tab('a')

    [player_controllers["1"].menu_tab('r') for _ in range(2)]  # cc
    player_controllers["1"].menu_tab('a')

    # First driver
    player_controllers["1"].menu_tab('d')
    [player_controllers["1"].menu_tab('r') for _ in range(DRIVER_1)]
    player_controllers["1"].menu_tab('a')
    # Second driver
    player_controllers["1"].menu_tab('d')
    [player_controllers["1"].menu_tab('r') for _ in range(DRIVER_2)]
    player_controllers["1"].menu_tab('a')
    # Car
    [player_controllers["1"].menu_tab('r') for _ in range(CAR)]
    player_controllers["1"].menu_tab('a')

    sleep(1.0)

    if FORCE_TOURNAMENT is None:
        tournament_id = randint(0, 3)
    else:
        tournament_id = FORCE_TOURNAMENT

    [player_controllers["1"].menu_tab('r') for i in range(tournament_id)]
    player_controllers["1"].menu_tab('a')

    player_controllers["1"].menu_tab('a')

    return {"tournament_id": tournament_id}


def go_to_next_race():
    for i in range(3):
        sleep(3.0)
        player_controllers["1"].menu_tab('a')

    sleep(2.0)


def complete_race(game_info, state_datas, actions, frames):
    [player_controller.reset() for player_controller in player_controllers.values()]
    game_info = store_race_in_replay_buffer(game_info, state_datas, actions)

    print("fps count:", float(frames) / game_info["race_time"])
    print(game_info)


def process_race(game_info):
    state_datas = []
    actions = []

    state_data = None

    if game_info["court_id"] > 0:
        go_to_next_race()

    sleep(3.0)
    player_controllers["1"].key_tab('a')

    sleep(6.65)
    race_time = default_timer()

    player_controllers["1"].perform_action([[0, 1, 0], [1, 0, 0]])

    detection_models.reset_for_race()

    frames = 0

    while 1:
        t = default_timer()
        frames += 1

        if USE_HUMAN_PLAYER_POLICY:
            actions.append(policy_manager.next_action(state_data))

        if ULTIMATE_PERFORMANCE_MODE:
            state_data = np.reshape(make_screen_shot((180, 135)), (1, 135, 180, 3))
        else:
            screenshot = make_screen_shot((360, 270))
            state_data = detection_models.get_state_data(screenshot)
            state_datas.append(np.array(state_data[0], dtype=np.float16))

        if not USE_HUMAN_PLAYER_POLICY:
            action = policy_manager.next_action(state_data)
            actions.append(action)
            player_controllers["1"].perform_action(action)

        game_info["race_time"] = default_timer() - race_time

        end_race_for_finish = detection_models.detect_goal(state_data)

        if ULTIMATE_PERFORMANCE_MODE or USE_HUMAN_PLAYER_POLICY:
            end_race_for_max_duration = False
            end_race_for_last_for_frames = False
        else:
            end_race_for_max_duration = game_info["race_time"] > MAX_RACE_DURATION
            end_race_for_last_for_frames = detection_models.detect_end_last_positions(screenshot)

        if game_info["skip_replay_buffer_data_storage"]:
            end_race_for_last_for_frames = False

        if end_race_for_finish or end_race_for_max_duration or end_race_for_last_for_frames:
            game_info["has_finished"] = 0
            if end_race_for_finish:
                game_info["has_finished"] = 1
            break

        if not ULTIMATE_PERFORMANCE_MODE:
            sleep(max(0.0, FRAME_TIME - (default_timer() - t)))

    complete_race(game_info, state_datas, actions, frames)
