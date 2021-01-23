import sqlite3

from src.config import *

import pickle
from os import listdir


def process_db_request(handler, args):
    connection = sqlite3.connect(REGISTRY_DB_FILEPATH)
    cursor = connection.cursor()

    returns = handler(connection, cursor, *args)

    connection.close()

    return returns


class QModelObject:
    def __init__(self, data):
        self.id = data[0]
        self.date_time_created = data[1]
        self.description = data[2]

    def get_id(self):
        return self.id

    def get_date_time_created(self):
        return self.date_time_created

    def get_description(self):
        return self.description


class RaceObject:
    def __init__(self, data):
        self.id = data[0]
        self.active_q_model_id = data[1]
        self.tournament_id = data[2]
        self.court_id = data[3]
        self.human_controlled = data[4]
        self.policy_try_factor = data[5]
        self.frame_count = data[6]
        self.race_time = data[7]
        self.has_finished = data[8]
        self.finished_position = data[9]
        self.ultimate_performance_mode = data[10]
        self.skip_replay_buffer_storage = data[11]

    def get_id(self):
        return self.id

    def get_active_q_model_id(self):
        return self.active_q_model_id

    def get_tournament_id(self):
        return self.tournament_id

    def get_court_id(self):
        return self.court_id

    def get_human_controlled(self):
        return self.human_controlled

    def get_policy_try_factor(self):
        return self.policy_try_factor

    def get_frame_count(self):
        return self.frame_count

    def get_race_time(self):
        return self.race_time

    def get_has_finished(self):
        return self.has_finished

    def get_finished_position(self):
        return self.finished_position

    def get_ultimate_performance_mode(self):
        return self.ultimate_performance_mode

    def get_skip_replay_buffer_storage(self):
        return self.skip_replay_buffer_storage


class CourtSettingsObject:
    def __init__(self, data):
        self.tournament_id = data[0]
        self.court_id = data[1]
        self.max_policy_try_factor = data[2]
        self.skip_replay_buffer_storage = data[3]

    def get_tournament_id(self):
        return self.tournament_id

    def get_court_id(self):
        return self.court_id

    def get_max_policy_try_factor(self):
        return self.max_policy_try_factor

    def get_skip_replay_buffer_storage(self):
        return self.skip_replay_buffer_storage


def add_q_model_handler(connection, cursor, id_, date_time_created, description):
    t = (id_, date_time_created, description)
    cursor.execute('INSERT INTO q_models VALUES (?, ?, ?)', t)
    connection.commit()


def add_race_data_handler(connection, cursor, game_info):
    human_controlled = 0
    if game_info["human_controlled"]:
        human_controlled = 1

    ultimate_performance_mode = 0
    if game_info["ultimate_performance_mode"]:
        ultimate_performance_mode = 1

    t = (game_info["race_id"], game_info["active_q_model_id"],
         game_info["tournament_id"], game_info["court_id"],
         human_controlled, game_info["policy_try_factor"],
         game_info["frame_count"], game_info["race_time"],
         game_info["has_finished"], int(game_info["finished_position"]),
         ultimate_performance_mode, game_info["skip_replay_buffer_data_storage"])
    cursor.execute('INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', t)
    connection.commit()


def get_lastest_q_model_object_handler(connection, cursor):
    cursor.execute('SELECT * FROM q_models ORDER BY date_time_created DESC')
    return QModelObject(cursor.fetchall()[0])


def get_race_objects_handler(connection, cursor, ultimate_performance_mode_entries):
    ultimate_performance_mode_int = 0
    if ultimate_performance_mode_entries:
        ultimate_performance_mode_int = 1

    t = (ultimate_performance_mode_int,)
    cursor.execute('SELECT * FROM races WHERE ultimate_performance_mode == ?', t)
    datas = cursor.fetchall()
    return [RaceObject(data) for data in datas]


def get_race_objects_in_replay_buffer_handler(connection, cursor):
    cursor.execute('SELECT * FROM races WHERE skip_replay_buffer_data_storage == 0')
    datas = cursor.fetchall()
    return [RaceObject(data) for data in datas]


def get_court_setting_handler(connection, cursor, game_info):
    t = (game_info["tournament_id"], game_info["court_id"])
    cursor.execute('SELECT * FROM court_settings WHERE tournament_id == ? and court_id == ?', t)
    return CourtSettingsObject(cursor.fetchone())


class Registry:
    def __init__(self):
        pass

    def add_q_model(self, id_, date_time_created, description):
        return process_db_request(add_q_model_handler, [id_, date_time_created, description])

    def add_race(self, game_info):
        return process_db_request(add_race_data_handler, [game_info])

    def get_latest_q_model_object(self):
        return process_db_request(get_lastest_q_model_object_handler, [])

    def get_race_data_objects(self, ultimate_performance_mode_entries=False):
        return process_db_request(get_race_objects_handler, [ultimate_performance_mode_entries])

    def get_race_data_objects_in_replay_buffer(self):
        return process_db_request(get_race_objects_in_replay_buffer_handler, [])

    def get_court_settings_object(self, game_info):
        return process_db_request(get_court_setting_handler, [game_info])


registry = Registry()
