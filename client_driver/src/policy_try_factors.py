from src.config import *
from src.db_registry import registry
from src.policies import policy_manager


def update_policy_try_factor(game_info, loop_id, total_loops):
    if ULTIMATE_PERFORMANCE_MODE:
        return 0.0

    policy_try_factor = (total_loops - loop_id) / total_loops
    #policy_try_factor **= 1.5
    policy_try_factor *= registry.get_court_settings_object(game_info).get_max_policy_try_factor()

    policy_manager.set_policy_try_factor(policy_try_factor)

    game_info["policy_try_factor"] = policy_try_factor

    return game_info
