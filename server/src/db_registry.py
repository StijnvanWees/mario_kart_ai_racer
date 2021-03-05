from src.config import *

import sys
sys.path.append('..')
from shared.db_registry import Registry


registry = Registry(REGISTRY_DB_FILEPATH)
