import socket
from datetime import datetime
from config.hyper_parameters import *


def get_new_id():
    return socket.gethostname() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
