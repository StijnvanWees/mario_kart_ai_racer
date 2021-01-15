

class SharedProcessData:
    def __init__(self):
        self.mp = None
        self.q = None

        self.process = None

    def set(self, mp):
        self.mp = mp
        self.q = mp.Queue()

    def set_process(self, process):
        self.process = process

    def join_process(self):
        if not self.process:
            return

        self.process.join()
        self.process = None

    def get_mp(self):
        return self.mp

    def get_q(self):
        return self.q

    def get_process(self):
        return self.process


shared_process_data = SharedProcessData()
