class KronFES:
    def __init__(self, fes_time, fes_space):
        self.time = fes_time
        self.space = fes_space
        self.time.fd = [
            i for (i, free) in enumerate(fes_time.FreeDofs()) if free
        ]
        self.space.fd = [
            i for (i, free) in enumerate(fes_space.FreeDofs()) if free
        ]
