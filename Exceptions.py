## TODO: Extend and improve exceptions

class InvalidOperation(Exception):
    def __init__(self, message):
        self.message = message


class InitializationError(Exception):
    def __init__(self, message):
        self.message = message


class InvalidGate(Exception):
    def __init__(self, message):
        self.message = message
