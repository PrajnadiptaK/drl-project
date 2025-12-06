import datetime

class Logger:
    def __init__(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"{filename}_{timestamp}.log"
        self.file = open(self.path, "w")
        self.write("=== Logging Started ===")

    def write(self, text):
        self.file.write(text + "\n")
        self.file.flush()  # ensures writes even if script crashes

    def close(self):
        self.write("=== Logging Finished ===")
        self.file.close()
