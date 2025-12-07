import os
import datetime

class Logger:
    def __init__(self, name):
        """
        name: short identifier for the log
               e.g., train_colight, fixed_baseline, eval_independent
        """
        # ensure logs folder exists
        os.makedirs("logs", exist_ok=True)

        # timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # full file path
        self.path = f"logs/{name}_{timestamp}.log"

        # open file
        self.file = open(self.path, "w")
        self.write("=== Logging Started ===")

    def write(self, text):
        """Append a line of log text"""
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        self.write("=== Logging Finished ===")
        self.file.close()
