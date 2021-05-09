import pandas as pd

# Modules expanding time details, including:
# Start{Month| Day | Hour}
# Weekend(binary)
# Duration
# End {Month | Day | Hour}

class time_expand:
    def __init__(self, filepath):
        self.path = str(filepath)
        df = pd.open_csv(self.path)