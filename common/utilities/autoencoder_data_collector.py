import hashlib
import numpy as np
class AutoencoderDataCollector:

    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path

    def collect_data(self, state, n_state):
        # Str to sha256
        state_hash = str(hashlib.sha256(str(state).encode('utf-8')).hexdigest())
        n_state_hash = str(hashlib.sha256(str(n_state).encode('utf-8')).hexdigest())
        # Store numpy state array into file
        np.save(self.folder_path + "/" + state_hash+".npy", state)
        np.save(self.folder_path + "/" + n_state_hash+".npy", n_state)
