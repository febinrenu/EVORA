"""Process-wide locks shared by the perception layers."""
import threading

# creating a LanceDB table is not safe against another thread creating the same one
STORE_SETUP = threading.Lock()
