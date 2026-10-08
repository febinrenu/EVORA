"""`make up`: check the machine and start evora (API and built UI). Run with the backend environment."""
import sys

from evora.core.launcher import main

if __name__ == "__main__":
    sys.exit(main())
