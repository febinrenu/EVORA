"""`make doctor`: is this machine ready to demo? Run with the backend environment (see the Makefile)."""
import sys

from evora.core.doctor import main

if __name__ == "__main__":
    sys.exit(main())
