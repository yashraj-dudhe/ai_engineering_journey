import sys
import datetime

def init_check():
    print(f"✅ Python executable: {sys.executable}")
    print(f"✅ Current Time: {datetime.datetime.now()}")
    print("🚀 System is locked in. Let's grind.")

if __name__ == "__main__":
    init_check()