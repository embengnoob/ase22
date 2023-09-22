import os
import sys
from config import Config
from datetime import datetime

def main():

    cfg = Config()
    cfg.from_pyfile("config_my.py")
    # model_path = os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)
    model_path = os.path.join(cfg.SDC_MODELS_DIR, "dave2-mc-053.h5")
    print(f"Model path: {model_path}")
    # Get the creation and modification datetime of the file
    stat_info = os.stat(model_path)
    modification_time = datetime.fromtimestamp(stat_info.st_mtime)
    if modification_time.year < 2023:
        print("Activating udacity-self-driving-car conda environment ...")
        return 1
    else:
        print("Activating saeed_local conda environment ...")
        return 0 

if __name__ == '__main__':
    sys.exit(main())
