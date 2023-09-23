import glob
import os
import sys
sys.path.append("..")
from natsort import natsorted
from heatmap import compute_heatmap
try:
    from config import load_config
except:
    from config import Config
from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction

def simExists(cfg, TESTING_DATA_DIR, SIMULATION_NAME, attention_type):
    SIM_PATH = os.path.join(TESTING_DATA_DIR, SIMULATION_NAME)
    HEATMAP_PATH = os.path.join(SIM_PATH, "heatmaps-" + attention_type.lower())
    if not os.path.exists(SIM_PATH):
        raise ValueError(f"The provided simulation path does not exist: {SIM_PATH}")
    elif (not os.path.exists(HEATMAP_PATH)) or (os.path.exists(HEATMAP_PATH) and len(os.listdir(os.path.join(HEATMAP_PATH, "IMG")))==0):
        print(f"Simulation folder exists. Generating heatmaps ...")
        compute_heatmap(cfg, SIMULATION_NAME, attention_type=attention_type)
    else:
        print(f"Simulation path and heatmap scores for {SIMULATION_NAME} exist.")

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    try:
        cfg = Config("config_my.py")
    except:
        cfg = load_config("config_my.py")
    # cfg.from_pyfile("config_my.py")

    # check whether nominal simulation and the corresponding heatmaps are already generated, generate them otherwise
    simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME="gauss-journal-track1-nominal", attention_type="SmoothGrad")   
    # check whether the heatmaps are already generated, generate them otherwise
    simExists(cfg, cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, attention_type="SmoothGrad")
    

    for ht in ['smoothgrad']:
        for st in ['-avg', '-avg-grad']:
            for am in ['mean', 'max']:
                evaluate_failure_prediction(cfg,
                                            heatmap_type=ht,
                                            simulation_name=cfg.SIMULATION_NAME,
                                            summary_type=st,
                                            aggregation_method=am,
                                            condition='ood')
                                
    # for condition in ['icse20', 'mutants', 'ood']:
    #     simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
    #     for ht in ['smoothgrad']:
    #         for st in ['-avg', '-avg-grad']:
    #             for am in ['mean', 'max']:
    #                 for sim in simulations:
    #                     if "nominal" not in sim:
    #                         sim = sim.replace("simulations/", "")
    #                         if "nominal" not in sim or "Normal" not in sim:
    #                             evaluate_failure_prediction(cfg,
    #                                                         heatmap_type=ht,
    #                                                         simulation_name=sim,
    #                                                         summary_type=st,
    #                                                         aggregation_method=am,
    #                                                         condition=condition)
