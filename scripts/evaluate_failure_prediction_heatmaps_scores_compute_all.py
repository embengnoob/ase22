import glob
import os
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from heatmap import compute_heatmap
try:
    from config import load_config
except:
    from config import Config
from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction, get_crash_frames

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

    SIMULATION_NAME_ANOMALOUS = cfg.SIMULATION_NAME
    SIMULATION_NAME_NOMINAL = "gauss-journal-track1-nominal"

    # check whether nominal simulation and the corresponding heatmaps are already generated, generate them otherwise
    simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME=SIMULATION_NAME_NOMINAL, attention_type="SmoothGrad")   
    # check whether the heatmaps are already generated, generate them otherwise
    simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME=SIMULATION_NAME_ANOMALOUS, attention_type="SmoothGrad")
    
    # get number of crashes
    path = os.path.join(cfg.TESTING_DATA_DIR,
                    SIMULATION_NAME_ANOMALOUS,
                    'heatmaps-' + 'smoothgrad',
                    'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)
    number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()
    all_first_frame_position_crashed_sequences = get_crash_frames(data_df_anomalous, number_frames_anomalous)
    number_of_crashes = len(all_first_frame_position_crashed_sequences)
    print("identified %d crash(es)" % number_of_crashes)

    heatmap_types = ['smoothgrad']
    summary_types = ['-avg', '-avg-grad']
    aggregation_methods = ['mean', 'max']

    if len(aggregation_methods) == 3:
        figsize = (15, 12)
        hspace = 0.69
    elif len(aggregation_methods) == 2:
        figsize = (15, 10)
        hspace = 0.44
    else:
        raise ValueError("No predefined settings for this number of aggregation methods.")
    
    fig, axs = plt.subplots(len(aggregation_methods)*2, 1, figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    plt.suptitle("Heatmap scores and thresholds", fontsize=15, y=0.95)

    run_counter = 0
    subplot_counter = 0
    for ht in heatmap_types:
        for st in summary_types:
            for am in aggregation_methods:
                run_counter += 1
                print(f'\n########### using aggregation method >>{am}<< run number {run_counter} ########### {subplot_counter} ####################################################################################################################################################################################################################################################################################################################')
                subplot_counter = evaluate_failure_prediction(cfg,
                                                            heatmap_type=ht,
                                                            anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                            nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                            summary_type=st,
                                                            aggregation_method=am,
                                                            condition='ood',
                                                            fig=fig,
                                                            axs=axs,
                                                            subplot_counter=subplot_counter,
                                                            number_of_crashes=number_of_crashes,
                                                            run_counter=run_counter)
    plt.show()
    print(subplot_counter)                 
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
