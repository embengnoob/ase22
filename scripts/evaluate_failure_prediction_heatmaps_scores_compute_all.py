import glob
import os
import sys
sys.path.append("..")
import utils
from utils import *
from utils_models import *

import pandas as pd
# from natsort import natsorted
# from colorama import Fore
from heatmap import compute_heatmap
try:
    from config import load_config
except:
    from config import Config

from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction, evaluate_p2p_failure_prediction, get_OOT_frames, test

def simExists(cfg, TESTING_DATA_DIR, SIMULATION_NAME, attention_type):
    SIM_PATH = os.path.join(TESTING_DATA_DIR, SIMULATION_NAME)
    MAIN_CSV_PATH = os.path.join(SIM_PATH, "driving_log.csv")
    HEATMAP_PATH = os.path.join(SIM_PATH, "heatmaps-" + attention_type.lower())
    HM_CSV_PATH = os.path.join(SIM_PATH, "heatmaps-" + attention_type.lower(), "driving_log.csv")
    IMG_PATH = os.path.join(SIM_PATH, "heatmaps-" + attention_type.lower(), "IMG")

    # add field names to main csv file if it's not there (manual training recording)
    csv_df = correct_field_names(MAIN_CSV_PATH)
    # get number of frames
    NUM_OF_FRAMES = pd.Series.max(csv_df['frameId']) + 1 
    if not os.path.exists(SIM_PATH):
        raise ValueError(Fore.RED + f"The provided simulation path does not exist: {SIM_PATH}" + Fore.RESET)

    # 1- IMG folder doesn't exist
    elif (not os.path.exists(IMG_PATH)):
        cprintf(f"IMG folder doesn't exist", 'l_blue')
        cprintf(f"Creating image folder at {IMG_PATH}", 'l_blue')
        os.makedirs(IMG_PATH)
        cprintf(f"Generating heatmaps/CSV file ...", 'l_blue')
        MODE = 'new_calc'
        compute_heatmap(cfg, SIMULATION_NAME, NUM_OF_FRAMES, MODE, attention_type=attention_type)
    # 2- heatmap folder exists, but there are less heatmaps than there should be
    elif len(os.listdir(IMG_PATH)) < NUM_OF_FRAMES:
        cprintf(f"IMG folder exists, but there are less heatmaps than there should be. Generating heatmaps/CSV file ...", 'l_blue')
        cprintf(f"Deleting folder at {IMG_PATH}", 'yellow')
        shutil.rmtree(IMG_PATH)
        MODE = 'new_calc'
        compute_heatmap(cfg, SIMULATION_NAME, NUM_OF_FRAMES, MODE, attention_type=attention_type)
    # 3- heatmap folder exists and correct number of heatmaps, but no csv file was generated.
    elif not 'driving_log.csv' in os.listdir(HEATMAP_PATH):
        cprintf(f"Correct number of heatmaps exist. CSV File doesn't. Generating heatmaps CSV file ...", 'l_blue')
        MODE = 'csv_missing'
        compute_heatmap(cfg, SIMULATION_NAME, NUM_OF_FRAMES, MODE, attention_type=attention_type)
    else:
        cprintf(f"Simulation path and heatmaps for \"{SIMULATION_NAME}\" simulation and \"{attention_type}\" method exist.", 'green')
    return NUM_OF_FRAMES

def correct_field_names(MAIN_CSV_PATH):
    try:
        # autonomous mode simulation data
        data_df = pd.read_csv(MAIN_CSV_PATH)
        data = data_df["center"]
    except:
        # manual mode simulation data
        df = pd.read_csv(MAIN_CSV_PATH, header=None)
        df.rename(columns={ 0: utils.csv_fieldnames_in_manual_mode[0], 1: utils.csv_fieldnames_in_manual_mode[1],
                            2: utils.csv_fieldnames_in_manual_mode[2], 3: utils.csv_fieldnames_in_manual_mode[3],
                            4: utils.csv_fieldnames_in_manual_mode[4], 5: utils.csv_fieldnames_in_manual_mode[5],
                            6: utils.csv_fieldnames_in_manual_mode[6], 7: utils.csv_fieldnames_in_manual_mode[7],
                            8: utils.csv_fieldnames_in_manual_mode[8], 9: utils.csv_fieldnames_in_manual_mode[9],
                            10: utils.csv_fieldnames_in_manual_mode[10]}, inplace=True)
        df['frameId'] = df.index
        df.to_csv(MAIN_CSV_PATH, index=False) # save to new csv file
        data_df = pd.read_csv(MAIN_CSV_PATH)
        data = data_df["center"]
    return data_df

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    try:
        cfg = Config("config_my.py")
    except:
        cfg = load_config("config_my.py")
    # cfg.from_pyfile("config_my.py")

    SIMULATION_NAME_ANOMALOUS = cfg.SIMULATION_NAME
    SIMULATION_NAME_NOMINAL = cfg.SIMULATION_NAME_NOMINAL

    # check whether nominal simulation and the corresponding heatmaps are already generated, generate them otherwise
    NUM_FRAMES_NOM = simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME=SIMULATION_NAME_NOMINAL, attention_type="SmoothGrad")
    NUM_FRAMES_NOM = simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME=SIMULATION_NAME_NOMINAL, attention_type="GradCam++")    
    # check whether the heatmaps are already generated, generate them otherwise
    NUM_FRAMES_ANO = simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME=SIMULATION_NAME_ANOMALOUS, attention_type="SmoothGrad")
    NUM_FRAMES_ANO = simExists(cfg, cfg.TESTING_DATA_DIR, SIMULATION_NAME=SIMULATION_NAME_ANOMALOUS, attention_type="GradCam++")

    heatmap_types = ['smoothgrad'] #gradcam++, smoothgrad
    summary_types = ['-avg', '-avg-grad']
    aggregation_methods = ['mean', 'max']
    # distance_methods = ['pairwise_distance',
    #                     'cosine_similarity',
    #                     'polynomial_kernel',
    #                     'sigmoid_kernel',
    #                     'rbf_kernel',
    #                     'laplacian_kernel'] #'chi2_kernel'
    distance_methods = ['pairwise_distance'] #'chi2_kernel'
    abstraction_methods = ['avg', 'variance']
    # Number between 0 and min(n_samples, n_features)
    pca_dimensions = [100, 500, NUM_FRAMES_ANO-1]

    if cfg.METHOD == 'thirdeye':
        # get number of OOTs
        path = os.path.join(cfg.TESTING_DATA_DIR,
                        SIMULATION_NAME_ANOMALOUS,
                        'heatmaps-' + 'smoothgrad',
                        'driving_log.csv')
        data_df_anomalous = pd.read_csv(path)
        number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
        OOT_anomalous = data_df_anomalous['crashed']
        OOT_anomalous.is_copy = None
        OOT_anomalous_in_anomalous_conditions = OOT_anomalous.copy()
        all_first_frame_position_OOT_sequences = get_OOT_frames(data_df_anomalous, number_frames_anomalous)
        number_of_OOTs = len(all_first_frame_position_OOT_sequences)
        print("identified %d OOT(s)" % number_of_OOTs)

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
                    cprintf(f'\n########### using aggregation method >>{am}<< run number {run_counter} ########### {subplot_counter} ###########', 'yellow')
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
                                                                number_of_OOTs=number_of_OOTs,
                                                                run_counter=run_counter)
        plt.show()
        print(subplot_counter)

    elif cfg.METHOD == 'p2p':
        figsize = (15, 12)
        hspace = 0.69
        fig, axs = plt.subplots(len(abstraction_methods)*len(pca_dimensions)*len(heatmap_types), 1, figsize=figsize)
        plt.subplots_adjust(hspace=hspace)
        plt.suptitle("P2P Heatmap Distances", fontsize=15, y=0.95)
        for ht in heatmap_types:
            for am in abstraction_methods:
                for dim in pca_dimensions:
                    for dm in distance_methods:
                        cprintf(f'\n########### using distance method >>{dm}<< ########### pca dimension {dim} ##############', 'yellow')
                        evaluate_p2p_failure_prediction(cfg,
                                                        heatmap_type=ht,
                                                        heatmap_types = heatmap_types,
                                                        anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                        nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                        distance_method=dm,
                                                        distance_methods = distance_methods,
                                                        pca_dimension=dim,
                                                        pca_dimensions = pca_dimensions,
                                                        abstraction_method = am,
                                                        abstraction_methods = abstraction_methods,
                                                        fig=fig,
                                                        axs=axs)
        plt.show()
    elif cfg.METHOD == 'test':
        for pca_dimension in pca_dimensions:
            cprintb(f'\n########### using distance method >>pairwise_distance<<########### pca dimension {pca_dimension} ##############', 'l_blue')
            test(cfg,
                heatmap_type='smoothgrad',
                anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                distance_method='pairwise_distance',
                distance_type= 'euclidean',
                pca_dimension=pca_dimension) #NUM_FRAMES_ANO-1
