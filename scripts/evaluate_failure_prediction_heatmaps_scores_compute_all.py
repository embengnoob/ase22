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

def simExists(cfg, run_id, sim_name, attention_type, nominal):
    SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, sim_name)
    MAIN_CSV_PATH = os.path.join(SIM_PATH, "driving_log.csv")
    HEATMAP_PARENT_FOLDER_PATH = os.path.join(SIM_PATH, "heatmaps-" + attention_type.lower())
    if not nominal:
        HEATMAP_FOLDER_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id)
        HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "driving_log.csv")
        HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "IMG")
    else:
        HEATMAP_FOLDER_PATH = HEATMAP_PARENT_FOLDER_PATH
        HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "driving_log.csv")
        HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "IMG")

    # add field names to main csv file if it's not there (manual training recording)
    csv_df = correct_field_names(MAIN_CSV_PATH)
    # get number of frames
    NUM_OF_FRAMES = pd.Series.max(csv_df['frameId']) + 1 
    if not os.path.exists(SIM_PATH):
        raise ValueError(Fore.RED + f"The provided simulation path does not exist: {SIM_PATH}" + Fore.RESET)

    # 1- IMG folder doesn't exist
    elif (not os.path.exists(HEATMAP_IMG_PATH)):
        cprintf(f"Heatmap IMG folder doesn't exist. Creating image folder at {HEATMAP_IMG_PATH}", 'l_blue')
        os.makedirs(HEATMAP_IMG_PATH)
        MODE = 'new_calc'
        compute_heatmap(cfg, nominal, sim_name, NUM_OF_FRAMES, MODE, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH)

    # 2- heatmap folder exists, but there are less heatmaps than there should be
    elif len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES:
        cprintf(f"Heatmap IMG folder exists, but there are less heatmaps than there should be.", 'yellow')
        cprintf(f"Deleting folder at {HEATMAP_IMG_PATH}", 'yellow')
        shutil.rmtree(HEATMAP_IMG_PATH)
        MODE = 'new_calc'
        compute_heatmap(cfg, nominal, sim_name, NUM_OF_FRAMES, MODE, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH)
    # 3- heatmap folder exists and correct number of heatmaps, but no csv file was generated.

    elif not 'driving_log.csv' in os.listdir(HEATMAP_FOLDER_PATH):
        cprintf(f"Correct number of heatmaps exist. CSV File doesn't.", 'yellow')
        MODE = 'csv_missing'
        compute_heatmap(cfg, nominal, sim_name, NUM_OF_FRAMES, MODE, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH)
    else:
        if nominal:
            cprintf(f"Heatmaps for nominal sim \"{sim_name}\" of attention type \"{attention_type}\" exist.", 'l_green')
        else:
            cprintf(f"Heatmaps for anomalous sim \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" exist.", 'l_green')

    PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH]
    return NUM_OF_FRAMES, PATHS

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
    start_time = time.monotonic()
    os.chdir(os.getcwd().replace('scripts', ''))

    try:
        cfg = Config("config_my.py")
    except:
        cfg = load_config("config_my.py")
    # cfg.from_pyfile("config_my.py")

    ANO_SIMULATIONS = ['test1', 'test2', 'test3', 'test4', 'test5', 'track1-sunny-positioned-nominal-as-anomalous'] # , 'test2', 'test3', 'test4', 'test5'
    NOM_SIMULATIONS = ['track1-sunny-positioned-nominal',
                       'track1-sunny-positioned-nominal',
                       'track1-sunny-positioned-nominal',
                       'track1-sunny-positioned-nominal',
                       'track1-sunny-positioned-nominal',
                       'track1-sunny-positioned-nominal']
    RUN_ID_NUMBERS = [[1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3]]
    SUMMARY_COLLAGES = [[False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False]]
    
    if len(ANO_SIMULATIONS) != len(NOM_SIMULATIONS):
        raise ValueError(Fore.RED + f"Mismatch in number of specified ANO and NOM simulations: {len(ANO_SIMULATIONS)} != {len(NOM_SIMULATIONS)} " + Fore.RESET)
    elif len(ANO_SIMULATIONS) != len(RUN_ID_NUMBERS):
        raise ValueError(Fore.RED + f"Mismatch in number of runs and specified simulations: {len(ANO_SIMULATIONS)} != {len(RUN_ID_NUMBERS)} " + Fore.RESET)
    elif len(SUMMARY_COLLAGES) != len(RUN_ID_NUMBERS):
        raise ValueError(Fore.RED + f"Mismatch in number of runs and specified summary collage patterns: {len(SUMMARY_COLLAGES)} != {len(RUN_ID_NUMBERS)} " + Fore.RESET)
    
    total_runs = 0
    for idx, run_pattern in enumerate(RUN_ID_NUMBERS):
        total_runs += len(run_pattern)
        if len(run_pattern) != len(SUMMARY_COLLAGES[idx]):
            raise ValueError(Fore.RED + f"Mismatch in number of runs per simlation and specified summary collage binary pattern of simulation {idx}: {len(run_pattern)} != {len(SUMMARY_COLLAGES[idx])} " + Fore.RESET) 
    
    # Starting evaluation
    for sim_idx, sim_name in enumerate(ANO_SIMULATIONS):
        for run_number in range(len(RUN_ID_NUMBERS[sim_idx])):
            # Check if a simulation with this run number already exists
            run_id = RUN_ID_NUMBERS[sim_idx][run_number]
            cprintb(f'\n\n########### Simulation {sim_name} ########### run number {run_number+1} ############## run id {run_id} ##############', 'l_red')
            SIMULATION_NAME_ANOMALOUS = sim_name
            SIMULATION_NAME_NOMINAL = NOM_SIMULATIONS[sim_idx]
            cfg.SIMULATION_NAME = SIMULATION_NAME_ANOMALOUS
            cfg.SIMULATION_NAME_NOMINAL = SIMULATION_NAME_NOMINAL
            cfg.GENERATE_SUMMARY_COLLAGES = SUMMARY_COLLAGES[sim_idx][run_number]
        
            HEATMAP_TYPES = ['SmoothGrad'] #GradCam++, SmoothGrad
            summary_types = ['-avg', '-avg-grad']
            aggregation_methods = ['mean', 'max']
            # distance_methods = ['pairwise_distance',
            #                     'cosine_similarity',
            #                     'polynomial_kernel',
            #                     'sigmoid_kernel',
            #                     'rbf_kernel',
            #                     'laplacian_kernel'] #'chi2_kernel'
            DISTANCE_METHODS = ['pairwise_distance']
            DISTANCE_TYPES = ['euclidean']
            abstraction_methods = ['avg', 'variance']

            # check whether nominal and anomalous simulation and the corresponding heatmaps are already generated, generate them otherwise
            for attention_type in HEATMAP_TYPES:
                NUM_FRAMES_NOM, NOMINAL_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_NOMINAL, attention_type=attention_type, nominal=True) 
                NUM_FRAMES_ANO, ANOMALOUS_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_ANOMALOUS, attention_type=attention_type, nominal=False)

            # Number between 0 and min(n_samples, n_features)
            PCA_DIMENSIONS = [100, 500, NUM_FRAMES_ANO]

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
                for ht in HEATMAP_TYPES:
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
                fig, axs = plt.subplots(len(abstraction_methods)*len(PCA_DIMENSIONS)*len(HEATMAP_TYPES), 1, figsize=figsize)
                plt.subplots_adjust(hspace=hspace)
                plt.suptitle("P2P Heatmap Distances", fontsize=15, y=0.95)
                for ht in HEATMAP_TYPES:
                    for am in abstraction_methods:
                        for dim in PCA_DIMENSIONS:
                            for dm in DISTANCE_METHODS:
                                cprintf(f'\n########### using distance method >>{dm}<< ########### pca dimension {dim} ##############', 'yellow')
                                evaluate_p2p_failure_prediction(cfg,
                                                                heatmap_type=ht,
                                                                heatmap_types = HEATMAP_TYPES,
                                                                anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                                nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                                distance_method=dm,
                                                                distance_methods = DISTANCE_METHODS,
                                                                pca_dimension=dim,
                                                                pca_dimensions = PCA_DIMENSIONS,
                                                                abstraction_method = am,
                                                                abstraction_methods = abstraction_methods,
                                                                fig=fig,
                                                                axs=axs)
                plt.show()
            elif cfg.METHOD == 'test':
                for heatmap_type in HEATMAP_TYPES:
                    for distance_method in DISTANCE_METHODS:
                        for distance_type in DISTANCE_TYPES:
                            for pca_dimension in PCA_DIMENSIONS:
                                cprintb(f'\n########### Using Distance Method: \"{distance_method}\" ###########', 'l_blue')
                                cprintb(f'########### Using Distance Type: \"{distance_type}\" ###########', 'l_blue')
                                cprintb(f'########### Using PCA Dimension: {pca_dimension} ###########', 'l_blue')
                                test(cfg,
                                     NOMINAL_PATHS,
                                     ANOMALOUS_PATHS,
                                     NUM_FRAMES_NOM,
                                     NUM_FRAMES_ANO,
                                     heatmap_type=heatmap_type,
                                     anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                     nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                     distance_method=distance_method,
                                     distance_type=distance_type,
                                     pca_dimension=pca_dimension,
                                     run_id=run_id)
    
    end_time = time.monotonic()
    cprintf(f"Completed {total_runs} evaluation run(s) of {len(ANO_SIMULATIONS)} simulation(s) in {timedelta(seconds=end_time-start_time)}", 'yellow')
