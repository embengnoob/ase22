import sys
sys.path.append("..")
import utils
from utils import *
from utils_models import *
from heatmap import compute_heatmap
try:
    from config import load_config
except:
    from config import Config

from evaluate_failure_prediction_p2p import evaluate_failure_prediction_p2p
from evaluate_failure_prediction_thirdeye import evaluate_failure_prediction_thirdeye

def simExists(cfg, method, run_id, sim_name, attention_type, sim_type, seconds_to_anticipate, threshold_extras=[]):
    if type(run_id) != str:
        run_id = str(run_id)
    get_paths_result = get_paths(cfg, run_id, sim_name, attention_type, sim_type, method)
    PATHS = get_paths_result['paths']
    SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, RUN_RESULTS_PATH, RUN_FIGS_PATH, NPY_SCORES_FOLDER_PATH = PATHS
    nominal, threshold = get_paths_result['sim_type']
    NUM_OF_FRAMES = get_paths_result['num_of_frames']

    MODE = heatmap_calculation_modus(cfg, run_id, sim_name, attention_type, sim_type, method)

    if MODE is not None:
        compute_heatmap(cfg, nominal, sim_name, NUM_OF_FRAMES, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, NPY_SCORES_FOLDER_PATH, MODE)
    # # check if img paths in the csv file need correcting (possible directory change of simulation data)
    # correct_img_paths_in_csv_files(HEATMAP_CSV_PATH)
    # PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH]
    if threshold:
        calc_distances = False
        THRESHOLD_VECTORS_FOLDER_PATH = HEATMAP_FOLDER_PATH
        NOMINAL_PATHS, NUM_FRAMES_NOM, SIMULATION_NAME_ANOMALOUS, SIMULATION_NAME_NOMINAL, DISTANCE_TYPES, ANALYSE_DISTANCE = threshold_extras
        for distance_type in DISTANCE_TYPES:
            if f'dist_vect_{distance_type}.csv' not in os.listdir(THRESHOLD_VECTORS_FOLDER_PATH):
                cprintf(f"WARNING: Threshold sim distance data of distance type {distance_type} is unavailable. Calculating distances ...", "red")
                calc_distances = True
        if calc_distances:
            evaluate_failure_prediction_p2p(cfg,
                                            NOMINAL_PATHS,
                                            PATHS,
                                            NUM_FRAMES_NOM,
                                            NUM_OF_FRAMES,
                                            heatmap_type=attention_type,
                                            anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                            nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                            distance_types=DISTANCE_TYPES,
                                            analyse_distance=ANALYSE_DISTANCE,
                                            run_id=run_id,
                                            threshold_sim = True,
                                            seconds_to_anticipate=seconds_to_anticipate)

        return THRESHOLD_VECTORS_FOLDER_PATH
    else:
        return NUM_OF_FRAMES, PATHS



if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        start_time = time.monotonic()
        os.chdir(os.getcwd().replace('scripts', ''))

        try:
            cfg = Config("config_my.py")
        except:
            cfg = load_config("config_my.py")
        
        # cfg.from_pyfile("config_my.py")
        if cfg.IGNORE_WARNINGS:
            warnings.filterwarnings("ignore")


    ##################### Simulation Selection #####################       
                 
        ANO_SIMULATIONS = [
                            'track1-night-moon'
                            # 'track1-day-fog-100',
                            # 'track1-day-rain-100',
                            # # 'track1-day-snow-100',
                            # 'track1-day-sunny',
                            # 'track1-night-rain-100',
                            # 'track1-night-fog-100',
                            # 'track1-night-snow-100'
                        ]

        
        NOM_SIMULATIONS = [
                            cfg.BASE_NOMINAL_SUNNY_SIM
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM
                        ]


        THRESHOLD_SIMULATIONS = [
                                    cfg.BASE_THRESHOLD_SUNNY_SIM
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM
                                ]
        
        RUN_ID_NUMBERS = [
                            [2]  # [1, 2]
                            # [1],
                            # [1],
                            # # [1],
                            # [1],
                            # [1, 2],
                            # [1, 2],
                            # [1, 2]
                            ]
        
        SUMMARY_COLLAGES = [
                            [False]
                            # [False],
                            # [False],
                            # # [False],
                            # [False],
                            # [False, False],
                            # [False, False],
                            # [False, False]
                            ]

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

    ##################### Heatmap and Distance Types / Seconds to Anticipate #####################

        # P2P Settings
        HEATMAP_TYPES = ['smoothgrad'] #, 'gradcam++', 'rectgrad', 'rectgrad_prr', 'saliency', 'guided_bp', 'gradient-input', 'integgrad', 'epsilon_lrp']
        DISTANCE_TYPES = ['sobolev-norm'] #['euclidean', 'manhattan', 'cosine', 'EMD', 'pearson', 'spearman', 'kendall', 'moran', 'kl-divergence', 'mutual-info', 'sobolev-norm']
        ANALYSE_DISTANCE = {
            'euclidean' : (True, 0.99),
            'manhattan' : (False, 0.99),
            'cosine' : (False, 0.99),
            'EMD' : (False, 0.99),
            'pearson' : (False, 0.99),
            'spearman' : (False, 0.99),
            'kendall' : (False, 0.99),
            'moran' : (False, 0.50),
            'kl-divergence' : (False, 0.99),
            'mutual-info' : (False, 0.50),
            'sobolev-norm' : (True, 0.99)}
        SECONDS_TO_ANTICIPATE = [1, 2, 3]
        # ThirdEye Settings
        summary_types = ['-avg', '-avg-grad']
        aggregation_methods = ['mean', 'max']
        # abstraction_methods = ['avg', 'variance']

    ##################### Result CSV Path Lists #####################

        dt_total_scores_paths = [[] for i in range(len(ANO_SIMULATIONS))]
        hm_total_scores_paths = [[] for i in range(len(ANO_SIMULATIONS))]

    ##################### Starting Evaluation #####################

        average_thresholds_path = ''
        prev_sim = ''
        for method in cfg.METHODS:
            for sim_idx, sim_name in enumerate(ANO_SIMULATIONS):

                # if (sim_idx+1) % 3 == 0:
                #     averaged_theshold = True
                # else:
                #     averaged_theshold = False

                run_results = []
                run_keys = []

                for run_number in range(len(RUN_ID_NUMBERS[sim_idx])):
                    # Check if a simulation with this run number already exists
                    run_id = RUN_ID_NUMBERS[sim_idx][run_number]
                    SIMULATION_NAME_ANOMALOUS = sim_name
                    SIMULATION_NAME_NOMINAL = NOM_SIMULATIONS[sim_idx]
                    SIMULATION_NAME_THRESHOLD = THRESHOLD_SIMULATIONS[sim_idx]
                    cfg.SIMULATION_NAME = SIMULATION_NAME_ANOMALOUS
                    cfg.SIMULATION_NAME_NOMINAL = SIMULATION_NAME_NOMINAL
                    cfg.GENERATE_SUMMARY_COLLAGES = SUMMARY_COLLAGES[sim_idx][run_number]

                    run_figs_p2p = []
                    run_figs_thirdeye = []
                    if sim_name != prev_sim:
                        # clear previous result CSVs:
                        path_res = get_paths(cfg, str(run_id), sim_name, attention_type='none', sim_type='anomalous', method=method)
                        results_folder_path = path_res['paths'][7]
                        figs_dir =  path_res['paths'][8]
                        print(figs_dir)
                        if os.path.exists(figs_dir) and os.path.exists(results_folder_path):
                            delete_contents_except(results_folder_path, figs_dir)

                    if not os.path.exists(figs_dir):
                        print(f'Figs dir doesn\'t exist: {figs_dir}')
                        cfg.PLOT_POINT_TO_POINT = True
                    elif cfg.FORCE_REWRITE_SCORES_PLOTS:
                        cfg.PLOT_POINT_TO_POINT = True
                    else:
                        if len(os.listdir(figs_dir)) < (len(HEATMAP_TYPES)):
                            print('Less figs than number of heatmaps')
                            cfg.PLOT_POINT_TO_POINT = True
                        else:
                            cfg.PLOT_POINT_TO_POINT = False



                    for heatmap_type in HEATMAP_TYPES:
                        cprintb(f'\n########### Prediction Method: {method} ###########', 'l_yellow')

                        cprintb(f'\n########### Simulation: {sim_name} ({sim_idx + 1} of {len(ANO_SIMULATIONS)}) ###########', 'l_red')
                        cprintb(f'########### Nominal Sim: {SIMULATION_NAME_NOMINAL}  ###########', 'l_red')
                        if method == 'p2p':
                            cprintb(f'########### Threshold Sim: {SIMULATION_NAME_THRESHOLD}  ###########', 'l_red')
                        cprintb(f'\n############## run number {run_id} ({RUN_ID_NUMBERS[sim_idx].index(run_id) + 1} of {len(RUN_ID_NUMBERS[sim_idx])}) ##############', 'l_blue')
                        cprintb(f'########### Using Heatmap Type: {heatmap_type} ({HEATMAP_TYPES.index(heatmap_type) + 1} of {len(HEATMAP_TYPES)}) ###########', 'l_blue')

###############################################################################################################################################################
################# check whether nominal and anomalous simulation and the corresponding heatmaps are already generated, generate them otherwise ################
########################################################################## get paths ##########################################################################
###############################################################################################################################################################

                        NUM_FRAMES_NOM, NOMINAL_PATHS = simExists(cfg, method, str(run_id), sim_name=SIMULATION_NAME_NOMINAL, attention_type=heatmap_type, sim_type='nominal', 
                                                                seconds_to_anticipate=SECONDS_TO_ANTICIPATE)
                        
                        NUM_FRAMES_ANO, ANOMALOUS_PATHS = simExists(cfg, method, str(run_id), sim_name=SIMULATION_NAME_ANOMALOUS, attention_type=heatmap_type, sim_type='anomalous', 
                                                                    seconds_to_anticipate=SECONDS_TO_ANTICIPATE)
                        if method == 'p2p':
                            THRESHOLD_VECTORS_FOLDER_PATH = simExists(cfg, method, '1', sim_name=SIMULATION_NAME_THRESHOLD, attention_type=heatmap_type, sim_type='threshold',      
                                                                    seconds_to_anticipate=SECONDS_TO_ANTICIPATE,
                                                                    threshold_extras=[NOMINAL_PATHS,
                                                                                        NUM_FRAMES_NOM,
                                                                                        SIMULATION_NAME_ANOMALOUS,
                                                                                        SIMULATION_NAME_NOMINAL,
                                                                                        DISTANCE_TYPES,
                                                                                        ANALYSE_DISTANCE])
                            ANOMALOUS_PATHS.append(THRESHOLD_VECTORS_FOLDER_PATH)
###############################################################################################################################################################
########################################### Run the methods, calculate results and get figs and total results paths ###########################################
###############################################################################################################################################################                        
                        if method == 'thirdeye':
                            fig_img_address, results_csv_path, results_folder_path = evaluate_failure_prediction_thirdeye(cfg,
                                                                                                                        NOMINAL_PATHS,
                                                                                                                        ANOMALOUS_PATHS,
                                                                                                                        seconds_to_anticipate=SECONDS_TO_ANTICIPATE,
                                                                                                                        anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                                                                                        nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                                                                                        heatmap_type=heatmap_type,
                                                                                                                        summary_types=summary_types,
                                                                                                                        aggregation_methods=aggregation_methods)
                            run_figs_thirdeye.append(fig_img_address)

                        elif method == 'p2p':
                            fig_img_address, results_csv_path, results_folder_path = evaluate_failure_prediction_p2p(cfg,
                                                                                                                    NOMINAL_PATHS,
                                                                                                                    ANOMALOUS_PATHS,
                                                                                                                    NUM_FRAMES_NOM,
                                                                                                                    NUM_FRAMES_ANO,
                                                                                                                    heatmap_type=heatmap_type,
                                                                                                                    anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                                                                                    nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                                                                                    distance_types=DISTANCE_TYPES,
                                                                                                                    analyse_distance=ANALYSE_DISTANCE,
                                                                                                                    run_id=run_id,
                                                                                                                    threshold_sim = False,
                                                                                                                    seconds_to_anticipate = SECONDS_TO_ANTICIPATE)
                            run_figs_p2p.append(fig_img_address)

                    # copy all figs of a run to a single folder
                    if method == 'p2p' and cfg.PLOT_POINT_TO_POINT:
                        copy_run_figs(cfg, ANOMALOUS_PATHS[0], run_id, run_figs_p2p, ANOMALOUS_PATHS[8])
                    if method == 'thirdeye' and cfg.PLOT_THIRDEYE:
                        copy_run_figs(cfg, ANOMALOUS_PATHS[0], run_id, run_figs_thirdeye, ANOMALOUS_PATHS[8])


###############################################################################################################################################################
#################################################################### Simulation Evaluation ####################################################################
###############################################################################################################################################################  
                    if cfg.CALCULATE_RESULTS:
                        # calcuate scores + get number of invalid thresholds
                        cprintf(f'{results_csv_path}', 'l_red')
                        results_df = pd.read_csv(results_csv_path)
                        
                        ###### comparison between heatmap types only ######
                        if method == 'p2p':
                            total_scores_hm = os.path.join(results_folder_path, f'p2p_heatmaps_results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores.csv')
                        else:
                            total_scores_hm = os.path.join(results_folder_path, f'thirdeye_heatmaps_results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores.csv')
                        if not os.path.exists(total_scores_hm):
                            with open(total_scores_hm, mode='w', newline='') as total_scores_hm_file:
                                writer = csv.writer(total_scores_hm_file,
                                                    delimiter=',',
                                                    quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL,
                                                    lineterminator='\n')
                                writer.writerow(["time_stamp", "heatmap_type", "sta", "TP", "FP", "TN", "FN", "precision", "recall", "accuracy", "fpr",
                                "TP_all", "FP_all", "TN_all", "FN_all", "precision_all", "recall_all", "accuracy_all", "fpr_all"])
                                        
                                for heatmap_type in HEATMAP_TYPES:
                                    for sta in SECONDS_TO_ANTICIPATE:
                                        heatmap_filter_valid_threshold = results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['sta'] == sta)]
                                        # print(heatmap_filter_valid_threshold['TP'].values)
                                        # print(heatmap_filter_valid_threshold['FP'].values)
                                        # print(heatmap_filter_valid_threshold['TN'].values)
                                        # print(heatmap_filter_valid_threshold['FN'].values)
                                        # print('----------------------------------------------')

                                        TP = np.sum(heatmap_filter_valid_threshold['TP'].values)
                                        FP = np.sum(heatmap_filter_valid_threshold['FP'].values)
                                        TN = np.sum(heatmap_filter_valid_threshold['TN'].values)
                                        FN = np.sum(heatmap_filter_valid_threshold['FN'].values)
                                        # print(TP, FP, TN, FN)
                                        # print('----------------------------------------------')
                                        # print('----------------------------------------------')
                                        precision = TP/(TP+FP)
                                        recall = TP/(TP+FN)
                                        accuracy = (TP+TN)/(TP+TN+FP+FN)
                                        fpr = FP/(FP+TN)

                                        # Filter the DataFrame for heatmap type and 'is_threshold_too_low' being True
                                        heatmap_filter_valid_threshold_all = results_df[(results_df['heatmap_type'] == f'{heatmap_type}')]
                                        TP_all = np.sum(heatmap_filter_valid_threshold_all['TP'].values)
                                        FP_all = np.sum(heatmap_filter_valid_threshold_all['FP'].values)
                                        TN_all = np.sum(heatmap_filter_valid_threshold_all['TN'].values)
                                        FN_all = np.sum(heatmap_filter_valid_threshold_all['FN'].values)
                                        precision_all = TP_all/(TP_all+FP_all)
                                        recall_all = TP_all/(TP_all+FN_all)
                                        accuracy_all = (TP_all+TN_all)/(TP_all+TN_all+FP_all+FN_all)
                                        fpr_all = FP_all/(FP_all+TN_all)

                                        writer.writerow([datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), heatmap_type,
                                                        #   str(too_low_count), str(too_high_count),
                                                        str(sta), str(TP), str(FP), str(TN), str(FN), str(precision),
                                                        str(recall), str(accuracy), str(fpr), str(TP_all), str(FP_all),
                                                        str(TN_all), str(FN_all), str(precision_all), str(recall_all),
                                                        str(accuracy_all), str(fpr_all)])
                        if method == 'p2p':
                            total_scores_dt = os.path.join(results_folder_path, f'p2p_distance_types_results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores.csv')
                            if not os.path.exists(total_scores_dt):
                                with open(total_scores_dt, mode='w',
                                            newline='') as total_scores_dt_file:
                                    writer = csv.writer(total_scores_dt_file,
                                                        delimiter=',',
                                                        quotechar='"',
                                                        quoting=csv.QUOTE_MINIMAL,
                                                        lineterminator='\n')
                                    writer.writerow(["time_stamp","distance_type", "sta", "TP", "FP", "TN", "FN", "precision",
                                    "recall", "accuracy", "fpr","TP_all", "FP_all", "TN_all", "FN_all", "precision_all", "recall_all", "accuracy_all", "fpr_all"])
                                        
                                    for distance_type in DISTANCE_TYPES:
                                        for sta in SECONDS_TO_ANTICIPATE:
                                            # Filter the DataFrame for distance type and 'is_threshold_too_low' being True
                                            distance_filter_valid_threshold = results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['sta'] == sta)]

                                            TP = np.sum(distance_filter_valid_threshold['TP'].values)
                                            FP = np.sum(distance_filter_valid_threshold['FP'].values)
                                            TN = np.sum(distance_filter_valid_threshold['TN'].values)
                                            FN = np.sum(distance_filter_valid_threshold['FN'].values)

                                            precision = TP/(TP+FP)
                                            recall = TP/(TP+FN)
                                            accuracy = (TP+TN)/(TP+TN+FP+FN)
                                            fpr = FP/(FP+TN)

                                            # Filter the DataFrame for distance type and 'is_threshold_too_low' being True
                                            distance_filter_valid_threshold_all = results_df[(results_df['distance_type'] == f'{distance_type}')]
                                            TP_all = np.sum(distance_filter_valid_threshold_all['TP'].values)
                                            FP_all = np.sum(distance_filter_valid_threshold_all['FP'].values)
                                            TN_all = np.sum(distance_filter_valid_threshold_all['TN'].values)
                                            FN_all = np.sum(distance_filter_valid_threshold_all['FN'].values)
                                            precision_all = TP_all/(TP_all+FP_all)
                                            recall_all = TP_all/(TP_all+FN_all)
                                            accuracy_all = (TP_all+TN_all)/(TP_all+TN_all+FP_all+FN_all)
                                            fpr_all = FP_all/(FP_all+TN_all)


                                            writer.writerow([datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), distance_type,
                                                            #   str(too_low_count), str(too_high_count),
                                                                str(sta), str(TP), str(FP), str(TN), str(FN), str(precision), str(recall), str(accuracy), str(fpr), str(TP_all), str(FP_all), str(TN_all), str(FN_all), str(precision_all), str(recall_all), str(accuracy_all), str(fpr_all)])

                    print(total_scores_hm)
                    print(total_scores_dt)
                    print(results_csv_path)
                    hm_total_scores_paths[sim_idx].append(total_scores_hm)
                    dt_total_scores_paths[sim_idx].append(total_scores_dt)
                prev_sim = sim_name



        ##################### Evaluation Results Post-processing #####################

            # create results folder
            RESULTS_DIR = os.path.join(cfg.FINAL_RESULTS_DIR, datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), method)
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            
            # save list of simulation names as a txt file
            with open(os.path.join(RESULTS_DIR, 'sims.txt'), 'w') as f:
                f.write(f"Anomalous Sims:\n")
                for sim in ANO_SIMULATIONS:
                    f.write(f"{sim}\n")
            
            seconds_to_anticipate_str = []
            for sta in SECONDS_TO_ANTICIPATE:
                seconds_to_anticipate_str.append(str(sta)+'s')
            seconds_to_anticipate_str.append('all')

            ##########################################################
            # What sta to plot and compare? [1s, 2s, 3s, all]
            STA_PLOT = [False, False, False, True]
            ##########################################################

            # what criterion to take into account
            PLOTTING_CRITERIA = ['Precision','Recall','F3','Accuracy']
            PLOTTING_CRITERION = 'Accuracy'
            criterion_idx = PLOTTING_CRITERIA.index(PLOTTING_CRITERION) + 1
            criterion_vals_ht = np.zeros((len(HEATMAP_TYPES)*len(ANO_SIMULATIONS), 4), dtype=float)
            criterion_vals_dt = np.zeros((len(DISTANCE_TYPES)*len(ANO_SIMULATIONS), 4), dtype=float)
            # save scores based on heatmap types
            ht_last_index = 0
            dt_last_index = 0
            for sim_idx, sim_name in enumerate(ANO_SIMULATIONS):
                if sim_idx == 0:
                    ht_scores_df = create_result_df(hm_total_scores_paths, HEATMAP_TYPES, ANO_SIMULATIONS, sim_idx, RUN_ID_NUMBERS, seconds_to_anticipate_str)
                    dt_scores_df = create_result_df(dt_total_scores_paths, DISTANCE_TYPES, ANO_SIMULATIONS, sim_idx, RUN_ID_NUMBERS, seconds_to_anticipate_str)
                ht_scores_df, ht_last_index = heatmap_or_distance_type_scores(hm_total_scores_paths, HEATMAP_TYPES, sim_idx, sim_name, RUN_ID_NUMBERS, SECONDS_TO_ANTICIPATE, ht_scores_df, ht_last_index, type = 'heatmap_type', print_results=False)
                dt_scores_df, dt_last_index = heatmap_or_distance_type_scores(dt_total_scores_paths, DISTANCE_TYPES, sim_idx, sim_name, RUN_ID_NUMBERS, SECONDS_TO_ANTICIPATE, dt_scores_df, dt_last_index, type = 'distance_type', print_results=False)
                for heatmap_type_idx in range(len(HEATMAP_TYPES)):
                    for sta_idx, sta in enumerate(seconds_to_anticipate_str):
                        row_idx_in_df = (sim_idx*(len(HEATMAP_TYPES)+1))+(heatmap_type_idx+1)
                        col_idx_in_df = (sta_idx+1)*criterion_idx
                        row_idx_in_crit_vals = sim_idx*len(HEATMAP_TYPES) + heatmap_type_idx
                        criterion_vals_ht[row_idx_in_crit_vals][sta_idx] = ht_scores_df.iloc[row_idx_in_df, col_idx_in_df]
                for distance_type_idx in range(len(DISTANCE_TYPES)):
                    for sta_idx, sta in enumerate(seconds_to_anticipate_str):
                        row_idx_in_df = (sim_idx*(len(DISTANCE_TYPES)+1))+(distance_type_idx+1)
                        col_idx_in_df = (sta_idx+1)*criterion_idx
                        row_idx_in_crit_vals = sim_idx*len(DISTANCE_TYPES) + distance_type_idx
                        criterion_vals_dt[row_idx_in_crit_vals][sta_idx] = dt_scores_df.iloc[row_idx_in_df, col_idx_in_df]
            cprintf('***************************************************************', 'l_yellow')
            cprintf(f'*********************** METHOD: {method} ***************************', 'l_yellow')
            cprintf('***************************************************************', 'l_yellow')
            print('---------------------------------------------------------------')
            print('HEATMAP TYPES COMPARISON TABLE')
            print('---------------------------------------------------------------')
            print(ht_scores_df)
            print('---------------------------------------------------------------')
            print('***************************************************************')
            print('---------------------------------------------------------------')
            print('DISTANCE TYPES COMPARISON TABLE')
            print('---------------------------------------------------------------')
            print(dt_scores_df)
            print('---------------------------------------------------------------')
            ht_scores_df.to_csv(os.path.join(RESULTS_DIR, 'heatmaps.csv'), index=False)
            ht_scores_df.to_excel(os.path.join(RESULTS_DIR, 'heatmaps.xlsx'))
            dt_scores_df.to_csv(os.path.join(RESULTS_DIR, 'distances.csv'), index=False)
            dt_scores_df.to_excel(os.path.join(RESULTS_DIR, 'distances.xlsx')) 



            # plot comparison graph
            ht_or_dt_comparison_plot(ANO_SIMULATIONS, seconds_to_anticipate_str, STA_PLOT, DISTANCE_TYPES,
            criterion_vals_dt, PLOTTING_CRITERION, FIG_SAVE_ADDRESS=os.path.join(RESULTS_DIR, 'distances.png'), type='Distance')

            ht_or_dt_comparison_plot(ANO_SIMULATIONS, seconds_to_anticipate_str, STA_PLOT, HEATMAP_TYPES,
            criterion_vals_ht, PLOTTING_CRITERION, FIG_SAVE_ADDRESS=os.path.join(RESULTS_DIR, 'heatmaps.png'), type='Heatmap')


        end_time = time.monotonic()
        cprintf(f"Completed {total_runs} evaluation run(s) of {len(ANO_SIMULATIONS)} simulation(s) in {timedelta(seconds=end_time-start_time)}", 'green')
