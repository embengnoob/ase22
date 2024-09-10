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

from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction_thirdeye, evaluate_failure_prediction_p2p

def simExists(cfg, run_id, sim_name, attention_type, sim_type, seconds_to_anticipate, threshold_extras=[]):
    if type(run_id) != str:
        run_id = str(run_id)
    get_paths_result = get_paths(cfg, run_id, sim_name, attention_type, sim_type)
    PATHS = get_paths_result['paths']
    SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, RUN_RESULTS_PATH, RUN_FIGS_PATH, NPY_SCORES_FOLDER_PATH = PATHS
    nominal, threshold = get_paths_result['sim_type']
    NUM_OF_FRAMES = get_paths_result['num_of_frames']

    MODE = heatmap_calculation_modus(cfg, run_id, sim_name, attention_type, sim_type)

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
        DISTANCE_TYPES = ['sobolev-norm', 'euclidean'] #['euclidean', 'manhattan', 'cosine', 'EMD', 'pearson', 'spearman', 'kendall', 'moran', 'kl-divergence', 'mutual-info', 'sobolev-norm']
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
        abstraction_methods = ['avg', 'variance']

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

                    run_figs = []
                    if sim_name != prev_sim:
                        # clear previous result CSVs:
                        path_res = get_paths(cfg, str(run_id), sim_name, attention_type='none', sim_type='anomalous')
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
                    # check whether nominal and anomalous simulation and the corresponding heatmaps are already generated, generate them otherwise
                    for heatmap_type in HEATMAP_TYPES:
                        cprintb(f'\n########### Prediction Method: {method} ###########', 'l_yellow')

                        cprintb(f'\n########### Simulation: {sim_name} ({sim_idx + 1} of {len(ANO_SIMULATIONS)}) ###########', 'l_red')
                        cprintb(f'########### Nominal Sim: {SIMULATION_NAME_NOMINAL}  ###########', 'l_red')
                        if method == 'p2p':
                            cprintb(f'########### Threshold Sim: {SIMULATION_NAME_THRESHOLD}  ###########', 'l_red')
                        cprintb(f'\n############## run number {run_id} ({RUN_ID_NUMBERS[sim_idx].index(run_id) + 1} of {len(RUN_ID_NUMBERS[sim_idx])}) ##############', 'l_blue')
                        cprintb(f'########### Using Heatmap Type: {heatmap_type} ({HEATMAP_TYPES.index(heatmap_type) + 1} of {len(HEATMAP_TYPES)}) ###########', 'l_blue')

                        NUM_FRAMES_NOM, NOMINAL_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_NOMINAL, attention_type=heatmap_type, sim_type='nominal', 
                                                                seconds_to_anticipate=SECONDS_TO_ANTICIPATE)
                        
                        NUM_FRAMES_ANO, ANOMALOUS_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_ANOMALOUS, attention_type=heatmap_type, sim_type='anomalous', 
                                                                    seconds_to_anticipate=SECONDS_TO_ANTICIPATE)
                        if method == 'p2p':
                            THRESHOLD_VECTORS_FOLDER_PATH = simExists(cfg, '1', sim_name=SIMULATION_NAME_THRESHOLD, attention_type=heatmap_type, sim_type='threshold',      
                                                                    seconds_to_anticipate=SECONDS_TO_ANTICIPATE,
                                                                    threshold_extras=[NOMINAL_PATHS,
                                                                                        NUM_FRAMES_NOM,
                                                                                        SIMULATION_NAME_ANOMALOUS,
                                                                                        SIMULATION_NAME_NOMINAL,
                                                                                        DISTANCE_TYPES,
                                                                                        ANALYSE_DISTANCE])
                            ANOMALOUS_PATHS.append(THRESHOLD_VECTORS_FOLDER_PATH)
                        
                        if method == 'thirdeye':
                            # # get number of OOTs
                            # path = os.path.join(cfg.TESTING_DATA_DIR,
                            #                 SIMULATION_NAME_ANOMALOUS,
                            #                 'heatmaps-' + 'smoothgrad',
                            #                 'driving_log.csv')
                            # data_df_anomalous = pd.read_csv(path)
                            # number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
                            # OOT_anomalous = data_df_anomalous['crashed']
                            # OOT_anomalous.is_copy = None
                            # OOT_anomalous_in_anomalous_conditions = OOT_anomalous.copy()
                            # all_first_frame_position_OOT_sequences = get_OOT_frames(data_df_anomalous, number_frames_anomalous)
                            # number_of_OOTs = len(all_first_frame_position_OOT_sequences)
                            # print("identified %d OOT(s)" % number_of_OOTs)

                            # if len(aggregation_methods) == 3:
                            #     figsize = (15, 12)
                            #     hspace = 0.69
                            # elif len(aggregation_methods) == 2:
                            #     figsize = (15, 10)
                            #     hspace = 0.44
                            # else:
                            #     raise ValueError("No predefined settings for this number of aggregation methods.")
                            
                            # fig, axs = plt.subplots(len(aggregation_methods)*2, 1, figsize=figsize)
                            # plt.subplots_adjust(hspace=hspace)
                            # plt.suptitle("Heatmap scores and thresholds", fontsize=15, y=0.95)

                            for st in summary_types:
                                for am in aggregation_methods:
                                    cprintf(f'\n########### using agg_method:{am}, summary type:{st} ###########', 'yellow')
                                    evaluate_failure_prediction_thirdeye(cfg,
                                                                        NOMINAL_PATHS,
                                                                        ANOMALOUS_PATHS,
                                                                        seconds_to_anticipate=SECONDS_TO_ANTICIPATE,
                                                                        anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                                        nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                                        heatmap_type=heatmap_type,
                                                                        summary_type=st,
                                                                        aggregation_method=am,
                                                                        condition='ood')
                            # plt.show()
                            # print(subplot_counter)

                        elif method == 'p2p':
                                
                            # if averaged_theshold:
                            #     average_thresholds = {}
                            #     # get all indexes of current ano sim
                            #     indices = [i for i, x in enumerate(ANO_SIMULATIONS) if x == SIMULATION_NAME_ANOMALOUS]
                            #     # find the nom sim name other than base nominal sunny sim
                            #     for i in indices:
                            #         if NOM_SIMULATIONS[i] != cfg.BASE_NOMINAL_SUNNY_SIM:
                            #             th_assess_nom_sim = NOM_SIMULATIONS[i]
                            #     average_thresholds_path = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'average_thresholds_{SIMULATION_NAME_ANOMALOUS}_nom_{th_assess_nom_sim}.csv')
                            #     average_thresholds_folder = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold')
                            #     if not os.path.exists(average_thresholds_folder):
                            #         os.makedirs(average_thresholds_folder)
                            #     with open(average_thresholds_path, 'r') as csvfile:
                            #         reader = csv.DictReader(csvfile)
                            #         for row in reader:
                            #             hm_type = row['heatmap_type']
                            #             dt_type = row['distance_type']
                            #             average_threshold = float(row['average_threshold'])  # Assuming the threshold is a float
                                        
                            #             if hm_type not in average_thresholds:
                            #                 average_thresholds[hm_type] = {}
                            #             average_thresholds[hm_type][dt_type] = average_threshold

                            #     average_thresholds = average_thresholds[heatmap_type]
                            # else:
                            #     average_thresholds = {}

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
                                                                                                                    #averaged_thresholds=average_thresholds)
                            run_figs.append(fig_img_address)

                    # copy all figs of a run to a single folder
                    if cfg.PLOT_POINT_TO_POINT:
                        copy_run_figs(cfg, ANOMALOUS_PATHS[0], run_id, run_figs)



                    ##################### Simulation Evaluation #####################
                    if cfg.CALCULATE_RESULTS:

                        # calcuate scores + get number of invalid thresholds
                        results_df = pd.read_csv(results_csv_path)
                        # if averaged_theshold:
                        #     total_scores_hm = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores_heatmaps.csv')
                        #     total_scores_dt = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores_distance_types.csv')
                        # else:
                        total_scores_hm = os.path.join(results_folder_path, f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores_heatmaps.csv')
                        total_scores_dt = os.path.join(results_folder_path, f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{SIMULATION_NAME_NOMINAL}_total_scores_distance_types.csv')
                        if not os.path.exists(total_scores_hm):
                            with open(total_scores_hm, mode='w',
                                        newline='') as total_scores_hm_file:
                                writer = csv.writer(total_scores_hm_file,
                                                    delimiter=',',
                                                    quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL,
                                                    lineterminator='\n')
                                # writer.writerow(["time_stamp","heatmap_type", "is_threshold_too_low_count", "is_threshold_too_high_count", "sta", "TP", "FP", "TN", "FN", "precision", "recall", "accuracy", "fpr","TP_all", "FP_all", "TN_all", "FN_all", "precision_all", "recall_all", "accuracy_all", "fpr_all"])
                                writer.writerow(["time_stamp","heatmap_type", "sta", "TP", "FP", "TN", "FN", "precision", "recall", "accuracy", "fpr", "TP_all", "FP_all", "TN_all", "FN_all", "precision_all", "recall_all", "accuracy_all", "fpr_all"])
                                        
                                for heatmap_type in HEATMAP_TYPES:
                                    for sta in SECONDS_TO_ANTICIPATE:
                                        num_crashes = results_df['crashes'].values[0]
                                        # # Filter the DataFrame for heatmap type and 'is_threshold_too_low' being True
                                        # too_low_count = len(results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['is_threshold_too_low'] == True) & (results_df['sta'] == sta)])
                                        # too_high_count = len(results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['is_threshold_too_high'] == True) & (results_df['sta'] == sta)])
                                        # cases where thresholds are not too high or too low
                                        # heatmap_filter_valid_threshold = results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['is_threshold_too_low'] == False) & (results_df['is_threshold_too_high'] == False & (results_df['sta'] == sta))]
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

                                        # # Filter the DataFrame for heatmap type and 'is_threshold_too_low' being True
                                        # too_low_count_all = len(results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['is_threshold_too_low'] == True)])
                                        # too_high_count_all = len(results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['is_threshold_too_high'] == True)])
                                        # cases where thresholds are not too high or too low
                                        # heatmap_filter_valid_threshold_all = results_df[(results_df['heatmap_type'] == f'{heatmap_type}') & (results_df['is_threshold_too_low'] == False) & (results_df['is_threshold_too_high'] == False)]
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
                                                        str(sta), str(TP), str(FP), str(TN), str(FN), str(precision), str(recall), str(accuracy), str(fpr), str(TP_all), str(FP_all), str(TN_all), str(FN_all), str(precision_all), str(recall_all), str(accuracy_all), str(fpr_all)])
                        if not os.path.exists(total_scores_dt):
                            with open(total_scores_dt, mode='w',
                                        newline='') as total_scores_dt_file:
                                writer = csv.writer(total_scores_dt_file,
                                                    delimiter=',',
                                                    quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL,
                                                    lineterminator='\n')
                                writer.writerow(["time_stamp","distance_type", "sta", "TP", "FP", "TN", "FN", "precision", "recall", "accuracy", "fpr","TP_all", "FP_all", "TN_all", "FN_all", "precision_all", "recall_all", "accuracy_all", "fpr_all"])
                                    
                                for distance_type in DISTANCE_TYPES:
                                    for sta in SECONDS_TO_ANTICIPATE:
                                        num_crashes = results_df['crashes'].values[0]
                                        # # Filter the DataFrame for distance type and 'is_threshold_too_low' being True
                                        # too_low_count = len(results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['is_threshold_too_low'] == True) & (results_df['sta'] == sta)])
                                        # too_high_count = len(results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['is_threshold_too_high'] == True) & (results_df['sta'] == sta)])
                                        # cases where thresholds are not too high or too low
                                        # distance_filter_valid_threshold = results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['is_threshold_too_low'] == False) & (results_df['is_threshold_too_high'] == False & (results_df['sta'] == sta))]
                                        distance_filter_valid_threshold = results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['sta'] == sta)]

                                        TP = np.sum(distance_filter_valid_threshold['TP'].values)
                                        FP = np.sum(distance_filter_valid_threshold['FP'].values)
                                        TN = np.sum(distance_filter_valid_threshold['TN'].values)
                                        FN = np.sum(distance_filter_valid_threshold['FN'].values)

                                        precision = TP/(TP+FP)
                                        recall = TP/(TP+FN)
                                        accuracy = (TP+TN)/(TP+TN+FP+FN)
                                        fpr = FP/(FP+TN)

                                        # # Filter the DataFrame for distance type and 'is_threshold_too_low' being True
                                        # too_low_count_all = len(results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['is_threshold_too_low'] == True)])
                                        # too_high_count_all = len(results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['is_threshold_too_high'] == True)])
                                        # cases where thresholds are not too high or too low
                                        # distance_filter_valid_threshold_all = results_df[(results_df['distance_type'] == f'{distance_type}') & (results_df['is_threshold_too_low'] == False) & (results_df['is_threshold_too_high'] == False)]
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

                        # if cfg.THRESHOLD_ASSESSMENT:
                        #     # get all indexes of current ano sim
                        #     indices = [i for i, x in enumerate(ANO_SIMULATIONS) if x == SIMULATION_NAME_ANOMALOUS]
                        #     # find the nom sim name other than base nominal sunny sim
                        #     for i in indices:
                        #         if NOM_SIMULATIONS[i] != cfg.BASE_NOMINAL_SUNNY_SIM:
                        #             th_assess_nom_sim = NOM_SIMULATIONS[i]
                        #     if averaged_theshold:
                        #         average_thresholds_folder = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold')
                        #         if not os.path.exists(average_thresholds_folder):
                        #             os.makedirs(average_thresholds_folder)
                        #         threshold_assessment_csv_path_sunny = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{cfg.BASE_NOMINAL_SUNNY_SIM}.csv')
                        #         threshold_assessment_csv_path_similar = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{th_assess_nom_sim}.csv')
                        #     else:
                        #         threshold_assessment_csv_path_sunny = os.path.join(ANOMALOUS_PATHS[0], str(run_id), f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{cfg.BASE_NOMINAL_SUNNY_SIM}.csv')
                        #         threshold_assessment_csv_path_similar = os.path.join(ANOMALOUS_PATHS[0], str(run_id), f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{th_assess_nom_sim}.csv')
                        #     if (os.path.exists(threshold_assessment_csv_path_sunny)) and (os.path.exists(threshold_assessment_csv_path_similar)):
                        #         if averaged_theshold:
                        #             threshold_assessment_csv_path = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{th_assess_nom_sim}_threshold_assessment.csv')
                        #         else:
                        #             threshold_assessment_csv_path = os.path.join(ANOMALOUS_PATHS[0], str(run_id), f'results_ano_{SIMULATION_NAME_ANOMALOUS}_nom_{th_assess_nom_sim}_threshold_assessment.csv')
                        #         if not os.path.exists(threshold_assessment_csv_path):
                        #             with open(threshold_assessment_csv_path, mode='w',
                        #                         newline='') as invalid_thresholds_file:
                        #                 writer = csv.writer(invalid_thresholds_file,
                        #                                     delimiter=',',
                        #                                     quotechar='"',
                        #                                     quoting=csv.QUOTE_MINIMAL,
                        #                                     lineterminator='\n')
                        #                 writer.writerow(
                        #                     ["time_stamp","heatmap_type", "distance_type", "similar_nominal_threshold", "sunny_nominal_threshold", "threshold_diff", "threshold_diff_sunny_percentage", "threshold_diff_similar_percentage", "average_threshold"])
                        #                 sunny_df = pd.read_csv(threshold_assessment_csv_path_sunny)
                        #                 # similar_df = pd.read_csv(threshold_assessment_csv_path_similar)
                        #                 average_thresholds = {}
                        #                 for heatmap_type in HEATMAP_TYPES:
                        #                     average_thresholds[heatmap_type] = {}
                        #                     for distance_type in DISTANCE_TYPES:
                        #                         thresholds = []
                        #                         # Filter the DataFrame for heatmap type and distance type
                        #                         filtered_sunny = sunny_df[(sunny_df['heatmap_type'] == f'{heatmap_type}') & (sunny_df['distance_type'] == f'{distance_type}')]
                        #                         # filtered_similar = similar_df[(similar_df['heatmap_type'] == f'{heatmap_type}') & (similar_df['distance_type'] == f'{distance_type}')]
                        #                         # Get the value of threshold
                        #                         threshold_sunny = filtered_sunny['threshold'].values[0]
                        #                         # threshold_similar = filtered_similar['threshold'].values[0]
                        #                         # thresholds.append(threshold_similar)
                        #                         thresholds.append(threshold_sunny)
                        #                         max_val_sunny = filtered_sunny['max_val'].values[0]
                        #                         min_val_sunny = filtered_sunny['min_val'].values[0]
                        #                         # max_val_similar = filtered_similar['max_val'].values[0]
                        #                         # min_val_similar = filtered_similar['min_val'].values[0]

                        #                         # threshold_diff = threshold_similar - threshold_sunny
                        #                         # threshold_diff_sunny_percentage = threshold_diff / (max_val_sunny - min_val_sunny)
                        #                         # threshold_diff_similar_percentage = threshold_diff / (max_val_similar - min_val_similar)
                        #                         average_threshold = np.mean(thresholds)     
                        #                         # (threshold_similar + threshold_sunny)//2.0
                        #                         writer.writerow([datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), heatmap_type, distance_type, threshold_similar, threshold_sunny, str(threshold_diff), str(threshold_diff_sunny_percentage), str(threshold_diff_similar_percentage), str(average_threshold)])

                        #                         if ANALYSE_DISTANCE[distance_type][0]:
                        #                             average_thresholds[heatmap_type][distance_type] = average_threshold

                        #             average_thresholds_path = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold', f'average_thresholds_{SIMULATION_NAME_ANOMALOUS}_nom_{th_assess_nom_sim}.csv')
                        #             average_thresholds_folder = os.path.join(ANOMALOUS_PATHS[0], str(run_id), 'averaged_theshold')
                        #             if not os.path.exists(average_thresholds_folder):
                        #                 os.makedirs(average_thresholds_folder)
                        #             # Assuming average_thresholds is a 2D dictionary
                        #             with open(average_thresholds_path, 'w', newline='') as csvfile:
                        #                 fieldnames = ['heatmap_type', 'distance_type', 'average_threshold']
                        #                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                        
                        #                 writer.writeheader()
                        #                 for heatmap_type, distance_data in average_thresholds.items():
                        #                     for distance_type, average_threshold in distance_data.items():
                        #                         writer.writerow({'heatmap_type': heatmap_type, 'distance_type': distance_type, 'average_threshold': average_threshold})
                    print(total_scores_hm)
                    print(total_scores_dt)
                    print(results_csv_path)
                    hm_total_scores_paths[sim_idx].append(total_scores_hm)
                    dt_total_scores_paths[sim_idx].append(total_scores_dt)

                    cprintf(f'{hm_total_scores_paths}', 'l_red')
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
            ht_or_dt_comparison_plot(ANO_SIMULATIONS, seconds_to_anticipate_str, STA_PLOT, DISTANCE_TYPES, criterion_vals_dt, PLOTTING_CRITERION, FIG_SAVE_ADDRESS=os.path.join(RESULTS_DIR, 'distances.png'), type='Distance')
            ht_or_dt_comparison_plot(ANO_SIMULATIONS, seconds_to_anticipate_str, STA_PLOT, HEATMAP_TYPES, criterion_vals_ht, PLOTTING_CRITERION, FIG_SAVE_ADDRESS=os.path.join(RESULTS_DIR, 'heatmaps.png'), type='Heatmap')



        end_time = time.monotonic()
        cprintf(f"Completed {total_runs} evaluation run(s) of {len(ANO_SIMULATIONS)} simulation(s) in {timedelta(seconds=end_time-start_time)}", 'green')
