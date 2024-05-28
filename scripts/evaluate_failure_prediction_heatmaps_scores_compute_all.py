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

from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction, evaluate_p2p_failure_prediction, get_OOT_frames

def simExists(cfg, run_id, sim_name, attention_type, sim_type, threshold_extras=[]):

    if sim_type == 'nominal':
        nominal = True
        threshold = False
    elif sim_type == 'anomalous':
        nominal = False
        threshold = False
    elif sim_type == 'threshold':
        nominal = False
        threshold = True

    SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, cfg.TRACK, sim_type, sim_name)

    if not (nominal or threshold):
        MAIN_CSV_PATH = os.path.join(SIM_PATH, "src", run_id, "driving_log.csv")
    else:
        MAIN_CSV_PATH = os.path.join(SIM_PATH, "src", "driving_log.csv")

    HEATMAP_PARENT_FOLDER_PATH = os.path.join(SIM_PATH, "heatmaps", "heatmaps-" + attention_type.lower())

    if not nominal:
        if cfg.SPARSE_ATTRIBUTION:
            HEATMAP_FOLDER_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE')
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE', "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE', "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, f'{run_id}_SPARSE', "IMG_GRADIENT")
        else:
            HEATMAP_FOLDER_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id)
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, run_id, "IMG_GRADIENT")
    else:
        HEATMAP_FOLDER_PATH = HEATMAP_PARENT_FOLDER_PATH
        if cfg.SPARSE_ATTRIBUTION:
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, 'SPARSE', "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, 'SPARSE', "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, 'SPARSE', "IMG_GRADIENT")
        else:
            HEATMAP_CSV_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "driving_log.csv")
            HEATMAP_IMG_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "IMG")
            HEATMAP_IMG_GRADIENT_PATH = os.path.join(HEATMAP_PARENT_FOLDER_PATH, "IMG_GRADIENT")

    NUM_OF_FRAMES = get_num_frames(run_id, SIM_PATH, MAIN_CSV_PATH)
    # check if img paths in the csv file need correcting (possible directory change of simulation data)
    correct_img_paths_in_csv_files(MAIN_CSV_PATH)
    if not os.path.exists(SIM_PATH):
        raise ValueError(Fore.RED + f"The provided simulation path does not exist: {SIM_PATH}" + Fore.RESET)
    else:
        # 1- IMG & Gradient folders don't exist 
        if (not os.path.exists(HEATMAP_IMG_PATH)) and (not os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Neither heatmap IMG folder nor gradient folder exist. Creating image folder at {HEATMAP_IMG_PATH} and {HEATMAP_IMG_GRADIENT_PATH}", 'l_blue')
            os.makedirs(HEATMAP_IMG_PATH)
            os.makedirs(HEATMAP_IMG_GRADIENT_PATH)
            MODE = 'new_calc'

        # 2- Heatmap IMG folder exists but gradient folder doesn't
        elif (os.path.exists(HEATMAP_IMG_PATH)) and (not os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            if (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES):
                cprintf(f"Heatmap folder exists, but there are less heatmaps than there should be.", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_PATH)
                cprintf(f"Switching to new_calc mode ...", 'yellow')
                MODE = 'new_calc'
            else:
                cprintf(f"Heatmap IMG folder exists and is complete, but gradient folder doesn't. Creating image folder at {HEATMAP_IMG_GRADIENT_PATH}", 'l_blue')
                os.makedirs(HEATMAP_IMG_GRADIENT_PATH)
                MODE = 'gradient_calc'

        # 3- Heatmap IMG folder doesn't exist but gradient folder does
        elif (not os.path.exists(HEATMAP_IMG_PATH)) and (os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            if (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
                cprintf(f"Gradient folder exists, but there are less gradients than there should be.", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
                cprintf(f"Switching to new_calc mode ...", 'yellow')
                MODE = 'new_calc'
            else:
                cprintf(f"Heatmap IMG folder doesn't exist but gradient folder exits and is complete. Creating image folder at {HEATMAP_IMG_PATH}", 'l_blue')
                os.makedirs(HEATMAP_IMG_PATH)
                MODE = 'heatmap_calc'

        # 4- Both folders exist, but there are less heatmaps/gradients than there should be
        elif (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Both folders exist, but there are less heatmaps/gradients than there should be.", 'yellow')
            cprintf(f"Deleting folder at {HEATMAP_IMG_PATH} and {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
            shutil.rmtree(HEATMAP_IMG_PATH)
            shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
            MODE = 'new_calc'

        # 5- both folders exist, but there are less heatmaps than there should be (gradients are the correct number)
        elif (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) == NUM_OF_FRAMES-1):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Both folders exist, but there are less heatmaps than there should be (gradients are the correct number).", 'yellow')
            cprintf(f"Deleting folder at {HEATMAP_IMG_PATH}", 'yellow')
            shutil.rmtree(HEATMAP_IMG_PATH)
            MODE = 'heatmap_calc'

        # 6- gradient folder exists, but there are less gradients than there should be (heatmaps are the correct number)
        elif (len(os.listdir(HEATMAP_IMG_PATH)) == NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Both folders exist, but there are less gradients than there should be (heatmaps are the correct number).", 'yellow')
            cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
            shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
            MODE = 'gradient_calc'

        # 3- heatmap folder exists and correct number of heatmaps, but no csv file was generated.
        elif not 'driving_log.csv' in os.listdir(HEATMAP_FOLDER_PATH):
            validation_warnings(False, sim_name, attention_type, run_id, nominal, threshold)
            cprintf(f"Correct number of heatmaps exist. CSV File doesn't.", 'yellow')
            MODE = 'csv_missing'
            
        else:
            MODE = None
            validation_warnings(True, sim_name, attention_type, run_id, nominal, threshold)

        if (MODE != 'csv_missing') and (MODE != 'heatmap_calc') and (MODE != 'new_calc'):
            # check if img paths in the csv file need correcting (possible directory change of simulation data)
            print(MODE)
            correct_img_paths_in_csv_files(HEATMAP_CSV_PATH)
        if MODE is not None:
            compute_heatmap(cfg, nominal, sim_name, NUM_OF_FRAMES, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, MODE)
    # # check if img paths in the csv file need correcting (possible directory change of simulation data)
    # correct_img_paths_in_csv_files(HEATMAP_CSV_PATH)
    PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH]
    if threshold:
        calc_distances = False
        THRESHOLD_VECTORS_FOLDER_PATH = HEATMAP_FOLDER_PATH
        NOMINAL_PATHS, NUM_FRAMES_NOM, SIMULATION_NAME_ANOMALOUS, SIMULATION_NAME_NOMINAL, DISTANCE_TYPES, ANALYSE_DISTANCE = threshold_extras
        for distance_type in DISTANCE_TYPES:
            if f'dist_vect_{distance_type}.csv' not in os.listdir(THRESHOLD_VECTORS_FOLDER_PATH):
                cprintf(f"WARNING: Threshold sim distance data of distance type {distance_type} is unavailable. Calculating distances ...", "red")
                calc_distances = True
        if calc_distances:
            evaluate_p2p_failure_prediction(cfg,
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
                                            threshold_sim = True)

        return THRESHOLD_VECTORS_FOLDER_PATH
    else:
        return NUM_OF_FRAMES, PATHS


def validation_warnings(valid, sim_name, attention_type, run_id, nominal, threshold):
    if valid:
        if nominal:
            cprintf(f"Heatmaps for nominal \u2713 \"{sim_name}\" of attention type \"{attention_type}\" exist.", 'l_green')
        elif threshold:
            cprintf(f"Heatmaps for threshold sim \u2713 \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" exist.", 'l_green')
        else:
            cprintf(f"Heatmaps for anomalous sim \u2713 \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" exist.", 'l_green')
    else:
        if nominal:
            cprintf(f"WARNING: Nominal \u2717 sim \"{sim_name}\" of attention type \"{attention_type}\" heatmap folder has deficiencies. Looking for a solution... ", "red")
        elif threshold:
            cprintf(f"WARNING: Threshold \u2717 sim \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" heatmap folder has deficiencies. Looking for a solution... ", "red")
        else:
            cprintf(f"WARNING: Anomalous \u2717 sim \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" heatmap folder has deficiencies. Looking for a solution... ", "red")

#########################################################################################################
# In case the data was copied into another directory the image paths inside csv files must be corrected #
#########################################################################################################
def correct_img_address(img_addr, csv_dir):
    img_name = Path(img_addr).stem
    corrected_path = os.path.join(csv_dir, 'IMG', img_name + '.jpg')
    return corrected_path

def check_addresses(center_img_addresses, csv_dir):
    first_center_img_address = fix_escape_sequences(center_img_addresses[0])
    if not os.path.exists(first_center_img_address):
        corrected_path = correct_img_address(first_center_img_address, csv_dir)
        if not os.path.exists(corrected_path):
            raise ValueError(Fore.RED + f"The provided img path in the csv file is not in the same dir or does not exist: {corrected_path}" + Fore.RESET)
        return False
    else:
        return True
    
def correct_img_paths_in_csv_files(CSV_PATH):

    csv_dir = os.path.dirname(CSV_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    center_img_addresses = csv_file["center"]

    if 'left' in csv_file.columns:
        left_img_addresses = csv_file["left"]
    
    if 'right' in csv_file.columns:
        right_img_addresses = csv_file["right"]

    # if the img exists in the correct path but the path in the csv file is wrong:
    if not check_addresses(center_img_addresses, csv_dir):
        # center images
        for img_addr in center_img_addresses:
            fixed_img_addr = fix_escape_sequences(img_addr)
            corrected_path = correct_img_address(fixed_img_addr, csv_dir)
            csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
        if 'left' in csv_file.columns:
            # left images
            for img_addr in left_img_addresses:
                fixed_img_addr = fix_escape_sequences(img_addr)
                corrected_path = correct_img_address(fixed_img_addr, csv_dir)
                csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
        if 'right' in csv_file.columns:
            # right images
            for img_addr in right_img_addresses:
                fixed_img_addr = fix_escape_sequences(img_addr)
                corrected_path = correct_img_address(fixed_img_addr, csv_dir)
                csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
        
        csv_file.to_csv(CSV_PATH, index=False)

#############################################################################################
# In case of manual training data the simulator doesn't save the field names (column names) #
#############################################################################################
def correct_csv_field_names(SIM_PATH, MAIN_CSV_PATH, run_id):
    UPDATED_CSV_PATH = os.path.join(SIM_PATH, "src", run_id, "driving_log_updated.csv")
    try:
        # autonomous mode simulation data (is going to fail if manual, since manual has address string in first column)
        data_df = pd.read_csv(MAIN_CSV_PATH)
        NUM_OF_FRAMES = pd.Series.max(data_df["frameId"]) + 1
    except:
        # manual mode simulation data
        with open(MAIN_CSV_PATH,'r') as f:
            lines = f.readlines()[1:]
            with open(UPDATED_CSV_PATH,'w') as f1:
                 for line in lines:
                    f1.write(line)
        df = pd.read_csv(UPDATED_CSV_PATH, header=None)
        df.rename(columns={ 0: utils.csv_fieldnames_in_manual_mode[0], 1: utils.csv_fieldnames_in_manual_mode[1],
                            2: utils.csv_fieldnames_in_manual_mode[2], 3: utils.csv_fieldnames_in_manual_mode[3],
                            4: utils.csv_fieldnames_in_manual_mode[4], 5: utils.csv_fieldnames_in_manual_mode[5],
                            6: utils.csv_fieldnames_in_manual_mode[6], 7: utils.csv_fieldnames_in_manual_mode[7],
                            8: utils.csv_fieldnames_in_manual_mode[8], 9: utils.csv_fieldnames_in_manual_mode[9],
                            10: utils.csv_fieldnames_in_manual_mode[10]}, inplace=True)
        df['frameId'] = df.index
        df.to_csv(MAIN_CSV_PATH, index=False) # save to new csv file
        os.remove(UPDATED_CSV_PATH)
        data_df = pd.read_csv(MAIN_CSV_PATH)
        NUM_OF_FRAMES = pd.Series.max(data_df["frameId"]) + 1
    return data_df


def get_num_frames(run_id, SIM_PATH, MAIN_CSV_PATH):
    # add field names to main csv file if it's not there (manual training recording)
    csv_df = correct_csv_field_names(SIM_PATH, MAIN_CSV_PATH, run_id)
    # get number of frames
    NUM_OF_FRAMES = pd.Series.max(csv_df['frameId']) + 1
    return NUM_OF_FRAMES

def comparison_plot_setup(comp_fig):
        comp_spec = comp_fig.add_gridspec(nrows=4, ncols=1, width_ratios= [1], height_ratios=[3, 3, 1, 1])
        pca_ax_nom = comp_fig.add_subplot(comp_spec[0, :], projection='3d')
        pca_ax_nom.set_title('Nominal PCAs')
        pca_ax_ano = comp_fig.add_subplot(comp_spec[1, :], projection='3d')
        pca_ax_ano.set_title('Anomalous PCAs')
        position_ax = comp_fig.add_subplot(comp_spec[2, :])
        position_ax.set_title('Positional Mappings')
        distance_ax = comp_fig.add_subplot(comp_spec[3, :])
        distance_ax.set_title('Distance Vectors')
        axes = [pca_ax_nom, pca_ax_ano, position_ax, distance_ax]
        return axes, comp_fig

def copy_run_figs(cfg, SIM_PATH, run_id, run_figs):
    if cfg.SPARSE_ATTRIBUTION:
        RUN_FIGS_FOLDER_PATH = os.path.join(SIM_PATH, 'results', str(run_id), 'FIGS_SPARSE')
    else:
        RUN_FIGS_FOLDER_PATH = os.path.join(SIM_PATH, 'results', str(run_id), 'FIGS')

    if not os.path.exists(RUN_FIGS_FOLDER_PATH):
        cprintf(f'Run figure folder does not exist. Creating folder ...' ,'l_blue')
        os.makedirs(RUN_FIGS_FOLDER_PATH)
    
    cprintf(f"Copying run figures of all assigned heatmap types to: {RUN_FIGS_FOLDER_PATH}", 'l_cyan')
    for run_fig_address in tqdm(run_figs):
        shutil.copy(run_fig_address, RUN_FIGS_FOLDER_PATH)


def delete_contents_except(directory, directory_to_keep):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # Skip the directory you want to keep
            if item_path != directory_to_keep:
                shutil.rmtree(item_path)
        else:
            # Delete files
            os.remove(item_path)



def f_beta_score(precision, recall, beta=3):
    numerator = (1 + beta ** 2) * (precision * recall)
    denominator = (beta ** 2 * precision) + recall
    f_beta_score = numerator / denominator
    return f_beta_score






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
                        
        ANO_SIMULATIONS = [
                            # 'track1-night-moon',
                            # 'track1-day-fog-100',
                            # 'track1-day-rain-100',
                            # 'track1-day-snow-100',
                            # 'track1-day-sunny',
                            # 'track1-night-rain-100',
                            'track1-night-fog-100',
                            'track1-night-snow-100',
                        ]

        
        NOM_SIMULATIONS = [
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            # cfg.BASE_NOMINAL_SUNNY_SIM,
                            cfg.BASE_NOMINAL_SUNNY_SIM,
                            cfg.BASE_NOMINAL_SUNNY_SIM
                        ]


        THRESHOLD_SIMULATIONS = [
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    # cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    cfg.BASE_THRESHOLD_SUNNY_SIM,
                                    cfg.BASE_THRESHOLD_SUNNY_SIM
                                ]
        
        RUN_ID_NUMBERS = [
                            # [1, 2],
                            # [1],
                            # [1],
                            # [1],
                            # [1],
                            # [1, 2],
                            [1, 2],
                            [1, 2]
                            ]
        
        SUMMARY_COLLAGES = [
                            # [False, False],
                            # [False],
                            # [False],
                            # [False],
                            # [False],
                            # [False, False],
                            [False, False],
                            [False, False]
                            ]
        HEATMAP_TYPES = ['SmoothGrad'] #, 'GradCam++', 'RectGrad', 'RectGrad_PRR', 'Saliency', 'Guided_BP', 'Gradient-Input', 'IntegGrad', 'Epsilon_LRP']


        if len(ANO_SIMULATIONS) != len(NOM_SIMULATIONS):
            raise ValueError(Fore.RED + f"Mismatch in number of specified ANO and NOM simulations: {len(ANO_SIMULATIONS)} != {len(NOM_SIMULATIONS)} " + Fore.RESET)
        elif len(ANO_SIMULATIONS) != len(RUN_ID_NUMBERS):
            raise ValueError(Fore.RED + f"Mismatch in number of runs and specified simulations: {len(ANO_SIMULATIONS)} != {len(RUN_ID_NUMBERS)} " + Fore.RESET)
        elif len(SUMMARY_COLLAGES) != len(RUN_ID_NUMBERS):
            raise ValueError(Fore.RED + f"Mismatch in number of runs and specified summary collage patterns: {len(SUMMARY_COLLAGES)} != {len(RUN_ID_NUMBERS)} " + Fore.RESET)
        
    # DISTANCE_TYPES = ['euclidean', 'manhattan', 'cosine', 'EMD', 'pearson', 'spearman', 'kendall', 'moran', 'kl-divergence', 'mutual-info', 'sobolev-norm']
        DISTANCE_TYPES = ['sobolev-norm']
        ANALYSE_DISTANCE = {
            'euclidean' : (False, 0.99),
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
        summary_types = ['-avg', '-avg-grad']
        aggregation_methods = ['mean', 'max']
        abstraction_methods = ['avg', 'variance']
        # distance_methods = ['pairwise_distance',
        #                     'cosine_similarity',
        #                     'polynomial_kernel',
        #                     'sigmoid_kernel',
        #                     'rbf_kernel',
        #                     'laplacian_kernel'] #'chi2_kernel'<

        total_runs = 0
        for idx, run_pattern in enumerate(RUN_ID_NUMBERS):
            total_runs += len(run_pattern)
            if len(run_pattern) != len(SUMMARY_COLLAGES[idx]):
                raise ValueError(Fore.RED + f"Mismatch in number of runs per simlation and specified summary collage binary pattern of simulation {idx}: {len(run_pattern)} != {len(SUMMARY_COLLAGES[idx])} " + Fore.RESET)
            
        average_thresholds_path = ''
        prev_sim = ''
        # Starting evaluation
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
                    results_folder_path = os.path.join(cfg.TESTING_DATA_DIR, sim_name, str(run_id))
                    figs_dir = os.path.join(cfg.TESTING_DATA_DIR, sim_name, str(run_id), 'FIGS')
                    if os.path.exists(figs_dir) and os.path.exists(results_folder_path):
                        delete_contents_except(results_folder_path, figs_dir)

                if not os.path.exists(figs_dir):
                    cfg.PLOT_POINT_TO_POINT = True
                else:
                    if len(os.listdir(figs_dir)) < (len(HEATMAP_TYPES)*3):
                        cfg.PLOT_POINT_TO_POINT = True

                # check whether nominal and anomalous simulation and the corresponding heatmaps are already generated, generate them otherwise
                for heatmap_type in HEATMAP_TYPES:

                    cprintb(f'\n########### Simulation {sim_name} ({sim_idx + 1} of {len(ANO_SIMULATIONS)}) ###########', 'l_red')
                    cprintb(f'########### Nominal Sim: {SIMULATION_NAME_NOMINAL}  ###########', 'l_red')
                    if cfg.METHOD == 'p2p':
                        cprintb(f'########### Threshold Sim: {SIMULATION_NAME_THRESHOLD}  ###########', 'l_red')
                    cprintb(f'\n############## run number {run_id} of {len(RUN_ID_NUMBERS[sim_idx])} ##############', 'l_blue')
                    cprintb(f'########### Using Heatmap Type: {heatmap_type} ({HEATMAP_TYPES.index(heatmap_type) + 1} of {len(HEATMAP_TYPES)}) ###########', 'l_blue')

                    NUM_FRAMES_NOM, NOMINAL_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_NOMINAL, attention_type=heatmap_type, sim_type='nominal')
                    NUM_FRAMES_ANO, ANOMALOUS_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_ANOMALOUS, attention_type=heatmap_type, sim_type='anomalous')
                    THRESHOLD_VECTORS_FOLDER_PATH = simExists(cfg, '1', sim_name=SIMULATION_NAME_THRESHOLD, attention_type=heatmap_type, sim_type='threshold',
                                                                threshold_extras=[NOMINAL_PATHS,
                                                                                NUM_FRAMES_NOM,
                                                                                SIMULATION_NAME_ANOMALOUS,
                                                                                SIMULATION_NAME_NOMINAL,
                                                                                DISTANCE_TYPES,
                                                                                ANALYSE_DISTANCE])
                    ANOMALOUS_PATHS.append(THRESHOLD_VECTORS_FOLDER_PATH)
                    
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

                        fig_img_address, results_csv_path, results_folder_path, seconds_to_anticipate_list = evaluate_p2p_failure_prediction(cfg,
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
                                                                                                                                            threshold_sim = False)
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
                            writer.writerow(["time_stamp","heatmap_type", "sta", "TP", "FP", "TN", "FN", "precision", "recall", "accuracy", "fpr","TP_all", "FP_all", "TN_all", "FN_all", "precision_all", "recall_all", "accuracy_all", "fpr_all"])
                                    
                            for heatmap_type in HEATMAP_TYPES:
                                for sta in seconds_to_anticipate_list:
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
                                for sta in seconds_to_anticipate_list:
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
            prev_sim = sim_name



        ##################### Evaluation of Results #####################
        # results_csv_path = r"D:\ThirdEye\ase22\simulations\track1-night-moon-anomalous\1\averaged_theshold\results_ano_track1-night-moon-anomalous_nom_track1-night-moon-nominal_total_scores_heatmaps.csv"
        # # results_csv_path = results_csv_path.replace("\\", "\\\\")
        # print(results_csv_path)
        # results_df = pd.read_csv(results_csv_path)
        # seconds_to_anticipate_list = [1, 2, 3]

        # # for heatmap_type in HEATMAP_TYPES:
        # for sta in seconds_to_anticipate_list:
        #     filter_by_sta = results_df[(results_df['sta'] == sta)]

        #     precision = filter_by_sta['precision'].values
        #     avg_precision = np.average(precision)
        #     # print(f'sta: precision: {sta}: {precision}')
        #     cprintf(f'sta: avg_precision: {sta}: {round(avg_precision*100)}', 'l_green')

        #     recall = filter_by_sta['recall'].values
        #     avg_recall = np.average(recall)
        #     # print(f'sta: recall: {sta}: {recall}')
        #     cprintf(f'sta: avg_recall: {sta}: {round(avg_recall*100)}', 'l_yellow')

        #     f3_score = f_beta_score(avg_precision, avg_recall, beta=3)
        #     cprintf(f'sta: f3_score: {sta}: {round(f3_score*100)}', 'l_red')

        #     accuracy = filter_by_sta['accuracy'].values
        #     avg_accuracy = np.average(accuracy)
        #     cprintf(f'sta: avg_accuracy: {sta}: {round(avg_accuracy*100)}', 'l_blue')

        # filter_by_sta = results_df[(results_df['sta'] == 1)]

        # precision_all = filter_by_sta['precision_all'].values
        # avg_precision_all = np.average(precision_all)
        # # print(f'sta: precision: all: {precision_all}')
        # cprintf(f'sta: avg_precision_all: all: {round(avg_precision_all*100)}', 'l_green')

        # recall_all = filter_by_sta['recall_all'].values
        # avg_recall_all = np.average(recall_all)
        # # print(f'sta: precision: all: {recall_all}')
        # cprintf(f'sta: avg_recall_all: all: {round(avg_recall_all*100)}', 'l_yellow')


        # f3_score_all = f_beta_score(avg_precision_all, avg_recall_all, beta=3)
        # cprintf(f'sta: f3_score_all: {sta}: {round(f3_score_all*100)}', 'l_red')

        # accuracy_all = filter_by_sta['accuracy_all'].values
        # avg_accuracy_all = np.average(accuracy_all)
        # cprintf(f'sta: avg_accuracy_all: all: {round(avg_accuracy_all*100)}', 'l_blue')











        end_time = time.monotonic()
        cprintf(f"Completed {total_runs} evaluation run(s) of {len(ANO_SIMULATIONS)} simulation(s) in {timedelta(seconds=end_time-start_time)}", 'yellow')
