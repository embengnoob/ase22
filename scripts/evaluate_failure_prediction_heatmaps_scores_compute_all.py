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

from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction, evaluate_p2p_failure_prediction, get_OOT_frames, test

def simExists(cfg, run_id, sim_name, attention_type, nominal):
    SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, sim_name)
    MAIN_CSV_PATH = os.path.join(SIM_PATH, "driving_log.csv")
    HEATMAP_PARENT_FOLDER_PATH = os.path.join(SIM_PATH, "heatmaps-" + attention_type.lower())
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



    NUM_OF_FRAMES = get_num_frames(cfg, sim_name)

    if not os.path.exists(SIM_PATH):
        raise ValueError(Fore.RED + f"The provided simulation path does not exist: {SIM_PATH}" + Fore.RESET)
    else:
        if cfg.SAME_IMG_TEST:
            cprintf(f"Same image test mode for {sim_name} of attention type {attention_type}", 'l_red')
            MODE = 'same_img_test'
    
        else:
            # 1- IMG & Gradient folders don't exist 
            if (not os.path.exists(HEATMAP_IMG_PATH)) and (not os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
                cprintf(f"Neither heatmap IMG folder nor gradient folder exist. Creating image folder at {HEATMAP_IMG_PATH} and {HEATMAP_IMG_GRADIENT_PATH}", 'l_blue')
                os.makedirs(HEATMAP_IMG_PATH)
                os.makedirs(HEATMAP_IMG_GRADIENT_PATH)
                MODE = 'new_calc'

            # 2- Heatmap IMG folder exists but gradient folder doesn't
            elif (os.path.exists(HEATMAP_IMG_PATH)) and (not os.path.exists(HEATMAP_IMG_GRADIENT_PATH)):
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
                cprintf(f"Both folders exist, but there are less heatmaps/gradients than there should be.", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_PATH} and {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_PATH)
                shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
                MODE = 'new_calc'

            # 5- both folders exist, but there are less heatmaps than there should be (gradients are the correct number)
            elif (len(os.listdir(HEATMAP_IMG_PATH)) < NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) == NUM_OF_FRAMES-1):
                cprintf(f"Both folders exist, but there are less heatmaps than there should be (gradients are the correct number).", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_PATH)
                MODE = 'heatmap_calc'

            # 6- gradient folder exists, but there are less gradients than there should be (heatmaps are the correct number)
            elif (len(os.listdir(HEATMAP_IMG_PATH)) == NUM_OF_FRAMES) and (len(os.listdir(HEATMAP_IMG_GRADIENT_PATH)) < NUM_OF_FRAMES-1):
                cprintf(f"Both folders exist, but there are less gradients than there should be (heatmaps are the correct number).", 'yellow')
                cprintf(f"Deleting folder at {HEATMAP_IMG_GRADIENT_PATH}", 'yellow')
                shutil.rmtree(HEATMAP_IMG_GRADIENT_PATH)
                MODE = 'gradient_calc'

            # 3- heatmap folder exists and correct number of heatmaps, but no csv file was generated.
            elif not 'driving_log.csv' in os.listdir(HEATMAP_FOLDER_PATH):
                cprintf(f"Correct number of heatmaps exist. CSV File doesn't.", 'yellow')
                MODE = 'csv_missing'
                
            else:
                MODE = None
                if nominal:
                    cprintf(f"Heatmaps for nominal sim \"{sim_name}\" of attention type \"{attention_type}\" exist.", 'l_green')
                else:
                    cprintf(f"Heatmaps for anomalous sim \"{sim_name}\" of attention type \"{attention_type}\" and run ID \"{run_id}\" exist.", 'l_green')
        
        if MODE is not None:
            compute_heatmap(cfg, nominal, sim_name, NUM_OF_FRAMES, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH, MODE)

    PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH, HEATMAP_IMG_GRADIENT_PATH]
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

def get_num_frames(cfg, sim_name):
    SIM_PATH = os.path.join(cfg.TESTING_DATA_DIR, sim_name)
    MAIN_CSV_PATH = os.path.join(SIM_PATH, "driving_log.csv")
    # add field names to main csv file if it's not there (manual training recording)
    csv_df = correct_field_names(MAIN_CSV_PATH)
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

if __name__ == '__main__':
    start_time = time.monotonic()
    os.chdir(os.getcwd().replace('scripts', ''))

    try:
        cfg = Config("config_my.py")
    except:
        cfg = load_config("config_my.py")

    # cfg.from_pyfile("config_my.py")

    if cfg.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore")

    # ANO_SIMULATIONS = ['test1', 'test2', 'test3', 'test4', 'test5', 'track1-sunny-positioned-nominal-as-anomalous'] # , 'test2', 'test3', 'test4', 'test5'
    # NOM_SIMULATIONS = ['track1-sunny-positioned-nominal',
    #                    'track1-sunny-positioned-nominal',
    #                    'track1-sunny-positioned-nominal',
    #                    'track1-sunny-positioned-nominal',
    #                    'track1-sunny-positioned-nominal',
    #                    'track1-sunny-positioned-nominal']
    # RUN_ID_NUMBERS = [[1, 2, 3],
    #                   [1, 2, 3],
    #                   [1, 2, 3],
    #                   [1, 2, 3],
    #                   [1, 2, 3],
    #                   [1, 2, 3]]
    # SUMMARY_COLLAGES = [[False, False, False],
    #                     [False, False, False],
    #                     [False, False, False],
    #                     [False, False, False],
    #                     [False, False, False],
    #                     [False, False, False]]
    if cfg.EVALUATE_ALL:
        ANO_SIMULATIONS = ['test1', 'test2', 'test3', 'test4', 'test5'] # , 'test2', 'test3', 'test4', 'test5'
        NOM_SIMULATIONS = ['track1-sunny-positioned-nominal',
                           'track1-sunny-positioned-nominal',
                           'track1-sunny-positioned-nominal',
                           'track1-sunny-positioned-nominal',
                           'track1-sunny-positioned-nominal']
        RUN_ID_NUMBERS = [[1],
                          [1],
                          [1],
                          [1],
                          [1]]
        SUMMARY_COLLAGES = [[False],
                            [False],
                            [False],
                            [False],
                            [False]]
    else:   
        ANO_SIMULATIONS = ['test1'] # , 'test2', 'test3', 'test4', 'test5'
        NOM_SIMULATIONS = ['track1-sunny-positioned-nominal']
        RUN_ID_NUMBERS = [[1]]
        SUMMARY_COLLAGES = [[False, False, False]]
    
    if len(ANO_SIMULATIONS) != len(NOM_SIMULATIONS):
        raise ValueError(Fore.RED + f"Mismatch in number of specified ANO and NOM simulations: {len(ANO_SIMULATIONS)} != {len(NOM_SIMULATIONS)} " + Fore.RESET)
    elif len(ANO_SIMULATIONS) != len(RUN_ID_NUMBERS):
        raise ValueError(Fore.RED + f"Mismatch in number of runs and specified simulations: {len(ANO_SIMULATIONS)} != {len(RUN_ID_NUMBERS)} " + Fore.RESET)
    elif len(SUMMARY_COLLAGES) != len(RUN_ID_NUMBERS):
        raise ValueError(Fore.RED + f"Mismatch in number of runs and specified summary collage patterns: {len(SUMMARY_COLLAGES)} != {len(RUN_ID_NUMBERS)} " + Fore.RESET)
    
    HEATMAP_TYPES = ['SmoothGrad'] #'GradCam++', 'SmoothGrad', 'RectGrad', 'RectGrad_PRR', 'Saliency', 'Guided_BP', 'SmoothGrad_2', 'Gradient-Input', 'IntegGrad', 'Epsilon_LRP'
  # DISTANCE_TYPES = ['euclidean', 'manhattan', 'cosine', 'EMD', 'pearson', 'spearman', 'kendall', 'moran', 'kl-divergence', 'mutual-info', 'sobolev-norm']
    DISTANCE_TYPES = ['euclidean', 'manhattan', 'cosine', 'EMD', 'pearson', 'spearman', 'kendall', 'moran', 'mutual-info', 'sobolev-norm']
    summary_types = ['-avg', '-avg-grad']
    aggregation_methods = ['mean', 'max']
    abstraction_methods = ['avg', 'variance']
    # distance_methods = ['pairwise_distance',
    #                     'cosine_similarity',
    #                     'polynomial_kernel',
    #                     'sigmoid_kernel',
    #                     'rbf_kernel',
    #                     'laplacian_kernel'] #'chi2_kernel'

    total_runs = 0
    for idx, run_pattern in enumerate(RUN_ID_NUMBERS):
        total_runs += len(run_pattern)
        if len(run_pattern) != len(SUMMARY_COLLAGES[idx]):
            raise ValueError(Fore.RED + f"Mismatch in number of runs per simlation and specified summary collage binary pattern of simulation {idx}: {len(run_pattern)} != {len(SUMMARY_COLLAGES[idx])} " + Fore.RESET) 
    
    # Starting evaluation
    for sim_idx, sim_name in enumerate(ANO_SIMULATIONS):
    
        num_of_frames = get_num_frames(cfg, sim_name)
        # Number between 0 and min(n_samples, n_features)
        PCA_DIMENSIONS = [100] # [100, 500, num_of_frames]

        # if cfg.COMPARE_RUNS:
        comparison_figs = []
        # General run comparison
        gen_axes, gen_comp_fig = comparison_plot_setup(plt.figure(figsize=(20,15), constrained_layout=False))
        # PCA_based run comparison
        pca_axes_list = []
        pca_comp_fig_list = []
        for pca_dim in PCA_DIMENSIONS:
            pca_axes, pca_comp_fig = comparison_plot_setup(plt.figure(figsize=(20,15), constrained_layout=False))
            pca_axes_list.append(pca_axes)
            pca_comp_fig_list.append(pca_comp_fig)

        run_results = []
        run_keys = []                
        for run_number in range(len(RUN_ID_NUMBERS[sim_idx])):
            # Check if a simulation with this run number already exists
            run_id = RUN_ID_NUMBERS[sim_idx][run_number]
            cprintb(f'\n\n########### Simulation {sim_name} ########### run number {run_number+1} ############## run id {run_id} ##############', 'l_red')
            SIMULATION_NAME_ANOMALOUS = sim_name
            SIMULATION_NAME_NOMINAL = NOM_SIMULATIONS[sim_idx]
            cfg.SIMULATION_NAME = SIMULATION_NAME_ANOMALOUS
            cfg.SIMULATION_NAME_NOMINAL = SIMULATION_NAME_NOMINAL
            cfg.GENERATE_SUMMARY_COLLAGES = SUMMARY_COLLAGES[sim_idx][run_number]

            # check whether nominal and anomalous simulation and the corresponding heatmaps are already generated, generate them otherwise
            for heatmap_type in HEATMAP_TYPES:
                if not cfg.SAME_IMG_TEST:
                    NUM_FRAMES_NOM, NOMINAL_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_NOMINAL, attention_type=heatmap_type, nominal=True) 
                    NUM_FRAMES_ANO, ANOMALOUS_PATHS = simExists(cfg, str(run_id), sim_name=SIMULATION_NAME_ANOMALOUS, attention_type=heatmap_type, nominal=False)
            
                if not cfg.SAME_IMG_TEST:
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

                    # elif cfg.METHOD == 'p2p':
                    #     figsize = (15, 12)
                    #     hspace = 0.69
                    #     fig, axs = plt.subplots(len(abstraction_methods)*len(PCA_DIMENSIONS)*len(HEATMAP_TYPES), 1, figsize=figsize)
                    #     plt.subplots_adjust(hspace=hspace)
                    #     plt.suptitle("P2P Heatmap Distances", fontsize=15, y=0.95)
                    #     for ht in HEATMAP_TYPES:
                    #         for am in abstraction_methods:
                    #             for dim in PCA_DIMENSIONS:
                    #                 for dm in DISTANCE_METHODS:
                    #                     cprintf(f'\n########### using distance method >>{dm}<< ########### pca dimension {dim} ##############', 'yellow')
                    #                     evaluate_p2p_failure_prediction(cfg,
                    #                                                     heatmap_type=ht,
                    #                                                     heatmap_types = HEATMAP_TYPES,
                    #                                                     anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                    #                                                     nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                    #                                                     distance_method=dm,
                    #                                                     distance_methods = DISTANCE_METHODS,
                    #                                                     pca_dimension=dim,
                    #                                                     pca_dimensions = PCA_DIMENSIONS,
                    #                                                     abstraction_method = am,
                    #                                                     abstraction_methods = abstraction_methods,
                    #                                                     fig=fig,
                    #                                                     axs=axs)
                    #     plt.show()
                    elif cfg.METHOD == 'test':
                        # if cfg.NOM_VS_NOM_TEST:
                        #     pca_values = []
                        #     pca_keys = []
                        for pca_dimension in PCA_DIMENSIONS:
                            cprintb(f'\n########### run number {run_number+1} ############## run id {run_id} ##############', 'l_blue')
                            cprintb(f'########### Using PCA Dimension: {pca_dimension} ###########', 'l_blue')
                            cprintb(f'########### Using Heatmap Type: {heatmap_type} ###########', 'l_blue')
                            x_ano_all_frames, x_nom_all_frames, pca_ano, pca_nom = test(cfg,
                                                                                        NOMINAL_PATHS,
                                                                                        ANOMALOUS_PATHS,
                                                                                        NUM_FRAMES_NOM,
                                                                                        NUM_FRAMES_ANO,
                                                                                        heatmap_type=heatmap_type,
                                                                                        anomalous_simulation_name=SIMULATION_NAME_ANOMALOUS,
                                                                                        nominal_simulation_name=SIMULATION_NAME_NOMINAL,
                                                                                        distance_types=DISTANCE_TYPES,
                                                                                        pca_dimension=pca_dimension,
                                                                                        PCA_DIMENSIONS=PCA_DIMENSIONS,
                                                                                        run_id=run_id,
                                                                                        gen_axes=gen_axes,
                                                                                        pca_axes_list=pca_axes_list)
            #                 if cfg.NOM_VS_NOM_TEST:
            #                     pca_values.append([x_ano_all_frames, x_nom_all_frames, pca_ano, pca_nom])
            #                     pca_keys.append([f'x_ano_all_frames-{pca_dimension}d', f'x_nom_all_frames-{pca_dimension}d', f'pca_ano-{pca_dimension}d', f'pca_nom-{pca_dimension}d'])

            #     if cfg.NOM_VS_NOM_TEST:
            #         run_results.append(pca_values)
            #         run_keys.append(pca_keys)
            # if cfg.NOM_VS_NOM_TEST:
            #     COMP_VARIABLES = ['x_ano_all_frames', 'x_nom_all_frames', 'pca_ano', 'pca_nom']
            #     for pca_idx in range(3):
            #         for var_idx in range(4):
            #             for run_num in range(3):
            #                 print(f'run num:{run_num+1}, pca_dim:{PCA_DIMENSIONS[pca_idx]}[{pca_idx}], len({COMP_VARIABLES[var_idx]}): {run_results[run_num][pca_idx][var_idx].shape}')

                # for pca_dim_idx in range(3):
                #     for compared_var_idx in range(4):
                #         print(f"\npca_dim_idx: {pca_dim_idx}, compared_var_idx: {compared_var_idx}")    
                #         if np.array_equal(run_results[0][pca_dim_idx][compared_var_idx], run_results[1][pca_dim_idx][compared_var_idx]):
                #             pass
                #         else:
                #             term1 = run_results[0][pca_dim_idx][compared_var_idx]
                #             term2 = run_results[1][pca_dim_idx][compared_var_idx]
                #             print(f'1: run1 {run_keys[0][pca_dim_idx][compared_var_idx]}: {term1.shape}')
                #             print(f'2: run2 {run_keys[1][pca_dim_idx][compared_var_idx]}: {term2.shape}')
                #             cprintf(f"different results run1 vs run2, pca_dim:{PCA_DIMENSIONS[pca_dim_idx]}, {COMP_VARIABLES[compared_var_idx]}", 'l_red')
                #             num_of_different = np.sum(term1 == term2)
                #             cprintf(f"number of different results: {num_of_different} of {term1.shape[0]*term1.shape[1]}\n", 'l_yellow')


                        # if np.array_equal(run_results[1][pca_dim_idx][compared_var_idx], run_results[2][pca_dim_idx][compared_var_idx]):
                        #     pass
                        # else:
                        #     term1 = run_results[1][pca_dim_idx][compared_var_idx]
                        #     term2 = run_results[2][pca_dim_idx][compared_var_idx]
                        #     print(f'1: run2 {run_keys[1][pca_dim_idx][compared_var_idx]}: {term1.shape}')
                        #     print(f'2: run3 {run_keys[2][pca_dim_idx][compared_var_idx]}: {term2.shape}')
                        #     cprintf(f"different results run2 vs run3, pca_dim:{PCA_DIMENSIONS[pca_dim_idx]}, {COMP_VARIABLES[compared_var_idx]}", 'l_red')
                        #     num_of_different = np.sum(term1 == term2)
                        #     cprintf(f"number of different results: {num_of_different} of {term1.shape[0]*term1.shape[1]}\n", 'l_yellow')

                        # if np.array_equal(run_results[0][pca_dim_idx][compared_var_idx], run_results[2][pca_dim_idx][compared_var_idx]):
                        #     pass
                        # else:
                        #     term1 = run_results[0][pca_dim_idx][compared_var_idx]
                        #     term2 = run_results[2][pca_dim_idx][compared_var_idx]
                        #     print(f'1: run1 {run_keys[0][pca_dim_idx][compared_var_idx]}: {term1.shape}')
                        #     print(f'2: run3 {run_keys[2][pca_dim_idx][compared_var_idx]}: {term2.shape}')
                        #     cprintf(f"different results run1 vs run3, pca_dim:{PCA_DIMENSIONS[pca_dim_idx]}, {COMP_VARIABLES[compared_var_idx]}", 'l_red')
                        #     num_of_different = np.sum(term1 == term2)
                        #     cprintf(f"number of different results: {num_of_different} of {term1.shape[0]*term1.shape[1]}\n", 'l_yellow')

            # # save png of run comparisons
            # if cfg.COMPARE_RUNS:
            #     RUN_ID_NUMBERS_STR = '-'.join(str(rn) for rn in RUN_ID_NUMBERS[sim_idx])
            #     ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH = ANOMALOUS_PATHS[2]
            #     cprintf(f'\nSaving comparison figures of runs {RUN_ID_NUMBERS_STR} ...', 'magenta') 
            #     gen_comp_fig.savefig(os.path.join(ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH, f"comparison_of_runs_{RUN_ID_NUMBERS_STR}_all.png"), bbox_inches='tight', dpi=300)
            #     for idx, pca_based_comp_fig in enumerate(pca_comp_fig_list):
            #         pca_based_comp_fig.savefig(os.path.join(ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH, f"comparison_of_runs_{RUN_ID_NUMBERS_STR}_pca_dim_{PCA_DIMENSIONS[idx]}.png"), bbox_inches='tight', dpi=300)

    end_time = time.monotonic()
    cprintf(f"Completed {total_runs} evaluation run(s) of {len(ANO_SIMULATIONS)} simulation(s) in {timedelta(seconds=end_time-start_time)}", 'yellow')
