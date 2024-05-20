import utils
from utils import *
from utils_models import *

######################################################################################
############ EVALUATION FUNCTION FOR THE THIRDEYE METHOD #################
######################################################################################


def evaluate_failure_prediction(cfg, heatmap_type, anomalous_simulation_name, nominal_simulation_name, summary_type,
                                aggregation_method, condition, fig,
                                axs, subplot_counter, number_of_OOTs, run_counter):
    
    print("Using summarization average" if summary_type == '-avg' else "Using summarization gradient")
    print("Using aggregation mean" if aggregation_method == 'mean' else "Using aggregation max")

    # 1. load heatmap scores in nominal conditions
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        nominal_simulation_name,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
    original_losses = np.load(path)
    
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        nominal_simulation_name,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')

    print(f"Path for data_df_nominal: {path}")
    data_df_nominal = pd.read_csv(path)

    data_df_nominal['loss'] = original_losses

    # 2. load heatmap scores in anomalous conditions
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        anomalous_simulation_name,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
    
    anomalous_losses = np.load(path)
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        anomalous_simulation_name,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')

    print(f"Path for data_df_anomalous: {path}")
    data_df_anomalous = pd.read_csv(path)
    data_df_anomalous['loss'] = anomalous_losses

    # 3. compute a threshold from nominal conditions, and FP and TN
    false_positive_windows, true_negative_windows, threshold, subplot_counter = compute_fp_and_tn(data_df_nominal,
                                                                                                aggregation_method,
                                                                                                condition,
                                                                                                fig,
                                                                                                axs,
                                                                                                subplot_counter,
                                                                                                run_counter,
                                                                                                cfg.PLOT_NOMINAL,
                                                                                                cfg.PLOT_NOMINAL_ALL)
    
    # 4. compute TP and FN using different time to misbehaviour windows
    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows, subplot_counter = compute_tp_and_fn(data_df_anomalous,
                                                                                                                anomalous_losses,
                                                                                                                threshold,
                                                                                                                seconds,
                                                                                                                fig,
                                                                                                                axs,
                                                                                                                subplot_counter,
                                                                                                                number_of_OOTs,
                                                                                                                run_counter,
                                                                                                                summary_type,
                                                                                                                cfg.PLOT_ANOMALOUS_ALL_WINDOWS,
                                                                                                                cfg.PLOT_THIRDEYE,
                                                                                                                aggregation_method,
                                                                                                                condition)

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)
            accuracy = (true_positive_windows + true_negative_windows) / (
                    true_positive_windows + true_negative_windows + false_positive_windows + false_negative_windows)
            fpr = false_positive_windows / (false_positive_windows + true_negative_windows)

            if precision != 0 or recall != 0:
                f3 = true_positive_windows / (
                        true_positive_windows + 0.1 * false_positive_windows + 0.9 * false_negative_windows)

                print("Accuracy: " + str(round(accuracy * 100)) + "%")
                print("False Positive Rate: " + str(round(fpr * 100)) + "%")
                print("Precision: " + str(round(precision * 100)) + "%")
                print("Recall: " + str(round(recall * 100)) + "%")
                print("F-3: " + str(round(f3 * 100)) + "%\n")
            else:
                precision = recall = f3 = accuracy = fpr = 0
                print("Accuracy: undefined")
                print("False Positive Rate: undefined")
                print("Precision: undefined")
                print("Recall: undefined")
                print("F-3: undefined\n")
        else:
            precision = recall = f3 = accuracy = fpr = 0
            print("Accuracy: undefined")
            print("False Positive Rate: undefined")
            print("Precision: undefined")
            print("Recall: undefined")
            print("F-1: undefined")
            print("F-3: undefined\n")

        # 5. write results in a CSV files
        if not os.path.exists(heatmap_type + '-' + str(condition) + '.csv'):
            with open(heatmap_type + '-' + str(condition) + '.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ["heatmap_type", "summarization_method", "aggregation_type", "simulation_name", "failures",
                     "detected", "undetected", "undetectable", "ttm", 'accuracy', "fpr", "precision", "recall",
                     "f3"])
                writer.writerow([heatmap_type, summary_type[1:], aggregation_method, anomalous_simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

        else:
            with open(heatmap_type + '-' + str(condition) + '.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow([heatmap_type, summary_type[1:], aggregation_method, anomalous_simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

    K.clear_session()
    gc.collect()
    return subplot_counter


def compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, threshold, seconds_to_anticipate, fig,
                      axs, subplot_counter, number_of_OOTs, run_counter, summary_type, PLOT_ANOMALOUS_ALL_WINDOWS=True,
                      PLOT_THIRDEYE=True, aggregation_method='mean', cond='ood'):
    
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&& time to misbehaviour (s): %d &&&&&&&&&&&&&&&&&&&&&&&&&&&&" % seconds_to_anticipate)
    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    '''
        prepare dataset to get TP and FN from unexpected
    '''
    if cond == "icse20":
        number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
        fps_anomalous = 15  # only for icse20 configurations
    else:
        number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
        simulation_time_anomalous = pd.Series.max(data_df_anomalous['time'])
        fps_anomalous = number_frames_anomalous // simulation_time_anomalous

    # creates the ground truth
    all_first_frame_position_OOT_sequences, OOT_anomalous_in_anomalous_conditions = get_OOT_frames(data_df_anomalous, number_frames_anomalous)
    print("identified %d OOT(s)" % len(all_first_frame_position_OOT_sequences))
    print(all_first_frame_position_OOT_sequences)
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence / length of time window (seconds to anticipate) in terms of number of frames

    # first frame n seconds before the failure / length of time window one second shorter
    # than seconds_to_anticipate in terms of number of frame
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)
    # frames_to_reassign_2 = 1  # first frame before the failure

    reaction_window = pd.Series()
    (reaction_window_x_min, reaction_window_x_max) = (0,0)

###################################################################################################################################
    if PLOT_ANOMALOUS_ALL_WINDOWS:
        # anomalous cross track errors
        cte_anomalous = data_df_anomalous['cte']
        # car speed in anomaluos mode
        speed_anomalous = data_df_anomalous['speed']
        # anomalous losses
        sma_anomalous = pd.Series(losses_on_anomalous)
        # calculate simulation time window and cut extra samples
        num_windows_anomalous = len(data_df_anomalous) // fps_anomalous
        if len(data_df_anomalous) % fps_anomalous != 0:
            num_to_delete = len(data_df_anomalous) - (num_windows_anomalous * fps_anomalous) - 1
    
            if num_to_delete != 0:
                sma_anomalous_all_win = sma_anomalous[:-num_to_delete]
                cte_anomalous_all_win = cte_anomalous[:-num_to_delete]
                speed_anomalous_all_win = speed_anomalous[:-num_to_delete]
            else:
                sma_anomalous_all_win = sma_anomalous
                cte_anomalous_all_win = cte_anomalous
                speed_anomalous_all_win = speed_anomalous

        list_aggregated = []
        list_aggregated_indexes = []

        # calculate mean or max of windows the length of fps_anomalous starting from idx-fps_anomalous to idx
        for idx, loss in enumerate(sma_anomalous_all_win):

            if idx > 0 and idx % fps_anomalous == 0:

                aggregated_score = None
                if aggregation_method == "mean":
                    aggregated_score = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).mean()
                elif aggregation_method == "max":
                    aggregated_score = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).max()
                elif aggregation_method == "both":
                    aggregated_score_max = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).max()
                    aggregated_score_mean = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).mean()
                    aggregated_score = (aggregated_score_mean + aggregated_score_max)/2

                list_aggregated_indexes.append(idx)
                list_aggregated.append(aggregated_score)

            elif idx == len(sma_anomalous_all_win) - 1:

                aggregated_score = None
                if aggregation_method == "mean":
                    aggregated_score = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).mean()
                elif aggregation_method == "max":
                    aggregated_score = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).max()
                elif aggregation_method == "both":
                    aggregated_score_mean = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).mean()
                    aggregated_score_max = pd.Series(sma_anomalous_all_win.iloc[idx - fps_anomalous:idx]).max()
                    aggregated_score = (aggregated_score_mean + aggregated_score_max)/2
                
                list_aggregated_indexes.append(idx)
                list_aggregated.append(aggregated_score)
        
        print(f'{len(list_aggregated)},{num_windows_anomalous}')
        assert len(list_aggregated) == num_windows_anomalous

        for idx, aggregated_score in enumerate(list_aggregated):
            if aggregated_score >= threshold:
                color = 'red'
            else:
                color = 'cyan'
            axs[run_counter-1].hlines(y=aggregated_score, xmin=list_aggregated_indexes[idx] - fps_anomalous,
                                      xmax=list_aggregated_indexes[idx], color=color, linewidth=3)
###################################################################################################################################

    for item in all_first_frame_position_OOT_sequences:
        subplot_counter += 1
        print("\n(((((((((((((((((((((((analysing failure %d))))))))))))))))))))))))))" % item)
        # for the first frames smaller than a window
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            continue

        # the detection window overlaps with a previous OOT; skip it
        if OOT_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1
        else:
            OOT_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            (reaction_window_x_min, reaction_window_x_max) = (OOT_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2].index[0], OOT_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2].index[-1])
            reaction_window = reaction_window._append(
                OOT_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d \n" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            
            # print(f'Thresholds======================================================================================================>>>>>>{threshold}')

            sma_anomalous_cut = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous_cut)
            # print(type(sma_anomalous_cut))
            # print(sma_anomalous_cut)
            print('>>>>>>>>>>>>>>> aggregation_method: ' + aggregation_method + ' <<<<<<<<<<<<<<<<<')
            # print(sma_anomalous_cut.idxmax(axis=0))
            # print(type(sma_anomalous_cut.index[-1]))

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous_cut.mean()
            elif aggregation_method == "max":
                aggregated_score = sma_anomalous_cut.max()
            elif aggregation_method == "both":
                aggregated_score = (sma_anomalous_cut.mean() + sma_anomalous_cut.max())/2

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous_cut.mean()), str(sma_anomalous_cut.max())))

            print(f"aggregated_score >= threshold: {aggregated_score} >= {threshold}")
            if aggregated_score >= threshold:
                true_positive_windows += 1
            elif aggregated_score < threshold:
                false_negative_windows += 1

            # print(f'****************{reaction_window_x_min}*{reaction_window_x_max}***********************')

            if PLOT_THIRDEYE:
                # plot the 1s window thrsholds in lime green color  
                axs[run_counter-1].hlines(y= aggregated_score, xmin=reaction_window_x_min, xmax=reaction_window_x_max, color='lime', linewidth=3)

            print(f'********************************{subplot_counter}********************************')
            # is it the last run for the current diagram? Then plot everything else
            if (subplot_counter) % (3*len(all_first_frame_position_OOT_sequences)) == 0:
                # choose the right diagram
                ax = axs[run_counter-1]
                if PLOT_THIRDEYE:
                    # plot registered OOT instances
                    for OOT_frame in all_first_frame_position_OOT_sequences:
                        ax.axvline(x = OOT_frame, color = 'teal', linestyle = '--')
                #plot calculated threshold via gamma fitting
                ax.axhline(y = threshold, color = 'r', linestyle = '--')
                # plot loss values
                ax.plot(pd.Series(data_df_anomalous['frameId']), sma_anomalous, label='sma_anomalous', linewidth= 0.5, linestyle = 'dotted')

                if PLOT_ANOMALOUS_ALL_WINDOWS:
                    # plot cross track error values: 
                    # cte > 4: reaching the borders of the track: yellow
                    # 5> cte > 7: on the borders of the track (partial crossing): orange
                    # cte > 7: out of track (full crossing): red
                    yellow_condition = (abs(cte_anomalous_all_win)>3.6)&(abs(cte_anomalous_all_win)<5.0)
                    orange_condition = (abs(cte_anomalous_all_win)>5.0)&(abs(cte_anomalous_all_win)<7.0)
                    red_condition = (abs(cte_anomalous_all_win)>7.0)
                    yellow_ranges = get_ranges(yellow_condition)
                    orange_ranges = get_ranges(orange_condition)
                    red_ranges = get_ranges(red_condition)

                    # plot yellow ranges
                    plot_ranges(yellow_ranges, ax, color='yellow', alpha=0.2)
                    # plot orange ranges
                    plot_ranges(orange_ranges, ax, color='orange', alpha=0.2)
                    # plot red ranges
                    plot_ranges(red_ranges, ax, color='red', alpha=0.2)

                    # plot crash instances: speed < 1.0 
                    crash_condition = (abs(speed_anomalous_all_win)<1.0)
                    # remove the first 10 frames: starting out so speed is less than 1 
                    crash_condition[:10] = False
                    crash_ranges = get_ranges(crash_condition)
                    # plot_ranges(crash_ranges, ax, color='blue', alpha=0.2)
                    NUM_OF_FRAMES_TO_CHECK = 20
                    is_crash_instance = False
                    for rng in crash_ranges:
                        # check 20 frames before first frame with speed < 1.0. if not bigger than 15 it's not
                        # a crash instance it's reset instance
                        if isinstance(rng, list):
                            crash_frame = rng[0]
                        else:
                            crash_frame = rng
                        for speed in speed_anomalous_all_win[crash_frame-NUM_OF_FRAMES_TO_CHECK:crash_frame]:
                            if speed > 15.0:
                                is_crash_instance = True
                        if is_crash_instance == True:
                            is_crash_instance = False
                            reset_frame = crash_frame
                            ax.axvline(x = reset_frame, color = 'blue', linestyle = '--')
                            continue
                        # plot crash ranges (speed < 1.0)
                        if isinstance(rng, list):
                            ax.axvspan(rng[0], rng[1], color=color, alpha=0.2)
                        else:
                            ax.axvspan(rng, rng+1, color=color, alpha=0.2)
                    
                title = f"{aggregation_method} && {summary_type}"
                ax.set_title(title)
                ax.set_ylabel("heatmap scores")
                # ax.set_xlabel("frame number")
                ax.legend(loc='upper left')

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_OOT_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows, subplot_counter


def compute_fp_and_tn(data_df_nominal, aggregation_method, condition,fig,axs,subplot_counter,run_counter, PLOT_NOMINAL, PLOT_NOMINAL_ALL):
    # when conditions == nominal I count only FP and TN
    if condition == "icse20":
        fps_nominal = 15  # only for icse20 configurations
    else:
        number_frames_nominal = pd.Series.max(data_df_nominal['frameId']) # number of total frames in nominal simulation
        simulation_time_nominal = pd.Series.max(data_df_nominal['time']) # total simulation time
        fps_nominal = number_frames_nominal // simulation_time_nominal # fps of nominal simulation

    # calculate simulation time window and cut extra samples
    num_windows_nominal = len(data_df_nominal) // fps_nominal
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
        data_df_nominal = data_df_nominal[:-num_to_delete]

    # taking a rolling window average over heatmap scores (loss)
    losses = pd.Series(data_df_nominal['loss'])
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()
    # print(f"len(losses) ==================================> {len(losses)}")
    # print(f"len(sma_nominal) ==================================> {len(sma_nominal)}")

    list_aggregated = []
    list_aggregated_indexes = []

    # print(f"fps_nominal ==================================>==================================>==================================> {fps_nominal}")
    # calculate mean or max of windows the length of fps_nominal starting from idx-fps_nominal to idx
    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()
            elif aggregation_method == "both":
                aggregated_score_mean = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
                aggregated_score_max = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()
                aggregated_score = (aggregated_score_mean + aggregated_score_max)/2

            list_aggregated_indexes.append(idx)
            list_aggregated.append(aggregated_score)

        elif idx == len(sma_nominal) - 1:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()
            elif aggregation_method == "both":
                aggregated_score_mean = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
                aggregated_score_max = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()
                aggregated_score = (aggregated_score_mean + aggregated_score_max)//2.0
    
            list_aggregated_indexes.append(idx)
            list_aggregated.append(aggregated_score)

    assert len(list_aggregated) == num_windows_nominal

    # calculate threshold
    threshold = get_threshold(list_aggregated, conf_level=0.95)

    false_positive_windows = len([i for i in list_aggregated if i > threshold])
    true_negative_windows = len([i for i in list_aggregated if i <= threshold])

# #############################################################################################################################################################
    if PLOT_NOMINAL:
        if PLOT_NOMINAL_ALL:
            for idx, aggregated_score in enumerate(list_aggregated):
                axs[run_counter-1].hlines(y=aggregated_score, xmin=list_aggregated_indexes[idx] -fps_nominal,
                                        xmax=list_aggregated_indexes[idx], color='cyan', linewidth=3)
            ax.plot(list_aggregated_indexes, list_aggregated, label='agg', linewidth= 0.5, linestyle = 'dotted', color='cyan')
        ax = axs[run_counter-1]
        ax.plot(pd.Series(data_df_nominal['frameId']), sma_nominal, label='sma_nominal', linewidth= 0.5, linestyle = '-', color='g')
        ax.legend(loc='upper left')
#############################################################################################################################################################
    assert false_positive_windows + true_negative_windows == num_windows_nominal
    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    return false_positive_windows, true_negative_windows, threshold, subplot_counter




######################################################################################
############ EVALUATION FUNCTION FOR THE POINT TO POINT (P2P) METHOD #################
######################################################################################

def evaluate_p2p_failure_prediction(cfg, NOMINAL_PATHS, ANOMALOUS_PATHS, NUM_FRAMES_NOM, NUM_FRAMES_ANO, heatmap_type,
                                    anomalous_simulation_name, nominal_simulation_name, distance_types, analyse_distance,
                                    run_id, threshold_sim):  #,averaged_thresholds={}):

    #################################### GETTING THE RIGHT ROOT PATHS #########################################

    # PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH]
    NOMINAL_SIM_PATH = NOMINAL_PATHS[0]
    NOMINAL_MAIN_CSV_PATH = NOMINAL_PATHS[1]
    NOMINAL_HEATMAP_PARENT_FOLDER_PATH = NOMINAL_PATHS[2]
    NOMINAL_HEATMAP_FOLDER_PATH = NOMINAL_PATHS[3]
    NOMINAL_HEATMAP_CSV_PATH = NOMINAL_PATHS[4]
    NOMINAL_HEATMAP_IMG_PATH = NOMINAL_PATHS[5]
    NOMINAL_HEATMAP_IMG_GRADIENT_PATH = NOMINAL_PATHS[6]

    ANOMALOUS_SIM_PATH = ANOMALOUS_PATHS[0]
    ANOMALOUS_MAIN_CSV_PATH = ANOMALOUS_PATHS[1]
    ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH = ANOMALOUS_PATHS[2]
    ANOMALOUS_HEATMAP_FOLDER_PATH = ANOMALOUS_PATHS[3]
    ANOMALOUS_HEATMAP_CSV_PATH = ANOMALOUS_PATHS[4]
    ANOMALOUS_HEATMAP_IMG_PATH = ANOMALOUS_PATHS[5]
    ANOMALOUS_HEATMAP_IMG_GRADIENT_PATH = ANOMALOUS_PATHS[6]
    if not threshold_sim:
        THRESHOLD_VECTORS_FOLDER_PATH = ANOMALOUS_PATHS[7]
    
    #################################### INPUT DATA PRE-PROCESSING #########################################
    if cfg.NOM_VS_NOM_TEST:
        # check if nominal and anomalous heatmaps are exactly the same
        ano_heatmap = pd.read_csv(ANOMALOUS_HEATMAP_CSV_PATH)
        ano_heatmap_center = ano_heatmap["center"]
        print("Read %d anomalous heatmaps from file" % len(ano_heatmap_center))

        nom_heatmap = pd.read_csv(NOMINAL_HEATMAP_CSV_PATH)
        nom_heatmap_center = nom_heatmap["center"]
        print("Read %d nominal heatmaps from file" % len(nom_heatmap_center))

        different_heatmaps = 0
        differences = []
        for ano_idx, ano_img_address in enumerate(tqdm(ano_heatmap_center)):

            # convert Windows path, if needed
            ano_img_address = correct_windows_path(ano_img_address)
            nom_img_address = correct_windows_path(nom_heatmap_center[ano_idx])

            # load image
            ano_img = mpimg.imread(ano_img_address)
            nom_img = mpimg.imread(nom_img_address)

            if np.array_equal(ano_img, nom_img):
               pass
            else:
                different_heatmaps += 1
                num_of_different = np.sum(ano_img == nom_img)
                differences.append(num_of_different)
            # preprocess image
            # x = utils.resize(x).astype('float32')
        
        if different_heatmaps == 0:
            cprintf(f"All ano_img, nom_img are the same", 'l_green')
        else:
            cprintf(f"{different_heatmaps} different heatmaps", 'l_red')

    # 1. find the closest frame from anomalous sim to the frame in nominal sim (comparing car position)
    # nominal simulation
    if not cfg.MINIMAL_LOGGING:
        cprintf(f"Path for data_df_nominal: {NOMINAL_HEATMAP_CSV_PATH}", 'white')
    data_df_nominal = pd.read_csv(NOMINAL_HEATMAP_CSV_PATH)
    nominal = pd.DataFrame(data_df_nominal['frameId'].copy(), columns=['frameId'])
    nominal['position'] = data_df_nominal['position'].copy()
    nominal['center'] = data_df_nominal['center'].copy()
    nominal['steering_angle'] = data_df_nominal['steering_angle'].copy()
    nominal['throttle'] = data_df_nominal['throttle'].copy()
    nominal['speed'] = data_df_nominal['speed'].copy()
    nominal['cte'] = data_df_nominal['cte'].copy()
    # total number of nominal frames
    num_nominal_frames = pd.Series.max(nominal['frameId']) + 1
    if NUM_FRAMES_NOM != num_nominal_frames:
        raise ValueError(Fore.RED + f'Mismatch in number of \"nominal\" frames: {NUM_FRAMES_NOM} != {num_nominal_frames}' + Fore.RESET)
    
    # anomalous simulation
    if not cfg.MINIMAL_LOGGING:
        cprintf(f"Path for data_df_anomalous: {ANOMALOUS_HEATMAP_CSV_PATH}", 'white')
    data_df_anomalous = pd.read_csv(ANOMALOUS_HEATMAP_CSV_PATH)
    anomalous = pd.DataFrame(data_df_anomalous['frameId'].copy(), columns=['frameId'])
    anomalous['position'] = data_df_anomalous['position'].copy()
    anomalous['center'] = data_df_anomalous['center'].copy()
    anomalous['steering_angle'] = data_df_anomalous['steering_angle'].copy()
    anomalous['throttle'] = data_df_anomalous['throttle'].copy()
    anomalous['speed'] = data_df_anomalous['speed'].copy()
    # total number of anomalous frames
    num_anomalous_frames = pd.Series.max(anomalous['frameId']) + 1
    if NUM_FRAMES_ANO != num_anomalous_frames:
        raise ValueError(Fore.RED + f'Mismatch in number of \"anomalous\" frames: {NUM_FRAMES_ANO} != {num_anomalous_frames}' + Fore.RESET)


    #################################### LOCALIZATION AND POSITIONAL MAPPING #########################################

    # path to csv files containing mapped positions and point distances
    POSITIONAL_MAPPING_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'pos_mappings_{nominal_simulation_name}.csv')
    POINT_DISTANCES_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'point_distances_{nominal_simulation_name}.csv')

    nominal_positions = np.zeros((num_nominal_frames, 3), dtype=float)
    closest_point_nominal_positions = np.zeros((num_anomalous_frames, 3), dtype=float)
    anomalous_positions = np.zeros((num_anomalous_frames, 3), dtype=float)

    # check if positional mapping list file in csv format already exists 
    if (not os.path.exists(POSITIONAL_MAPPING_PATH)) or (not os.path.exists(POINT_DISTANCES_PATH)) or (cfg.PLOT_POSITION_CLOUD):
        cprintf(f"Positional mapping list for {anomalous_simulation_name} and {nominal_simulation_name} does not exist. Generating list ...", 'l_blue')
        pos_mappings = np.zeros(num_anomalous_frames, dtype=float)
        point_distances = np.zeros(num_anomalous_frames, dtype=float)
        nominal_positions = np.zeros((num_nominal_frames, 3), dtype=float)
        # cluster of all nominal positions
        cprintf(f"Generating nominal cluster ...", 'l_cyan')
        for nominal_frame in range(num_nominal_frames):
            vector = string_to_np_array(nominal['position'].iloc[nominal_frame], nominal_frame)
            nominal_positions[nominal_frame] = vector
        # compare each anomalous position with the nominal cluster and find the closest nom position (mapping)
        cprintf(f"Number of frames in anomalous conditions:{num_anomalous_frames}", 'l_magenta')
        cprintf(f"Finding the closest point in nom cluster to each anomalous point ...", 'l_cyan')
        for anomalous_frame in tqdm(range(num_anomalous_frames)):
            vector = string_to_np_array(anomalous['position'].iloc[anomalous_frame], anomalous_frame)
            sample_point = vector.reshape(1, -1)
            anomalous_positions[anomalous_frame] = sample_point
            closest_point_row, closest_distance = pairwise_distances_argmin_min(sample_point, nominal_positions)
            closest_point_nominal_positions[anomalous_frame] = nominal_positions[closest_point_row]
            pos_mappings[anomalous_frame] = closest_point_row
            point_distances[anomalous_frame] = closest_distance
        # save list of positional mappings
        cprintf(f"Saving positional mappings to CSV file at {POSITIONAL_MAPPING_PATH} and point distances at {POINT_DISTANCES_PATH} ...", 'magenta')
        np.savetxt(POSITIONAL_MAPPING_PATH, pos_mappings, delimiter=",")
        np.savetxt(POINT_DISTANCES_PATH, point_distances, delimiter=",")
    else:
        cprintf(f"Positional mapping and point distances list exist.", 'l_green')
        # load list of mapped positions
        if not cfg.MINIMAL_LOGGING:
            cprintf(f"Loading CSV file from {POSITIONAL_MAPPING_PATH} and at {POINT_DISTANCES_PATH} ...", 'l_yellow')
        pos_mappings = np.loadtxt(POSITIONAL_MAPPING_PATH, dtype='int')
        point_distances = np.loadtxt(POINT_DISTANCES_PATH, dtype='float')

    # np.set_printoptions(threshold=sys.maxsize)
    print(pos_mappings)
    if len(pos_mappings) != num_anomalous_frames:
        raise ValueError(Fore.RED + f"Length of loaded positional mapping array (from CSV file above) \"{len(pos_mappings)}\" does not match the number of anomalous frames: {num_anomalous_frames}" + Fore.RESET)
    
    if len(point_distances) != num_anomalous_frames:
        raise ValueError(Fore.RED + f"Length of loaded point distance array (from CSV file above) \"{len(point_distances)}\" does not match the number of anomalous frames: {num_anomalous_frames}" + Fore.RESET)
    # np.set_printoptions(threshold=False)

    #################################### ARRAY INITIALISATION #########################################
        
    # initialize arrays
    ht_height, ht_width = get_heatmaps(0, anomalous, nominal, pos_mappings, return_size=True, return_IMAGE=False)
    x_ano_all_frames = np.zeros((num_anomalous_frames, ht_height*ht_width))
    x_nom_all_frames = np.zeros((num_anomalous_frames, ht_height*ht_width))
    closest_nom_cte_all_frames = np.zeros((num_anomalous_frames))

    # Earth mover (wasserstein's) distance(s)
    if 'EMD' in distance_types:
        # distance_vector_EMD = np.zeros((num_anomalous_frames))
        distance_vector_EMD_std = np.zeros((num_anomalous_frames))
        # distance_vector_EMD_norm = np.zeros((num_anomalous_frames))
        # distance_vector_EMD_std_norm = np.zeros((num_anomalous_frames))

    # correlations
    if 'pearson' in distance_types:
        pearson_res = np.zeros((num_anomalous_frames))
        pearson_p = np.zeros((num_anomalous_frames))
    if 'spearman' in distance_types:  
        spearman_res = np.zeros((num_anomalous_frames))
        spearman_p = np.zeros((num_anomalous_frames))
    if 'kendall' in distance_types:  
        kendall_res = np.zeros((num_anomalous_frames))
        kendall_p = np.zeros((num_anomalous_frames))

    # Moran's I: spatial autocorrelation of averaged image (ano, nom)
    if 'moran' in distance_types:
        alpha = 0.7
        beta = (1.0 - alpha)
        moran_i = np.zeros((num_anomalous_frames))
        moran_i_ano = np.zeros((num_anomalous_frames))
        moran_i_nom = np.zeros((num_anomalous_frames))

    # Kullback-Leibler Divergence
    if 'kl-divergence' in distance_types:
        kl_divergence = np.zeros((num_anomalous_frames))
        kl_divergence_std = np.zeros((num_anomalous_frames))

    # Mutual information
    if 'mutual-info' in distance_types:
        mutual_info_std = np.zeros((num_anomalous_frames))

    if 'sobolev-norm' in distance_types:
        sobolev_norms = np.zeros((num_anomalous_frames))

    # distance types: are csv files already available?
    csv_file_available = []
    for distance_type in distance_types:
        # direct distances
        DISTANCE_VECTOR_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                            f'dist_vect_{distance_type}.csv')
        # check if distance/correlation vectors in csv format already exist
        if os.path.exists(DISTANCE_VECTOR_PATH):
            csv_file_available.append(True)
        else:
            csv_file_available.append(False)
    
    #################################### SUMMARY COLLAGES FOLDER CHECK #########################################
    
    if cfg.GENERATE_SUMMARY_COLLAGES:
        missing_collages = 0
        # path to plotted figure images folder
        COLLAGES_PARENT_FOLDER_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                                   f"collages_{anomalous_simulation_name}_{nominal_simulation_name}")
        COLLAGES_BASE_FOLDER_PATH = os.path.join(COLLAGES_PARENT_FOLDER_PATH, "base")

        if not os.path.exists(COLLAGES_BASE_FOLDER_PATH):
            cprintf(f'Collages base folder does not exist. Creating folder ...', 'l_blue')
            os.makedirs(COLLAGES_BASE_FOLDER_PATH)
            cprintf(f'Generating base summary collages, and ...', 'l_cyan')
        elif len(os.listdir(COLLAGES_BASE_FOLDER_PATH)) != num_anomalous_frames:
            cprintf(f'There are missing collages. Deleting and re-creating base folder ...', 'yellow')
            shutil.rmtree(COLLAGES_BASE_FOLDER_PATH)
            os.makedirs(COLLAGES_BASE_FOLDER_PATH)
            cprintf(f'Generating base summary collages, and ...', 'l_cyan')



    cprintf(f'Reshaping positionally corresponding heatmaps into two arrays ...', 'l_cyan')
    if cfg.NOM_VS_NOM_TEST:
        diff_ctr = 0
    for anomalous_frame in tqdm(range(num_anomalous_frames)):

    #################################### HEATMAP PROCESSING AND CORRELATION/DISTANCE CALCULATION #########################################
 
        # load the centeral camera heatmaps of this anomalous frame and the closest nominal frame in terms of position
        ano_heatmap, closest_nom_heatmap, closest_nom_cte = get_heatmaps(anomalous_frame, anomalous, nominal, pos_mappings, return_size=False, return_IMAGE=False, return_cte=True)
        closest_nom_cte_all_frames[anomalous_frame] = closest_nom_cte
        # Moran's I: spatial autocorrelation of averaged image (ano, nom)
        if 'moran' in distance_types and (not csv_file_available[distance_types.index('moran')]):
            dst = cv2.addWeighted(ano_heatmap, alpha, closest_nom_heatmap, beta, 0.0)
            moran_i[anomalous_frame] = Morans_I(dst, plot=False)
            moran_i_ano[anomalous_frame] = Morans_I(ano_heatmap, plot=False)
            moran_i_nom[anomalous_frame] = Morans_I(closest_nom_heatmap, plot=False)

        # convert to grayscale
        x_ano = cv2.cvtColor(ano_heatmap, cv2.COLOR_BGR2GRAY)
        x_nom = cv2.cvtColor(closest_nom_heatmap, cv2.COLOR_BGR2GRAY)

        ano_std_scale = preprocessing.StandardScaler().fit(x_ano)
        x_ano_std = ano_std_scale.transform(x_ano)
        nom_std_scale = preprocessing.StandardScaler().fit(x_nom)
        x_nom_std = nom_std_scale.transform(x_nom)
        # Kullback-Leibler Divergence
        if 'kl-divergence' in distance_types and (not csv_file_available[distance_types.index('kl-divergence')]): 
            kl_divergence_std[anomalous_frame] = entropy(x_ano_std.flatten()//np.sum(x_ano_std), x_nom_std.flatten()//np.sum(x_nom_std))
            # kl_divergence[anomalous_frame] = kl_div(x_ano.flatten(), x_nom.flatten())
        
        # Mutual information
        if 'mutual-info' in distance_types and (not csv_file_available[distance_types.index('mutual-info')]): 
            mutual_info_std[anomalous_frame] = mutual_info_score(x_ano_std.flatten(), x_nom_std.flatten())

        if 'sobolev-norm' in distance_types and (not csv_file_available[distance_types.index('sobolev-norm')]):
            sobolev_norms[anomalous_frame] = h_minus_1_sobolev_norm(x_ano_std, x_nom_std)

        # Earth mover's distances
        if 'EMD' in distance_types and (not csv_file_available[distance_types.index('EMD')]):
            distance_vector_EMD_std[anomalous_frame] = wasserstein_distance(x_ano_std.flatten(), x_nom_std.flatten())

        # Correlation scores
        if 'pearson' in distance_types and (not csv_file_available[distance_types.index('pearson')]):
            pearson_res[anomalous_frame], pearson_p[anomalous_frame] = pearsonr(x_ano_std.flatten(), x_nom_std.flatten())
            # cprintf(f'{pearson_res[anomalous_frame]}', 'cyan')
        if 'spearman' in distance_types and (not csv_file_available[distance_types.index('spearman')]): 
            spearman_res[anomalous_frame], spearman_p[anomalous_frame] = spearmanr(x_ano_std.flatten(), x_nom_std.flatten())
            # cprintf(f'{spearman_res[anomalous_frame]}', 'yellow')
        if 'kendall' in distance_types and (not csv_file_available[distance_types.index('kendall')]): 
            kendall_res[anomalous_frame], kendall_p[anomalous_frame] = kendalltau(x_ano_std.flatten(), x_nom_std.flatten())
            # cprintf(f'{kendall_res[anomalous_frame]}', 'blue')

        x_ano_all_frames[anomalous_frame,:] = x_ano_std.flatten()
        x_nom_all_frames[anomalous_frame,:] = x_nom_std.flatten()

        if cfg.NOM_VS_NOM_TEST:
            if np.array_equal(x_ano_std.flatten(), x_nom_std.flatten(), equal_nan=True):
                # cprintf(f"x_ano_std.flatten(), x_nom_std are the same.flatten()", 'l_green')
                pass
            else:
                num_of_different = np.sum(x_ano_std.flatten() == x_nom_std.flatten())
                cprintf(f"{x_ano_std.flatten() == x_nom_std.flatten()}\n", 'l_red')
                cprintf(f"STD: number of different results: {num_of_different}\n", 'l_red')
                diff_ctr += 1

    #################################### BASE SUMMARY COLLAGES #########################################
        if cfg.GENERATE_SUMMARY_COLLAGES:
            # double-check if every collage actually exists
            img_name = f'FID_{anomalous_frame}.png'
            if img_name in os.listdir(COLLAGES_BASE_FOLDER_PATH):
                continue
            else:
                missing_collages += 1 
            # load central camera images
            ano_img, closest_nom_img = get_images(cfg, anomalous_frame, pos_mappings)
            # load the centeral camera heatmaps of this anomalous frame and the closest nominal frame in terms of position
            ano_heatmap, closest_nom_heatmap = get_heatmaps(anomalous_frame, anomalous, nominal, pos_mappings, return_size=False, return_IMAGE=True)
            # convert to grayscale and resize
            ano_heatmap = ano_heatmap.resize((2*IMAGE_WIDTH,2*IMAGE_HEIGHT)) # .convert('LA')
            closest_nom_heatmap = closest_nom_heatmap.resize((2*IMAGE_WIDTH,2*IMAGE_HEIGHT)) #.convert('LA')
            ano_img = ano_img.resize((2*IMAGE_WIDTH,2*IMAGE_HEIGHT))
            closest_nom_img = closest_nom_img.resize((2*IMAGE_WIDTH,2*IMAGE_HEIGHT))

            alpha = 0.25
            ano_result = Image.blend(ano_heatmap, ano_img, alpha)
            nom_result = Image.blend(closest_nom_heatmap, closest_nom_img, alpha)

            ano_sa = anomalous['steering_angle'].iloc[anomalous_frame]
            nom_sa = nominal['steering_angle'].iloc[int(pos_mappings[anomalous_frame])]
            ano_throttle = anomalous['throttle'].iloc[anomalous_frame]
            nom_throttle = nominal['throttle'].iloc[int(pos_mappings[anomalous_frame])]
            ano_speed = anomalous['speed'].iloc[anomalous_frame]
            nom_speed = nominal['speed'].iloc[int(pos_mappings[anomalous_frame])]

            font = ImageFont.truetype(os.path.join(cfg.TESTING_DATA_DIR,"arial.ttf"), 18)
            draw_ano = ImageDraw.Draw(ano_result)
            draw_ano.text((0, 0),f"Anomalous: steering_angle:{float(ano_sa):.2f} // throttle:{float(ano_throttle):.2f} // speed:{float(ano_speed):.2f}",(255,255,255),font=font)
            draw_nom = ImageDraw.Draw(nom_result)
            draw_nom.text((10, 0),f"Nominal: steering_angle:{float(nom_sa):.2f} // throttle:{float(nom_throttle):.2f} // speed:{float(nom_speed):.2f}",(255,255,255),font=font)

            # create and save collage
            base_collage = Image.new("RGBA", (1280,320))
            base_collage.paste(ano_result, (0,0))
            base_collage.paste(nom_result, (640,0))
            base_collage.save(os.path.join(COLLAGES_BASE_FOLDER_PATH, f'FID_{anomalous_frame}.png'))
            
    if cfg.GENERATE_SUMMARY_COLLAGES:
        if missing_collages > 0 and missing_collages != num_anomalous_frames:
            cprintf(f'There are missing collages. Deleting base folder ...', 'l_red')
            shutil.rmtree(COLLAGES_BASE_FOLDER_PATH)
            raise ValueError(Fore.RED + "Error in number of saved collages. Removing all the collages! Please rerun the code." + Fore.RESET)
        elif missing_collages == 0:
            cprintf(f"All base collages already exist.", 'l_green')
    if not cfg.MINIMAL_LOGGING:    
        print(f'x_ano_std.shape is {x_ano_std.shape}')
        print(f'x_nom_std.shape is {x_nom_std.shape}')
        print(f'x_ano_all_frames.shape is {x_ano_all_frames.shape}')
        print(f'x_nom_all_frames.shape is {x_nom_all_frames.shape}')

    #################################### DISTANCE/CORRELATION VECTORS #########################################
        
    distance_vectors = []
    distance_vectors_avgs = []
    thresholds = {}
    ano_thresholds = {}
    for distance_type in distance_types:
        # direct distances
        DISTANCE_VECTOR_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                            f'dist_vect_{distance_type}.csv')
        if not threshold_sim:
            THRESHOLD_VECTOR_PATH = os.path.join(THRESHOLD_VECTORS_FOLDER_PATH,
                                                f'dist_vect_{distance_type}.csv')
        # check if distance/correlation vectors in csv format already exist
        if not os.path.exists(DISTANCE_VECTOR_PATH):
            cprintf(f"Distance/correlation vector list of type \"{distance_type}\" does not exist. Generating list ...", 'l_blue')
            if (distance_type == 'euclidean') or (distance_type == 'manhattan') or (distance_type == 'cosine'):
                distance_vector = pairwise.paired_distances(abs(x_ano_all_frames), abs(x_nom_all_frames), metric=distance_type)
            elif distance_type == 'moran':
                distance_vector = moran_i
                np.savetxt(os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'dist_vect_moran_i_ano.csv'), moran_i_ano, delimiter=",")
                np.savetxt(os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'dist_vect_moran_i_nom.csv'), moran_i_nom, delimiter=",")
            elif distance_type == 'EMD':
                distance_vector = distance_vector_EMD_std
            elif distance_type == 'pearson':
                distance_vector = pearson_res
            elif distance_type == 'spearman':  
                distance_vector = spearman_res
            elif distance_type == 'kendall':   
                distance_vector = kendall_res
            elif distance_type == 'kl-divergence': 
                distance_vector = kl_divergence_std
            elif distance_type == 'mutual-info': 
                distance_vector = mutual_info_std
            elif distance_type == 'sobolev-norm':
                distance_vector = sobolev_norms
            else:
                raise ValueError(f"Distance type \"{distance_type}\" is not defined.")
            # save distance vector
            cprintf(f"Saving distance vector of type \"{distance_type}\" to CSV file at {DISTANCE_VECTOR_PATH} ...", 'magenta')
            np.savetxt(DISTANCE_VECTOR_PATH, distance_vector, delimiter=",")
            # cprintf(f'{distance_type}', 'yellow')
            # cprintf(f'{distance_vector}', 'blue')
        else:
            cprintf(f"Distance vector list for distance type \"{distance_type}\" already exists.", 'l_green')
            # load distance vector
            # cprintf(f"Loading CSV file for distance type {distance_type} from {DISTANCE_VECTOR_PATH} ...", 'l_yellow')
            distance_vector = np.loadtxt(DISTANCE_VECTOR_PATH, dtype='float')
            ##################################################################################################################################################################
            # moran_i_ano = np.loadtxt(os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'dist_vect_moran_i_ano.csv'), dtype='float')
            # moran_i_nom = np.loadtxt(os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'dist_vect_moran_i_nom.csv'), dtype='float')
        if np.sum(distance_vector) == 0:
            cprintf(f'WARNING: All elements are zero: {distance_type}', 'l_red')
        distance_vectors.append(distance_vector)
        distance_vector_averaged = average_filter_1D(distance_vector)   
        distance_vectors_avgs.append(distance_vector_averaged)

        if not threshold_sim:    
            if analyse_distance[distance_type][0]:
                # Calculate thresholds
                if (distance_type == 'moran') or (distance_type == 'mutual-info'):
                    scores = np.loadtxt(THRESHOLD_VECTOR_PATH, dtype='float')
                    threshold = np.average(scores)
                    ano_threshold = np.average(distance_vector_averaged)
                else:
                    threshold = get_threshold(THRESHOLD_VECTOR_PATH, distance_type, analyse_distance[distance_type][1], min_log=cfg.MINIMAL_LOGGING)
                    ano_threshold = get_threshold(distance_vector_averaged, distance_type, 0.50, text_file=False, min_log=cfg.MINIMAL_LOGGING)

                if cfg.THRESHOLD_CORRECTION:
                    last_frame_idx = len(distance_vector_averaged) - 1
                    number_of_frames_to_cut = 50
                    offset_weight = 2
                    if (threshold < distance_vector_averaged[number_of_frames_to_cut:last_frame_idx-number_of_frames_to_cut].min()) or (threshold > distance_vector_averaged[number_of_frames_to_cut:last_frame_idx-number_of_frames_to_cut].max()):
                        threshold = (threshold + offset_weight*ano_threshold)/3
    
                thresholds[distance_type] = threshold
                ano_thresholds[distance_type] = ano_threshold 

    if not cfg.MINIMAL_LOGGING:
        print(f'Direct distance types: {distance_types}')

    #################################### PLOTTING SECTION #########################################

    # plot preprocessing
    # anomalous cross track errors
    cte_anomalous = data_df_anomalous['cte']
    cte_diff = np.zeros((num_anomalous_frames))
    # car speed in anomaluos mode
    speed_anomalous = data_df_anomalous['speed']
    # cte difference between anomalous and nominal conditions
    for cte_idx, cte_ano in enumerate(cte_anomalous):
        abs_diff = abs(cte_ano) - abs(closest_nom_cte_all_frames[cte_idx])
        if np.sign(cte_ano) != np.sign(abs_diff):
            abs_diff = -abs_diff
        cte_diff[cte_idx] = abs_diff
    # cte_anomalous = cte_anomalous.to_numpy()
    if cfg.PLOT_POINT_TO_POINT:
        num_of_axes = 0
        for distance_type in distance_types:
            num_of_axes += 1

        # p2p distance plot 
        num_of_axes += 1

        # cross track plot
        num_of_axes += 1
        
        height_ratios = []
        for i in range(num_of_axes):
            height_ratios.append(1)
        fig = plt.figure(figsize=(24,4*num_of_axes), constrained_layout=False)
        spec = fig.add_gridspec(nrows=num_of_axes, ncols=1, width_ratios= [1], height_ratios=height_ratios)

        # ['euclidean', 'manhattan', 'cosine', 'EMD', 'pearson', 'spearman', 'kendall', 'moran']
        distance_type_colors = {
            'euclidean' : ('deepskyblue', 'navy'),
            'manhattan' : ('indianred', 'brown'),
            'cosine' : ('mediumslateblue', 'darkslateblue'),
            'EMD' : ('mediumseagreen', 'darkgreen'),
            'pearson' :('lightslategrey', 'darkslategrey'),
            'spearman' : ('wheat','orange'),
            'kendall' : ('darkseagreen', 'darkolivegreen'),
            'moran' : ('goldenrod', 'darkgoldenrod'),
            'kl-divergence': ('salmon', 'maroon'),
            'mutual-info': ('pink', 'mediumvioletred'),
            'sobolev-norm': ('paleturquoise', 'teal')}

        cprintf(f'{distance_types}', 'green')
        for d_type_index, distance_type in enumerate(distance_types):
            fig_idx = d_type_index
            ax = fig.add_subplot(spec[fig_idx, :])
            # plot ranges
            YELLOW_BORDER = 3.6
            ORANGE_BORDER = 5.0
            RED_BORDER = 7.0
            plot_ranges(ax, cte_anomalous, cte_diff, alpha=0.2, YELLOW_BORDER=YELLOW_BORDER, ORANGE_BORDER=ORANGE_BORDER, RED_BORDER=RED_BORDER)
            plot_crash_ranges(ax, speed_anomalous)
            # plot distances
            distance_vector = distance_vectors[d_type_index]
            distance_vector_avg = distance_vectors_avgs[d_type_index]
            color = distance_type_colors[distance_type][0]
            color_avg = distance_type_colors[distance_type][1]

            if (analyse_distance[distance_type][0]) and (not threshold_sim):
                    threshold = thresholds[distance_type]
                    ano_threshold = ano_thresholds[distance_type]
                    lineplot(ax, distance_vector, distance_vector_avg, distance_type, heatmap_type, color, color_avg, eval_vars=[threshold], eval_method='threshold')
            else:
                lineplot(ax, distance_vector, distance_vector_avg, distance_type, heatmap_type, color, color_avg)

        # plot positional point distances between nominal and anomalous sims
        ax = fig.add_subplot(spec[num_of_axes-2, :])
        color = 'darkgoldenrod'
        ax.set_ylabel('p2p distances', color=color)
        ax.plot(point_distances, label='p2p distances', linewidth= 0.5, linestyle = '-', color=color)
        title = f"p2p distances"
        ax.set_title(title)
        ax.legend(loc='upper left')
        plot_ranges(ax, cte_anomalous, cte_diff, alpha=0.2, YELLOW_BORDER=YELLOW_BORDER, ORANGE_BORDER=ORANGE_BORDER, RED_BORDER=RED_BORDER)
        plot_crash_ranges(ax, speed_anomalous)

        # Plot cross track error
        ax = fig.add_subplot(spec[num_of_axes-1, :])
        color_ano = 'red'
        color_nom = 'green'
        ax.set_xlabel('Frame ID', color=color)
        ax.set_ylabel('cross-track error', color=color) 
        ax.plot(cte_anomalous, label='anomalous cross-track error', linewidth= 0.5, linestyle = '-', color=color_ano)
        ax.plot(closest_nom_cte_all_frames, label='nominal cross-track error', linewidth= 0.5, linestyle = '-', color=color_nom)
        ax.plot(cte_diff, label='cross-track error difference', linewidth= 0.5, linestyle = '--', color='brown')

        ax.axhline(y = YELLOW_BORDER, color = 'yellow', linestyle = '--')
        ax.axhline(y = -YELLOW_BORDER, color = 'yellow', linestyle = '--')
        ax.axhline(y = ORANGE_BORDER, color = 'orange', linestyle = '--')
        ax.axhline(y = -ORANGE_BORDER, color = 'orange', linestyle = '--')
        ax.axhline(y = RED_BORDER, color = 'red', linestyle = '--')
        ax.axhline(y = -RED_BORDER, color = 'red', linestyle = '--')

        plot_ranges(ax, cte_anomalous, cte_diff, alpha=0.2, YELLOW_BORDER=YELLOW_BORDER, ORANGE_BORDER=ORANGE_BORDER, RED_BORDER=RED_BORDER)
        plot_crash_ranges(ax, speed_anomalous, return_frames=False)

        title = f"cross-track error"
        ax.set_title(title)
        ax.legend(loc='upper left')

    #################################### GENERATE COMPLETED COLLAGES #########################################
    if cfg.GENERATE_SUMMARY_COLLAGES:
        missing_collages = 0
        # path to collages folder
        COLLAGES_FOLDER_PATH = os.path.join(COLLAGES_PARENT_FOLDER_PATH, f"{distance_type}_{pca_dimension}d")

        if not os.path.exists(COLLAGES_FOLDER_PATH):
            cprintf(f"Completed collages folder does not exist. Creating folder ...", 'l_blue')
            os.makedirs(COLLAGES_FOLDER_PATH)
        elif len(os.listdir(COLLAGES_FOLDER_PATH)) != num_anomalous_frames:
            cprintf(f'There are missing completed collages. Deleting and re-creating completed collages folder ...', 'yellow')
            shutil.rmtree(COLLAGES_FOLDER_PATH)
            os.makedirs(COLLAGES_FOLDER_PATH)

        create_collage = False

        for anomalous_frame in tqdm(range(num_anomalous_frames)):
            collage_name = ''
            if distance_type in cfg.SUMMARY_COLLAGE_DIST_TYPES:
                collage_name = collage_name + f'{distance_type}'
            else:
                cprintf(f' No summary collages for distance type: {distance_type} ...', 'yellow')
                cprintf(f'Removing completed collages folder ...', 'yellow')
                shutil.rmtree(COLLAGES_FOLDER_PATH)
                break

            create_collage = True
            if anomalous_frame == 0:
                cprintf(f' Generating complete summary collages for {distance_type} ...', 'l_cyan')

            # double-check if every collage actually exists
            if collage_name in os.listdir(COLLAGES_FOLDER_PATH):
                continue
            else:
                missing_collages += 1 
            # load saved base collage
            base_collage_img = Image.open(os.path.join(COLLAGES_BASE_FOLDER_PATH, f'FID_{anomalous_frame}.png'))
            # plot the current frame on the ax
            ########### TODO: ADD all relevant axes to the collage:
            ax = fig.get_axes()[num_of_axes-3]
            ax.axvline(x = anomalous_frame, color = 'black', linestyle = '-')
            if anomalous_frame < 150:
                h_text_offset = 100
            elif anomalous_frame > num_anomalous_frames - 150:
                h_text_offset = -10
            else:
                h_text_offset = 10
            ax.text(x=anomalous_frame+h_text_offset, y=distance_vectors_avgs[distance_types.index('sobolev-norm')].max()+100, s=f"{distance_vectors_avgs[distance_types.index('sobolev-norm')][anomalous_frame]:.2f}")
            distance_plot_img = save_ax_nosave(ax)
            distance_plot_img = distance_plot_img.resize((1280,160))
            ax.lines.pop(-1)
            ax.texts.pop(-1)

            ax2 = fig.get_axes()[num_of_axes-2]
            ax2.axvline(x = anomalous_frame, color = 'black', linestyle = '-')
            point_distance_img = save_ax_nosave(ax2)
            point_distance_img = point_distance_img.resize((1280,160))
            ax2.lines.pop(-1)

            ax3 = fig.get_axes()[num_of_axes-1]
            ax3.axvline(x = anomalous_frame, color = 'black', linestyle = '-')
            cte_img = save_ax_nosave(ax3)
            cte_img = cte_img.resize((1280,160))
            ax3.lines.pop(-1)

            collage = Image.new("RGBA", (1280,800))
            collage.paste(base_collage_img, (0,0))
            collage.paste(distance_plot_img, (0,320))
            collage.paste(point_distance_img, (0,480))
            collage.paste(cte_img, (0,640))
            collage.save(os.path.join(COLLAGES_FOLDER_PATH, collage_name))
    else:
        cprintf(f'Summary collages disabled for this run.', 'yellow')
            
    if cfg.GENERATE_SUMMARY_COLLAGES and create_collage:
        if missing_collages > 0 and missing_collages != num_anomalous_frames:
            cprintf('There are missing completed collages. Deleting completed collages folder ...', 'l_red')
            shutil.rmtree(COLLAGES_FOLDER_PATH)
            raise ValueError("Error in number of saved collages. Removing all the collages! Please rerun the code.")
        elif missing_collages == 0:
            cprintf(f"All completed collages already exist.", 'l_green')

        if cfg.CREATE_VIDEO:
            VIDEO_FOLDER_PATH = os.path.join(COLLAGES_PARENT_FOLDER_PATH, 'video')
            make_avi(COLLAGES_FOLDER_PATH, VIDEO_FOLDER_PATH, f"{distance_type}")

    
    # path to plotted figure images folder
    FIGURES_FOLDER_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                       f"figures_{anomalous_simulation_name}_{nominal_simulation_name}")
    if not os.path.exists(FIGURES_FOLDER_PATH):
        os.makedirs(FIGURES_FOLDER_PATH)

    fig_img_name = f"{heatmap_type}_plots_{anomalous_simulation_name}_{nominal_simulation_name}.pdf"
    fig_img_address = os.path.join(FIGURES_FOLDER_PATH, fig_img_name)
    if cfg.PLOT_POINT_TO_POINT:
        cprintf(f'\nSaving plotted figure to {FIGURES_FOLDER_PATH} ...', 'magenta')
        plt.savefig(fig_img_address, bbox_inches='tight', dpi=300)

    # plt.savefig(os.path.join(ALL_RUNS_FIGURE_PATH, f"run_id_{run_id}.png"), bbox_inches='tight', dpi=300)

    # plt.show()
    # a = ScrollableWindow(fig)
    # b = ScrollableGraph(fig, ax)

    #################################### CALCULATE RESULTS #########################################
    if cfg.CALCULATE_RESULTS and not threshold_sim:
        # get total simulation time
        main_csv_anomalous = pd.read_csv(ANOMALOUS_MAIN_CSV_PATH)
        main_csv = pd.DataFrame(main_csv_anomalous['center'].copy(), columns=['center'])

        first_img_path = main_csv['center'].iloc[0]
        last_img_path = main_csv['center'].iloc[-1]

        start_time, end_time = extract_time_from_str(first_img_path, last_img_path)
        start_time = f"{start_time[0]}:{start_time[1]}:{start_time[2]}.{start_time[3]}"
        end_time = f"{end_time[0]}:{end_time[1]}:{end_time[2]}.{end_time[3]}"


        # convert time string to datetime
        t1 = datetime.strptime(start_time, "%H:%M:%S.%f")
        t2 = datetime.strptime(end_time, "%H:%M:%S.%f")
        # get difference
        simulation_time_anomalous = t2 - t1
        number_frames_anomalous = len( anomalous['center'])
        # ano FPS
        fps_anomalous = number_frames_anomalous // simulation_time_anomalous.total_seconds()
        # Every thing that is considered crash
        red_frames, orange_frames, yellow_frames, collision_frames = colored_ranges(speed_anomalous, cte_anomalous, cte_diff,
                                                                                    alpha=0.2, YELLOW_BORDER = 3.6,ORANGE_BORDER = 5.0, RED_BORDER = 7.0)
        all_crash_frames = sorted(red_frames + orange_frames + yellow_frames + collision_frames)

        print(f"Identified %d crash(es): {len(all_crash_frames)}")
        print(all_crash_frames)
        print(f"Simulation FPS: {fps_anomalous}")
        # initializing arrays
        # threshold_too_low = {}
        # threshold_too_high = {}
        true_positive_windows = np.zeros((len(distance_types), 3))
        false_negative_windows = np.zeros((len(distance_types), 3))
        false_positive_windows = np.zeros((len(distance_types), 3))
        true_negative_windows = np.zeros((len(distance_types), 3))
        seconds_to_anticipate_list = [1, 2, 3]
        
        for seconds_to_anticipate in seconds_to_anticipate_list:
            for d_type_index, distance_type in enumerate(distance_types):
                if not cfg.MINIMAL_LOGGING:
                    cprintb(f'FP && TN: {distance_type}', 'l_yellow')
                # threshold_too_low[distance_type] = False
                # threshold_too_high[distance_type] = False
                distance_vector_avg = distance_vectors_avgs[d_type_index]
                threshold = thresholds[distance_type]
                _, no_alarm_ranges, all_ranges = get_alarm_frames(distance_vector_avg, threshold)
                # cprintf(f'{len(alarm_ranges)}: {alarm_ranges}', 'l_red')
                # cprintf(f'{len(no_alarm_ranges)}: {no_alarm_ranges}', 'l_green')
                # cprintf(f'{len(all_ranges)}: {all_ranges}', 'white')
                discarded_alarms = []
                discarded_no_alarms = []

                window_size = int(seconds_to_anticipate * fps_anomalous)
                boolean_ranges = np.zeros((len(all_ranges)), dtype=bool)
                # create a boolean array following the same pattern as all ranges.
                # Set true if no_alarm_range is bigger than window size; otherwise set false
                for rng_idx, rng in enumerate(all_ranges):
                    if isinstance(rng, list):
                        rng_length = rng[-1] - rng[0]
                    else:
                        rng_length = 1
                    if rng in no_alarm_ranges:
                        if rng_length < window_size:
                            boolean_ranges[rng_idx] = False
                        else:
                            boolean_ranges[rng_idx] = True
                # merge smaller no_alarm range into surrounding alarm ranges
                merged_ranges = merge_ranges(all_ranges, boolean_ranges)
                # if len(merged_ranges[True]) == 0:
                #     threshold_too_low[distance_type] = True
                # elif len(merged_ranges[False]) == 0:
                #     threshold_too_high[distance_type] = True
                ################ Calculate True and False Positives ################
                # if not threshold_too_high[distance_type]:
                for alarm_range in merged_ranges[False]:
                    if not isinstance(alarm_range, list):
                        discarded_alarms.append(alarm_range)
                        continue
                    alarm_range_start = alarm_range[0]
                    alarm_range_end = alarm_range[-1]
                    
                    # check if any crashes happend inside the alarm range
                    # or inside the window starting from the end of alarm range
                    # + window size (yes? TP per crash instance no? FP per alarm range)
                    alarm_rng_is_tp = False
                    if not cfg.MINIMAL_LOGGING:
                        cprintf(f'Assessing alarm range: {alarm_range}', 'l_yellow')
                    for crash_frame in all_crash_frames:
                        if (alarm_range_start <= crash_frame <= alarm_range_end) or (alarm_range_end <= crash_frame <= (alarm_range_end + window_size)):
                            alarm_rng_is_tp = True
                            true_positive_windows[d_type_index][seconds_to_anticipate-1] += 1
                            # cprintf(f'crash is predicted: {alarm_range_start} <= {crash_frame} <= {alarm_range_end} or {alarm_range_end} <= {crash_frame} <= {alarm_range_end + window_size}', 'l_green')
                    if not alarm_rng_is_tp:
                        # number_of_predictable_windows = round((alarm_range_end - alarm_range_start)/(window_size))
                        false_positive_windows[d_type_index][seconds_to_anticipate-1] += 1


                ################ Calculate True and False Negatives ################
                # if not threshold_too_low[distance_type]:
                for no_alarm_range in merged_ranges[True]:
                    if not isinstance(no_alarm_range, list):
                        discarded_no_alarms.append(no_alarm_range)
                        continue
                    no_alarm_range_start = no_alarm_range[0]
                    no_alarm_range_end = no_alarm_range[-1]
                    
                    # check if any crashes happend inside the NO alarm range
                    # or inside the window starting from the end of NO alarm range
                    # + window size (yes? FN per crash instance no? FP per no alarm range: changed that to windows inside no alarm range. Reason: very low accuracy for correct predictions(FNs) if threshold is mostly above the distance curve). 
                    no_alarm_rng_is_fn = False
                    # cprintf(f'Assessing NO alarm range: {no_alarm_range}', 'l_yellow')
                    for crash_frame in all_crash_frames:
                        if (no_alarm_range_start <= crash_frame <= no_alarm_range_end) or (no_alarm_range_end <= crash_frame <= (no_alarm_range_end + window_size)):
                            no_alarm_rng_is_fn = True
                            false_negative_windows[d_type_index][seconds_to_anticipate-1] += 1
                            # cprintf(f'crash in no_alarm_area: {no_alarm_range_start} <= {crash_frame} <= {no_alarm_range_end} or {no_alarm_range_end} <= {crash_frame} <= {no_alarm_range_end + window_size}', 'l_red')
                    if not no_alarm_rng_is_fn:
                        number_of_predictable_windows = round((no_alarm_range_end - no_alarm_range_start)/(window_size))
                        true_negative_windows[d_type_index][seconds_to_anticipate-1] += number_of_predictable_windows

        # prepare CSV file to write the results in
        results_folder_path = os.path.join(ANOMALOUS_SIM_PATH, str(run_id))
        results_csv_path = os.path.join(results_folder_path, f'results_ano_{anomalous_simulation_name}_nom_{nominal_simulation_name}.csv')
        if not os.path.exists(results_folder_path):
            os.makedirs(results_folder_path)
        if not os.path.exists(results_csv_path):
            with open(results_csv_path, mode='w',
                        newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                # writer.writerow(
                #     ["time_stamp","heatmap_type", "distance_type", "threshold", "is_threshold_too_low", "is_threshold_too_high", "crashes", "sta", "TP", "FP", "TN", "FN", "accuracy", "fpr", "precision", "recall", "f3", "max_val", "min_val"])
                writer.writerow(
                    ["time_stamp","heatmap_type", "distance_type", "threshold", "crashes", "sta", "TP", "FP", "TN", "FN", "accuracy", "fpr", "precision", "recall", "f3", "max_val", "min_val"])

        for seconds_to_anticipate in seconds_to_anticipate_list:
            for d_type_index, distance_type in enumerate(distance_types):
                if not cfg.MINIMAL_LOGGING:
                    cprintb(f'Results for distance type {distance_type} and {seconds_to_anticipate} seconds', 'l_green')
                    print('TP: ' + f'{true_positive_windows[d_type_index][seconds_to_anticipate-1]}')
                    print('FP: ' + f'{false_positive_windows[d_type_index][seconds_to_anticipate-1]}')
                    print('TN: ' + f'{true_negative_windows[d_type_index][seconds_to_anticipate-1]}')
                    print('FN: ' + f'{false_negative_windows[d_type_index][seconds_to_anticipate-1]}')

                if true_positive_windows[d_type_index][seconds_to_anticipate-1] != 0:
                    precision = true_positive_windows[d_type_index][seconds_to_anticipate-1] / (true_positive_windows[d_type_index][seconds_to_anticipate-1] + false_positive_windows[d_type_index][seconds_to_anticipate-1])
                    recall = true_positive_windows[d_type_index][seconds_to_anticipate-1] / (true_positive_windows[d_type_index][seconds_to_anticipate-1] + false_negative_windows[d_type_index][seconds_to_anticipate-1])
                    accuracy = (true_positive_windows[d_type_index][seconds_to_anticipate-1] + true_negative_windows[d_type_index][seconds_to_anticipate-1]) / (
                            true_positive_windows[d_type_index][seconds_to_anticipate-1] + true_negative_windows[d_type_index][seconds_to_anticipate-1] + false_positive_windows[d_type_index][seconds_to_anticipate-1] + false_negative_windows[d_type_index][seconds_to_anticipate-1])
                    fpr = false_positive_windows[d_type_index][seconds_to_anticipate-1] / (false_positive_windows[d_type_index][seconds_to_anticipate-1] + true_negative_windows[d_type_index][seconds_to_anticipate-1])

                    if precision != 0 or recall != 0:
                        f3 = true_positive_windows[d_type_index][seconds_to_anticipate-1] / (
                                true_positive_windows[d_type_index][seconds_to_anticipate-1] + 0.1 * false_positive_windows[d_type_index][seconds_to_anticipate-1] + 0.9 * false_negative_windows[d_type_index][seconds_to_anticipate-1])
                        try:
                            accuracy_percent = str(round(accuracy * 100))
                        except:
                            accuracy_percent = str(accuracy)

                        try:
                            fpr_percent = str(round(fpr * 100))
                        except:
                            fpr_percent = str(fpr)

                        try:
                            precision_percent = str(round(precision * 100))
                        except:
                            precision_percent = str(precision)

                        try:
                            recall_percent = str(round(recall * 100))
                        except:
                            recall_percent = str(recall)

                        try:
                            f3_percent = str(round(f3 * 100))
                        except:
                            f3_percent = str(f3)
                        if not cfg.MINIMAL_LOGGING:
                            print("Accuracy: " + accuracy_percent + "%")
                            print("False Positive Rate: " + fpr_percent + "%")
                            print("Precision: " + precision_percent + "%")
                            print("Recall: " + recall_percent + "%")
                            print("F-3: " + f3_percent + "%\n")
                    else:
                        precision = recall = f3 = accuracy = fpr = precision_percent = recall_percent = f3_percent = accuracy_percent = fpr_percent = 0
                        if not cfg.MINIMAL_LOGGING:
                            print("Accuracy: undefined")
                            print("False Positive Rate: undefined")
                            print("Precision: undefined")
                            print("Recall: undefined")
                            print("F-3: undefined\n")
                else:
                    precision = recall = f3 = accuracy = fpr = precision_percent = recall_percent = f3_percent = accuracy_percent = fpr_percent = 0
                    if not cfg.MINIMAL_LOGGING:
                        print("Accuracy: undefined")
                        print("False Positive Rate: undefined")
                        print("Precision: undefined")
                        print("Recall: undefined")
                        print("F-1: undefined")
                        print("F-3: undefined\n")

                with open(results_csv_path, mode='a',
                            newline='') as result_file:
                    writer = csv.writer(result_file,
                                        delimiter=',',
                                        quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL,
                                        lineterminator='\n')
                    writer.writerow([datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                                    heatmap_type,
                                    distance_type,
                                    thresholds[distance_type],
                                    # threshold_too_low[distance_type],
                                    # threshold_too_high[distance_type],
                                    str(len(all_crash_frames)),
                                    str(seconds_to_anticipate),
                                    str(true_positive_windows[d_type_index][seconds_to_anticipate-1]),
                                    str(false_positive_windows[d_type_index][seconds_to_anticipate-1]),
                                    str(true_negative_windows[d_type_index][seconds_to_anticipate-1]),
                                    str(false_negative_windows[d_type_index][seconds_to_anticipate-1]),
                                    accuracy_percent,
                                    fpr_percent,
                                    precision_percent,
                                    recall_percent,
                                    f3_percent,
                                    distance_vectors_avgs[d_type_index].max(),
                                    distance_vectors_avgs[d_type_index].min()])
    if not threshold_sim:
        return fig_img_address, results_csv_path, seconds_to_anticipate_list



