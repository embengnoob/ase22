import utils
from utils import *
from utils_models import *

######################################################################################
################### EVALUATION FUNCTION FOR THE THIRDEYE METHOD ######################
######################################################################################


def evaluate_failure_prediction_thirdeye(cfg, NOMINAL_PATHS, ANOMALOUS_PATHS, seconds_to_anticipate, anomalous_simulation_name,
                                        nominal_simulation_name, heatmap_type, summary_types, aggregation_methods):
    
    NOMINAL_SIM_PATH = NOMINAL_PATHS[0]
    NOMINAL_MAIN_CSV_PATH = NOMINAL_PATHS[1]
    NOMINAL_HEATMAP_PARENT_FOLDER_PATH = NOMINAL_PATHS[2]
    NOMINAL_HEATMAP_FOLDER_PATH = NOMINAL_PATHS[3]
    NOMINAL_HEATMAP_CSV_PATH = NOMINAL_PATHS[4]
    NOMINAL_HEATMAP_IMG_PATH = NOMINAL_PATHS[5]
    NOMINAL_HEATMAP_IMG_GRADIENT_PATH = NOMINAL_PATHS[6]
    NOM_NPY_SCORES_FOLDER_PATH = NOMINAL_PATHS[9]

    ANOMALOUS_SIM_PATH = ANOMALOUS_PATHS[0]
    ANOMALOUS_MAIN_CSV_PATH = ANOMALOUS_PATHS[1]
    ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH = ANOMALOUS_PATHS[2]
    ANOMALOUS_HEATMAP_FOLDER_PATH = ANOMALOUS_PATHS[3]
    ANOMALOUS_HEATMAP_CSV_PATH = ANOMALOUS_PATHS[4]
    ANOMALOUS_HEATMAP_IMG_PATH = ANOMALOUS_PATHS[5]
    ANOMALOUS_HEATMAP_IMG_GRADIENT_PATH = ANOMALOUS_PATHS[6]

    RUN_RESULTS_PATH = ANOMALOUS_PATHS[7]
    RUN_FIGS_PATH = ANOMALOUS_PATHS[8]
    ANO_NPY_SCORES_FOLDER_PATH = ANOMALOUS_PATHS[9]

    nom_csv_npy_paths = [NOMINAL_HEATMAP_CSV_PATH, NOM_NPY_SCORES_FOLDER_PATH]
    ano_csv_npy_paths = [ANOMALOUS_HEATMAP_CSV_PATH, ANO_NPY_SCORES_FOLDER_PATH, ANOMALOUS_MAIN_CSV_PATH]

#################################### CALCULATE WINDOW SCORES #########################################
    window_scores_nom, thresholds, loss_scores_nom, data_df_nominal = get_nom_window_scores(heatmap_type, summary_types, aggregation_methods, nom_csv_npy_paths)
    window_scores_ano, window_indices_ano, loss_scores_ano, data_df_anomalous, simulation_time_anomalous, fps_anomalous = get_ano_window_scores(heatmap_type, summary_types, aggregation_methods, ano_csv_npy_paths)
##################################### CALCULATE RESULTS #########################################
    if cfg.CALCULATE_RESULTS:
        anomalous = pd.DataFrame(data_df_anomalous['frameId'].copy(), columns=['frameId'])
        num_anomalous_frames = len( anomalous['frameId'])

        # anomalous cross track errors
        anomalous['cte'] = data_df_anomalous['cte'].copy()
        # car speed in anomalous mode
        anomalous['speed'] = data_df_anomalous['speed'].copy()
        # Every thing that is considered crash
        red_frames, orange_frames, yellow_frames, collision_frames = colored_ranges(anomalous['speed'], anomalous['cte'], cte_diff=None,
                                                                                    alpha=0.2, YELLOW_BORDER=3.6,ORANGE_BORDER=5.0, RED_BORDER=7.0)
        all_crash_frames = sorted(red_frames + orange_frames + yellow_frames + collision_frames)

        print(f"Identified %d crash(es): {len(all_crash_frames)}")
        print(all_crash_frames)
        print(f"Simulation FPS: {fps_anomalous}")

        true_positive_windows = np.zeros((4, 3))
        false_negative_windows = np.zeros((4, 3))
        false_positive_windows = np.zeros((4, 3))
        true_negative_windows = np.zeros((4, 3))
        
        for sta in seconds_to_anticipate:
            for product_idx, (summary_type, aggregation_method) in enumerate(product(summary_types, aggregation_methods)):
                if product_idx == 1:
                    raise ValueError('EOC')
                cprintf(f'\n########### FP && TN: sta:{sta}, agg_method:{aggregation_method}, summary type:{summary_type} ###########', 'blue')
                threshold = thresholds[summary_type][aggregation_method]
                window_scores = window_scores_ano[summary_type][aggregation_method]
                window_indices = window_indices_ano[summary_type][aggregation_method]
                if len(window_scores) != len(window_indices):
                            raise ValueError(Fore.RED + f"Mismatch in number of windows and window indices in window scores of {summary_type} and {aggregation_method}" + Fore.RESET)
                _, no_alarm_ranges, all_ranges = get_alarm_frames_thirdeye(window_scores, window_indices, threshold, num_anomalous_frames)

                discarded_alarms = []
                discarded_no_alarms = []

                window_size = int(sta * fps_anomalous)
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
                cprintf(f"no_alarm_ranges", 'l_green')
                highlight(no_alarm_ranges, 'l_green')
                cprintf(f"all_ranges", 'l_yellow')
                highlight(all_ranges, 'l_yellow')
                cprintf(f"merged_ranges", 'l_red')
                highlight(merged_ranges, 'l_red')
                ######################################### Calculate True and False Positives #########################################
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
                            true_positive_windows[product_idx][sta-1] += 1
                            # cprintf(f'crash is predicted: {alarm_range_start} <= {crash_frame} <= {alarm_range_end} or {alarm_range_end} <= {crash_frame} <= {alarm_range_end + window_size}', 'l_green')
                    if not alarm_rng_is_tp:
                        # number_of_predictable_windows = round((alarm_range_end - alarm_range_start)/(window_size))
                        false_positive_windows[product_idx][sta-1] += 1


                ######################################### Calculate True and False Negatives #########################################
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
                            false_negative_windows[product_idx][sta-1] += 1
                            # cprintf(f'crash in no_alarm_area: {no_alarm_range_start} <= {crash_frame} <= {no_alarm_range_end} or {no_alarm_range_end} <= {crash_frame} <= {no_alarm_range_end + window_size}', 'l_red')
                    if not no_alarm_rng_is_fn:
                        number_of_predictable_windows = round((no_alarm_range_end - no_alarm_range_start)/(window_size))
                        true_negative_windows[product_idx][sta-1] += number_of_predictable_windows

        # prepare CSV file to write the results in
        results_csv_path = os.path.join(RUN_RESULTS_PATH, f'thirdeye_total_results_ano_{anomalous_simulation_name}_nom_{nominal_simulation_name}.csv')
        if not os.path.exists(RUN_RESULTS_PATH):
            os.makedirs(RUN_RESULTS_PATH)
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
                    ["time_stamp","heatmap_type", "summary_type", "aggregation_method", "threshold", "crashes", "sta", "TP", "FP", "TN", "FN", "accuracy", "fpr", "precision", "recall", "f3", "max_val", "min_val"])

        for sta in seconds_to_anticipate:
            for product_idx, (summary_type, aggregation_method) in enumerate(product(summary_types, aggregation_methods)):
                if not cfg.MINIMAL_LOGGING:
                    cprintb(f'Results for summary type {summary_type}, aggregation method {aggregation_method}, and {sta} seconds', 'l_green')
                    print('TP: ' + f'{true_positive_windows[product_idx][sta-1]}')
                    print('FP: ' + f'{false_positive_windows[product_idx][sta-1]}')
                    print('TN: ' + f'{true_negative_windows[product_idx][sta-1]}')
                    print('FN: ' + f'{false_negative_windows[product_idx][sta-1]}')

                if true_positive_windows[product_idx][sta-1] != 0:
                    precision = true_positive_windows[product_idx][sta-1] / (true_positive_windows[product_idx][sta-1] + false_positive_windows[product_idx][sta-1])
                    recall = true_positive_windows[product_idx][sta-1] / (true_positive_windows[product_idx][sta-1] + false_negative_windows[product_idx][sta-1])
                    accuracy = (true_positive_windows[product_idx][sta-1] + true_negative_windows[product_idx][sta-1]) / (
                            true_positive_windows[product_idx][sta-1] + true_negative_windows[product_idx][sta-1] + false_positive_windows[product_idx][sta-1] + false_negative_windows[product_idx][sta-1])
                    fpr = false_positive_windows[product_idx][sta-1] / (false_positive_windows[product_idx][sta-1] + true_negative_windows[product_idx][sta-1])

                    if precision != 0 or recall != 0:
                        f3 = true_positive_windows[product_idx][sta-1] / (
                                true_positive_windows[product_idx][sta-1] + 0.1 * false_positive_windows[product_idx][sta-1] + 0.9 * false_negative_windows[product_idx][sta-1])
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
#["time_stamp","heatmap_type", "summary_type", "aggregation_method", "threshold", "crashes", "sta", "TP", "FP", "TN", "FN", "accuracy", "fpr", "precision", "recall", "f3", "max_val", "min_val"])
                    writer.writerow([datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                                    heatmap_type,
                                    summary_type,
                                    aggregation_method,
                                    thresholds[summary_type][aggregation_method],
                                    # threshold_too_low[distance_type],
                                    # threshold_too_high[distance_type],
                                    str(len(all_crash_frames)),
                                    str(sta),
                                    str(true_positive_windows[product_idx][sta-1]),
                                    str(false_positive_windows[product_idx][sta-1]),
                                    str(true_negative_windows[product_idx][sta-1]),
                                    str(false_negative_windows[product_idx][sta-1]),
                                    accuracy_percent,
                                    fpr_percent,
                                    precision_percent,
                                    recall_percent,
                                    f3_percent,
                                    max(window_scores_ano[summary_type][aggregation_method]),
                                    min(window_scores_ano[summary_type][aggregation_method])
                                    ])

#################################### PLOTTING SECTION #########################################
    if cfg.PLOT_THIRDEYE:

        num_of_axes = len(summary_types)*len(aggregation_methods)
        
        # cross track plot
        num_of_axes += 1
        
        height_ratios = []
        for i in range(num_of_axes):
            height_ratios.append(1)
        fig = plt.figure(figsize=(24,4*num_of_axes), constrained_layout=False)
        spec = fig.add_gridspec(nrows=num_of_axes, ncols=1, width_ratios= [1], height_ratios=height_ratios)

        fig_idx = -1
        for summary_type in summary_types:
            for aggregation_method in aggregation_methods:
                fig_idx += 1
                ax = fig.add_subplot(spec[fig_idx, :])

                # plot ranges
                plot_ranges(ax, data_df_anomalous['cte'], cte_diff=None)
                plot_crash_ranges(ax, data_df_anomalous['speed'])
                color = summary_type_colors[summary_type][0]

                # plot thirdeye scores
                ax.plot(pd.Series(data_df_anomalous['frameId']), loss_scores_ano[summary_type], label='ano_losses', linewidth= 0.5, linestyle = 'dotted', color=color)

                # plot threshold
                threshold = thresholds[summary_type][aggregation_method]
                ax.axhline(y = threshold, color = 'r', linestyle = '--')

                window_scores = window_scores_ano[summary_type][aggregation_method]
                window_indices = window_indices_ano[summary_type][aggregation_method]
                for idx, aggregated_score in enumerate(window_scores):
                    if aggregated_score >= threshold:
                        color = 'red'
                    else:
                        color = 'cyan'
                    ax.hlines(y=aggregated_score, xmin=window_indices[idx] - fps_anomalous, xmax=window_indices[idx], color=color, linewidth=3)
                    
                    spine_color = 'black'
                    ax_spines = ['top', 'right', 'left', 'bottom']
                    for spine in ax_spines:
                        ax.spines[spine].set_color(spine_color)

                    bolded_part = f": {summary_type}: {aggregation_method}"
                    title = heatmap_type + r"$\bf{" + bolded_part + "}$"
                    ax.set_title(title, color=spine_color)
                    ax.legend(loc='upper left')

                    # set tick and ticklabel color
                    ax.tick_params(axis='x', colors=spine_color)    #setting up X-axis tick color to red
                    ax.tick_params(axis='y', colors=spine_color)  #setting up Y-axis tick color to black
                    ax.set_xticks(np.arange(0, len(loss_scores_ano[summary_type]), 50.0), minor=False)

        # Plot cross track error
        ax = fig.add_subplot(spec[num_of_axes-1, :])
        color_ano = 'red'
        color_nom = 'green'
        ax.set_xlabel('Frame ID', color=color)
        ax.set_ylabel('cross-track error', color=color) 
        ax.plot(data_df_anomalous['cte'], label='anomalous cross-track error', linewidth= 0.5, linestyle = '-', color=color_ano)

        YELLOW_BORDER = 3.6
        ORANGE_BORDER = 5.0
        RED_BORDER = 7.0

        ax.axhline(y = YELLOW_BORDER, color = 'yellow', linestyle = '--')
        ax.axhline(y = -YELLOW_BORDER, color = 'yellow', linestyle = '--')
        ax.axhline(y = ORANGE_BORDER, color = 'orange', linestyle = '--')
        ax.axhline(y = -ORANGE_BORDER, color = 'orange', linestyle = '--')
        ax.axhline(y = RED_BORDER, color = 'red', linestyle = '--')
        ax.axhline(y = -RED_BORDER, color = 'red', linestyle = '--')

        plot_ranges(ax, data_df_anomalous['cte'], cte_diff=None)
        plot_crash_ranges(ax, data_df_anomalous['speed'], return_frames=False)

        title = f"cross-track error"
        ax.set_title(title)
        ax.legend(loc='upper left')


        # path to plotted figure images folder
        FIGURES_FOLDER_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                        f"figures_{anomalous_simulation_name}_{nominal_simulation_name}", 'thirdeye')
        if not os.path.exists(FIGURES_FOLDER_PATH):
            os.makedirs(FIGURES_FOLDER_PATH)

        fig_img_name = f"thirdeye_{heatmap_type}_plots_{anomalous_simulation_name}_{nominal_simulation_name}.pdf"
        fig_img_address = os.path.join(FIGURES_FOLDER_PATH, fig_img_name)
        cprintf(f'\nSaving plotted figure to {FIGURES_FOLDER_PATH} ...', 'magenta')
        plt.savefig(fig_img_address, bbox_inches='tight', dpi=300)

    return fig_img_address, results_csv_path, RUN_RESULTS_PATH