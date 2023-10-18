import csv
import gc
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from scipy.stats import gamma


def get_threshold(losses, conf_level=0.95):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t

def get_crash_frames(data_df_anomalous, number_frames_anomalous):
    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[idx + 1] == 1: # if next frame is a crash
            first_crash_index = idx + 1
            all_first_frame_position_crashed_sequences.append(first_crash_index) # makes a list of all frames where crash first happened
            # print("first_crash_index: %d" % first_crash_index) 
    return all_first_frame_position_crashed_sequences

def evaluate_failure_prediction(cfg, heatmap_type, anomalous_simulation_name, nominal_simulation_name, summary_type,
                                aggregation_method, condition, fig,
                                axs,
                                subplot_counter, number_of_crashes, run_counter):
    print("Using summarization average" if summary_type is '-avg' else "Using summarization gradient")
    print("Using aggregation mean" if aggregation_method is 'mean' else "Using aggregation max")

    # 0. 
    # 1. load heatmap scores in nominal conditions
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        nominal_simulation_name,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
    original_losses = np.load(path)
    
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        nominal_simulation_name,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')

    logging.warning(f"Path for data_df_nominal: {path}")
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

    logging.warning(f"Path for data_df_anomalous: {path}")
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
                                                                                                cfg.PLOT_NOMINAL)
    
    # 4. compute TP and FN using different time to misbehaviour windows
    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows, subplot_counter = compute_tp_and_fn(data_df_anomalous,
                                                                                                                anomalous_losses,
                                                                                                                threshold,
                                                                                                                seconds,
                                                                                                                fig,
                                                                                                                axs,
                                                                                                                subplot_counter,
                                                                                                                number_of_crashes,
                                                                                                                run_counter,
                                                                                                                summary_type,
                                                                                                                cfg.PLOT_ANOMALOUS_ALL_WINDOWS,
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
                      axs, subplot_counter, number_of_crashes, run_counter, summary_type, PLOT_ANOMALOUS_ALL_WINDOWS=True,
                      aggregation_method='mean', cond='ood'):
    
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

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = get_crash_frames(data_df_anomalous, number_frames_anomalous)
    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence / length of time window (seconds to anticipate) in terms of number of frames

    # first frame n seconds before the failure / length of time window one second shorter
    # than seconds_to_anticipate in terms of number of frame
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)
    # frames_to_reassign_2 = 1  # first frame before the failure

    reaction_window = pd.Series()

    crash_counter = 0
    (reaction_window_x_min, reaction_window_x_max) = (0,0)

###################################################################################################################################
    if PLOT_ANOMALOUS_ALL_WINDOWS:
        # anomalous losses
        sma_anomalous = pd.Series(losses_on_anomalous)
        # calculate simulation time window and cut extra samples
        num_windows_anomalous = len(data_df_anomalous) // fps_anomalous
        if len(data_df_anomalous) % fps_anomalous != 0:
            num_to_delete = len(data_df_anomalous) - (num_windows_anomalous * fps_anomalous) - 1
            data_df_anomalous_all_win = data_df_anomalous[:-num_to_delete]
            sma_anomalous_all_win = sma_anomalous[:-num_to_delete]

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

    for item in all_first_frame_position_crashed_sequences:
        print("\n(((((((((((((((((((((((analysing failure %d))))))))))))))))))))))))))" % item)
        # for the first frames smaller than a window
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            continue

        # the detection window overlaps with a previous crash; skip it
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            # if crash_counter == 1:
            (reaction_window_x_min, reaction_window_x_max) = (crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2].index[0], crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2].index[-1])
            reaction_window = reaction_window._append(
                crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

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
            axs[run_counter-1].hlines(y= aggregated_score, xmin=reaction_window_x_min, xmax=reaction_window_x_max, color='lime', linewidth=3)

            subplot_counter += 1
            print(f'********************************{subplot_counter}********************************')
            if subplot_counter % 9 == 0:
                #print(subplot_counter)
                ax = axs[run_counter-1]
                for crash_frame in all_first_frame_position_crashed_sequences:
                    ax.axvline(x = crash_frame, color = 'teal', linestyle = '--')
                ax.axhline(y = threshold, color = 'r', linestyle = '--')
                ax.plot(pd.Series(data_df_anomalous['frameId']), sma_anomalous, label='sma_anomalous', linewidth= 0.5, linestyle = 'dotted')
                # ax.hlines(y= aggregated_score, xmin=reaction_window_x_min, xmax=reaction_window_x_max, color='r')
                title = f"{aggregation_method} && {summary_type}"
                ax.set_title(title)
                ax.set_ylabel("heatmap scores")
                ax.set_xlabel("frame number")
                ax.legend(loc='upper left')

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows, subplot_counter


def compute_fp_and_tn(data_df_nominal, aggregation_method, condition,fig,axs,subplot_counter,run_counter, PLOT_NOMINAL):
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

    print(f"fps_nominal ==================================>==================================>==================================> {fps_nominal}")
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
        for idx, aggregated_score in enumerate(list_aggregated):
            axs[run_counter-1].hlines(y=aggregated_score, xmin=list_aggregated_indexes[idx] - fps_nominal, xmax=list_aggregated_indexes[idx], color='cyan', linewidth=3)
        ax = axs[run_counter-1]
        ax.plot(pd.Series(data_df_nominal['frameId']), sma_nominal, label='sma_nominal', linewidth= 0.5, linestyle = 'dotted', color='g')
        ax.plot(list_aggregated_indexes, list_aggregated, label='agg', linewidth= 0.5, linestyle = 'dotted', color='cyan')
        # ax.hlines(y= aggregated_score, xmin=reaction_window_x_min, xmax=reaction_window_x_max, color='r')
        ax.legend(loc='upper left')
#############################################################################################################################################################
    assert false_positive_windows + true_negative_windows == num_windows_nominal
    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    return false_positive_windows, true_negative_windows, threshold, subplot_counter
