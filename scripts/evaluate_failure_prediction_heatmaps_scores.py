import csv
import gc
import os
import sys
import math
import numpy as np
import pandas as pd
import glob
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from keras import backend as K
from scipy.stats import gamma
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances, pairwise
from sklearn.decomposition import PCA
from tqdm import tqdm
import utils
from utils import *
from utils_models import *
import re

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

def evaluate_p2p_failure_prediction(cfg, heatmap_type, heatmap_types, anomalous_simulation_name, nominal_simulation_name,
                                    distance_method, distance_methods, pca_dimension, pca_dimensions, abstraction_method,
                                    abstraction_methods, fig, axs):
    
    print(f"Using distance method {distance_method}")
    print(f"Calculating with {pca_dimension} dimension(s).")

    # 1. find the closest frame from anomalous sim to the frame in nominal sim (comparing car position)
    nom_path = os.path.join(cfg.TESTING_DATA_DIR,
                            nominal_simulation_name,
                            'heatmaps-' + heatmap_type,
                            'driving_log.csv')
    print(f"Path for data_df_nominal: {nom_path}")
    data_df_nominal = pd.read_csv(nom_path)

    ano_path = os.path.join(cfg.TESTING_DATA_DIR,
                            anomalous_simulation_name,
                            'heatmaps-' + heatmap_type,
                            'driving_log.csv')
    print(f"Path for data_df_anomalous: {ano_path}")
    data_df_anomalous = pd.read_csv(ano_path)

    # car position in nominal simulation
    nominal = pd.DataFrame(data_df_nominal['frameId'].copy(), columns=['frameId'])
    nominal['position'] = data_df_nominal['position'].copy()
    nominal['center'] = data_df_nominal['center'].copy()
    # total number of nominal frames
    num_nominal_frames = pd.Series.max(nominal['frameId'])

    # car position in anomalous mode
    anomalous = pd.DataFrame(data_df_anomalous['frameId'].copy(), columns=['frameId'])
    anomalous['position'] = data_df_anomalous['position'].copy()
    anomalous['center'] = data_df_anomalous['center'].copy()
    # total number of anomalous frames
    num_anomalous_frames = pd.Series.max(anomalous['frameId'])

    # path to csv file containing mapped positions
    pos_map_path = os.path.join(cfg.TESTING_DATA_DIR,
                                    anomalous_simulation_name,
                                    "heatmaps-" + heatmap_type,
                                    f'pos_mappings_{nominal_simulation_name}.csv')
    # path to csv file containing distance_vectors
    dist_vector_path = os.path.join(cfg.TESTING_DATA_DIR,
                                    anomalous_simulation_name,
                                    "heatmaps-" + heatmap_type,
                                    'distances',
                                    f'dist_vector_{distance_method}_{pca_dimension}_{abstraction_method}.csv')
      
    # check if positional mapping list file in csv format already exists 
    if not os.path.exists(pos_map_path):
        cprintf(f"Positional mapping list does not exist. Generating list ...", 'l_blue')
        pos_mappings = np.zeros(num_anomalous_frames, dtype=float)
        nominal_positions = np.zeros((num_nominal_frames, 3), dtype=float)
        # cluster of all nominal positions
        for nominal_frame in range(num_nominal_frames):
            vector = string_to_np_array(nominal.iloc[nominal_frame, 1])
            nominal_positions[nominal_frame] = vector
        # compare each anomalous position with the nominal cluster and find the closest nom position (mapping)
        cprintf(f"Number of frames in anomalous conditions: {num_anomalous_frames}", 'l_magenta')
        for anomalous_frame in tqdm(range(num_anomalous_frames)):
            vector = string_to_np_array(anomalous.iloc[anomalous_frame, 1])
            sample_point = vector.reshape(1, -1)
            closest, _ = pairwise_distances_argmin_min(sample_point, nominal_positions)
            pos_mappings[anomalous_frame] = closest
        # save list of positional mappings
        print(pos_mappings)
        print("Saving CSV file ...")
        np.savetxt(pos_map_path, pos_mappings, delimiter=",")
    else:
        cprintf(f"Positional mapping list exists.", 'l_green')       
        # load list of mapped positions
        print(f"Loading CSV file {pos_map_path} ...")
        pos_mappings = np.loadtxt(pos_map_path, dtype='int')
        print(pos_mappings)

    # 2. Principal component analysis to project heat-maps to a point in a lower dim space
    pca = PCA(n_components=pca_dimension)

    distance_vector_abs = []

    distance_method_colors = {
        'pairwise_distance' : 'indianred',
        'cosine_similarity' : 'orange',
        'polynomial_kernel' : 'gold',
        'sigmoid_kernel' : 'lawngreen',
        'rbf_kernel' :'cyan',
        'laplacian_kernel' : 'blueviolet',
        'chi2_kernel' : 'deepskyblue'}
    # check if the chosen distance method and pca dimension vectors already exist as csv file
    if not os.path.exists(dist_vector_path):
        # create directory for the heatmaps
        dist_vectors_folder_path = os.path.join(cfg.TESTING_DATA_DIR,
                                        anomalous_simulation_name,
                                        "heatmaps-" + heatmap_type,
                                        'distances')
        if not os.path.exists(dist_vectors_folder_path):
            os.makedirs(dist_vectors_folder_path)
        # initialize arrays
        _, _, ht_height, ht_width = get_heatmaps(0, anomalous, nominal, pos_mappings, return_size=True)
        x_ano_all_frames = np.zeros((num_anomalous_frames, ht_height*ht_width))
        x_nom_all_frames = np.zeros((num_anomalous_frames, ht_height*ht_width))

        cprintf(f"Distance vector of distance method {distance_method} and of PCA dim {pca_dimension} does not exist. Generating array ...", 'l_blue') 
        for anomalous_frame in tqdm(range(num_anomalous_frames)):
            # load corresponding heatmaps
            ano_img, closest_nom_img = get_heatmaps(anomalous_frame, anomalous, nominal, pos_mappings, return_size=False)
            # convert to grayscale
            x_ano = cv2.cvtColor(ano_img, cv2.COLOR_BGR2GRAY).reshape(1, -1)
            x_nom = cv2.cvtColor(closest_nom_img, cv2.COLOR_BGR2GRAY).reshape(1, -1)
            # print(f'x_ano shape is {x_ano.shape}')
            # print(f'x_nom shape is {x_nom.shape}')
            # standardization and normalization
            ano_std_scale = preprocessing.StandardScaler().fit(x_ano)
            x_ano_std = ano_std_scale.transform(x_ano)
            nom_std_scale = preprocessing.StandardScaler().fit(x_nom)
            x_nom_std = nom_std_scale.transform(x_nom)

            x_ano_all_frames[anomalous_frame] = x_ano_std
            x_nom_all_frames[anomalous_frame] = x_nom_std

        # PCA conversion (row to point)
        print(f'x_ano_all_frames shape is {x_ano_all_frames.shape}')
        print(f'x_nom_all_frames shape is {x_nom_all_frames.shape}')
        pca_ano = pca.fit_transform(x_ano_all_frames)
        pca_nom = pca.fit_transform(x_nom_all_frames)
        print(f'pca_ano shape is {pca_ano.shape}')
        print(f'pca_nom shape is {pca_nom.shape}')

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        if pca_dimension == 3:
            ax.scatter(pca_ano[:,0], pca_ano[:,1], pca_ano[:,2], color = 'r')
            ax.scatter(pca_nom[:,0], pca_nom[:,1], pca_nom[:,2], color = 'b')
        elif pca_dimension == 2:
            ax.scatter(pca_ano[:,0], pca_ano[:,1], color = 'r')
            ax.scatter(pca_nom[:,0], pca_nom[:,1], color = 'b')            
        plt.show()
    #     # 3. Using different pairwise distance methods to calculate distance between two anomalous and nominal pca clusters
    #     if distance_method == 'pairwise_distance':
    #         distance_vector = pairwise.paired_distances(pca_ano, pca_nom)
    #     elif distance_method == 'cosine_similarity':
    #         distance_vector = pairwise.cosine_similarity(pca_ano, pca_nom)
    #     elif distance_method == 'polynomial_kernel':
    #         distance_vector = pairwise.polynomial_kernel(pca_ano, pca_nom)
    #     elif distance_method == 'sigmoid_kernel':
    #         distance_vector = pairwise.sigmoid_kernel(pca_ano, pca_nom)
    #     elif distance_method == 'rbf_kernel':
    #         distance_vector = pairwise.rbf_kernel(pca_ano, pca_nom)  
    #     elif distance_method == 'laplacian_kernel':
    #         distance_vector = pairwise.laplacian_kernel(pca_ano, pca_nom)
    #     elif distance_method == 'chi2_kernel':
    #         distance_vector = pairwise.chi2_kernel(pca_ano, pca_nom)
    #     # # compute a representative point of the distance vector of this frame based on the abstraction method
    #     # if abstraction_method == 'avg':
    #     #     distance_vector_abs.append(np.average(distance_vector))
    #     # elif abstraction_method == 'variance':
    #     #     distance_vector_abs.append(np.var(distance_vector))

    #     print(Fore.WHITE + f"Saving CSV file ..." + Fore.RESET)
    #     np.savetxt(dist_vector_path, distance_vector_abs, delimiter=",")
    # else:
    #     print(Fore.GREEN + f"Distance vector exists." + Fore.RESET)
    #     print(Fore.WHITE + f"Loading CSV file ..." + Fore.RESET)
    #     distance_vector_abs = np.loadtxt(dist_vector_path, dtype='float')

    # # calculate threshold via gamma fitting
    # threshold = get_threshold(distance_vector_abs)
    # # get the correct ax index
    # hm_index = heatmap_types.index(heatmap_type)
    # pca_index = pca_dimensions.index(pca_dimension)
    # abs_index = abstraction_methods.index(abstraction_method)
    # correct_index = (hm_index/(len(heatmap_types)))*len(abstraction_methods)*len(pca_dimensions)*len(heatmap_types) + \
    #                 (pca_index/(len(pca_dimensions)))*len(abstraction_methods)*len(pca_dimensions) + abs_index
    # ax = axs[int(correct_index)]
    # # plot
    # # anomalous cross track errors
    # cte_anomalous = data_df_anomalous['cte']
    # # car speed in anomaluos mode
    # speed_anomalous = data_df_anomalous['speed']

    # # plot cross track error values: 
    # # cte > 4: reaching the borders of the track: yellow
    # # 5> cte > 7: on the borders of the track (partial crossing): orange
    # # cte > 7: out of track (full crossing): red
    # yellow_condition = (abs(cte_anomalous)>3.6)&(abs(cte_anomalous)<5.0)
    # orange_condition = (abs(cte_anomalous)>5.0)&(abs(cte_anomalous)<7.0)
    # red_condition = (abs(cte_anomalous)>7.0)
    # yellow_ranges = get_ranges(yellow_condition)
    # orange_ranges = get_ranges(orange_condition)
    # red_ranges = get_ranges(red_condition)

    # # plot yellow ranges
    # plot_ranges(yellow_ranges, ax, color='yellow', alpha=0.2)
    # # plot orange ranges
    # plot_ranges(orange_ranges, ax, color='orange', alpha=0.2)
    # # plot red ranges
    # plot_ranges(red_ranges, ax, color='red', alpha=0.2)

    # # plot crash instances: speed < 1.0 
    # crash_condition = (abs(speed_anomalous)<1.0)
    # # remove the first 10 frames: starting out so speed is less than 1 
    # crash_condition[:10] = False
    # crash_ranges = get_ranges(crash_condition)
    # # plot_ranges(crash_ranges, ax, color='blue', alpha=0.2)
    # NUM_OF_FRAMES_TO_CHECK = 20
    # is_crash_instance = False
    # for rng in crash_ranges:
    #     # check 20 frames before first frame with speed < 1.0. if not bigger than 15 it's not
    #     # a crash instance it's reset instance
    #     if isinstance(rng, list):
    #         crash_frame = rng[0]
    #     else:
    #         crash_frame = rng
    #     for speed in speed_anomalous[crash_frame-NUM_OF_FRAMES_TO_CHECK:crash_frame]:
    #         if speed > 15.0:
    #             is_crash_instance = True
    #     if is_crash_instance == True:
    #         is_crash_instance = False
    #         reset_frame = crash_frame
    #         ax.axvline(x = reset_frame, color = 'blue', linestyle = '--')
    #         continue
    #     # plot crash ranges (speed < 1.0)
    #     if isinstance(rng, list):
    #         ax.axvspan(rng[0], rng[1], color='teal', alpha=0.2)
    #     else:
    #         ax.axvspan(rng, rng+1, color='teal', alpha=0.2)
    # ax.plot(distance_vector_abs, label=distance_method, linewidth= 0.5, linestyle = '-', color=distance_method_colors[distance_method])
    # title = f"{heatmap_type} && {pca_dimension}d && {abstraction_method}"
    # ax.set_title(title)
    # ax.set_ylabel("distance scores")
    # # ax.set_xlabel("frame number")
    # ax.legend(loc='upper left')
    
    

######################################################################################
############################### AUXILIARY FUNCTIONS ##################################
######################################################################################

def string_to_np_array(vector_string, frame_num):
    if '[' in vector_string:
        # autonomous mode
        vector_string = ' '.join(vector_string.split())
        vector_string = vector_string.strip("[]").strip().replace(' ', ',')
        vector = np.fromstring(vector_string, dtype=float, sep=',')
    elif '(' in vector_string:
        # manual training mode
        vector_string = vector_string.strip("()").replace('  ', ' ')
        vector = np.fromstring(vector_string, dtype=float, sep=' ')
    if vector.shape != (3,):
        cprintf(str(vector.shape), 'l_red')
        print(vector_string)
        print(vector)
        raise ValueError(f"Car position format of frame number {frame_num} can't be interpreted.")
    return vector

def correct_windows_path(address):
    if "\\\\" in address:
        address = address.replace("\\\\", "/")
    elif "\\\\" in address:
        address = address.replace("\\\\", "/")
    return address

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

def get_OOT_frames(data_df_anomalous, number_frames_anomalous):
    OOT_anomalous = data_df_anomalous['crashed']
    OOT_anomalous.is_copy = None
    OOT_anomalous_in_anomalous_conditions = OOT_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_OOT_sequences = []
    for idx, item in enumerate(OOT_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if OOT_anomalous_in_anomalous_conditions[idx] == 0 and OOT_anomalous_in_anomalous_conditions[idx + 1] == 1: # if next frame is an OOT
            first_OOT_index = idx + 1
            all_first_frame_position_OOT_sequences.append(first_OOT_index) # makes a list of all frames where OOT first happened
            # print("first_OOT_index: %d" % first_OOT_index) 
    return all_first_frame_position_OOT_sequences, OOT_anomalous_in_anomalous_conditions

def get_ranges(boolean_cte_array):
    list_of_ranges = []
    rng_min = -1
    rng_max = -1
    counting_range = False
    for idx, condition in enumerate(boolean_cte_array):
        if condition == True:
            if not counting_range:
                rng_min = idx
                counting_range = True
            else:
                rng_max = idx
                counting_range = True
        else:
            if counting_range:
                if rng_max == -1:
                    list_of_ranges.append(rng_min)
                else:
                    list_of_ranges.append([rng_min,rng_max])
                counting_range = False
                rng_min = -1
                rng_max = -1
    return list_of_ranges

def plot_ranges(ax, cte_anomalous, alpha=0.2, YELLOW_BORDER = 3.6,ORANGE_BORDER = 5.0, RED_BORDER = 7.0):
    # plot cross track error values: 
    # yellow_condition: reaching the borders of the track: yellow
    # orange_condition: on the borders of the track (partial crossing): orange
    # red_condition: out of track (full crossing): red

    yellow_condition = (abs(cte_anomalous)>YELLOW_BORDER)&(abs(cte_anomalous)<ORANGE_BORDER)
    orange_condition = (abs(cte_anomalous)>ORANGE_BORDER)&(abs(cte_anomalous)<RED_BORDER)
    red_condition = (abs(cte_anomalous)>RED_BORDER)

    yellow_ranges = get_ranges(yellow_condition)
    orange_ranges = get_ranges(orange_condition)
    red_ranges = get_ranges(red_condition)

    all_ranges = [yellow_ranges, orange_ranges, red_ranges]
    colors = ['yellow', 'orange', 'red']
    for idx, list_of_ranges in enumerate(all_ranges):
        for rng in list_of_ranges:
            if isinstance(rng, list):
                ax.axvspan(rng[0], rng[1], color=colors[idx], alpha=alpha)
            else:
                ax.axvspan(rng, rng+1, color=colors[idx], alpha=alpha)

def plot_crash_ranges(ax, speed_anomalous):
    # plot crash instances: speed < 1.0 
    crash_condition = (abs(speed_anomalous)<1.0)
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
        for speed in speed_anomalous[crash_frame-NUM_OF_FRAMES_TO_CHECK:crash_frame]:
            if speed > 15.0:
                is_crash_instance = True
        if is_crash_instance == True:
            is_crash_instance = False
            reset_frame = crash_frame
            ax.axvline(x = reset_frame, color = 'blue', linestyle = '--')
            continue
        # plot crash ranges (speed < 1.0)
        if isinstance(rng, list):
            ax.axvspan(rng[0], rng[1], color='teal', alpha=0.2)
        else:
            ax.axvspan(rng, rng+1, color='teal', alpha=0.2)

def get_heatmaps(anomalous_frame, anomalous, nominal, pos_mappings, return_size=False, return_IMAGE = False):
    # load the addresses of centeral camera heatmap of this anomalous frame and the closest nominal frame in terms of position
    ano_hm_address = anomalous['center'].iloc[anomalous_frame]
    closest_nom_hm_address = nominal['center'].iloc[int(pos_mappings[anomalous_frame])]
    # correct windows path, if necessary
    ano_hm_address = correct_windows_path(ano_hm_address)
    closest_nom_hm_address = correct_windows_path(closest_nom_hm_address)
    # load corresponding heatmaps
    if not return_IMAGE:
        ano_hm = mpimg.imread(ano_hm_address)
        closest_nom_hm = mpimg.imread(closest_nom_hm_address)
        if ano_hm.shape != closest_nom_hm.shape:
            raise ValueError(Fore.RED + f"Different heatmap sizes for nominal and anomalous conditions!" + Fore.RESET)
    else:
        ano_hm = Image.open(ano_hm_address)
        closest_nom_hm = Image.open(closest_nom_hm_address)
    if return_size:
        return ano_hm.shape[0], ano_hm.shape[1]
    else:
        return ano_hm, closest_nom_hm

def get_images(cfg, anomalous_frame, pos_mappings):
    # load the image file paths from main csv
    ano_csv_path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME,
                                'driving_log.csv')
    nom_csv_path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME_NOMINAL,
                                'driving_log.csv')
    ano_data = pd.read_csv(ano_csv_path)
    ano_img_address = ano_data["center"].iloc[anomalous_frame]
    nom_data = pd.read_csv(nom_csv_path)
    closest_nom_img_address = nom_data['center'].iloc[int(pos_mappings[anomalous_frame])]
    ano_img_address = correct_windows_path(ano_img_address)
    closest_nom_img_address = correct_windows_path(closest_nom_img_address)
    ano_img = Image.open(ano_img_address)
    closest_nom_img = Image.open(closest_nom_img_address)   
    return ano_img, closest_nom_img

def save_ax_nosave(ax, **kwargs):
    import io
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted() 
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox,  **kwargs)
    ax.axis("on")
    buff.seek(0)
    # im = plt.imread(buff)
    im = Image.open(buff)
    return im

# Video creation functions

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def make_avi(image_folder, video_folder_path, name):
    video_name = f'{name}.avi'
    video_path = os.path.join(video_folder_path, video_name)
    if not os.path.exists(video_path):
        cprintf('Creating video ...', 'l_cyan')
        # path to video folder
        if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path)
        images = [img for img in os.listdir(image_folder)]
        sort_nicely(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        fps = 10
        video = cv2.VideoWriter(video_path, 0, fps, (width,height))
        for image in tqdm(images):
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()
    else:
        cprintf('Video already exists. Skipping video creation ...', 'l_green')

def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{name}.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

######################################################################################
#####################################   TEST   #######################################
######################################################################################

def test(cfg, NOMINAL_PATHS, ANOMALOUS_PATHS, NUM_FRAMES_NOM, NUM_FRAMES_ANO, 
         heatmap_type, anomalous_simulation_name, nominal_simulation_name,
         distance_method, distance_type, pca_dimension, run_id):

    # PATHS = [SIM_PATH, MAIN_CSV_PATH, HEATMAP_PARENT_FOLDER_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH]
    NOMINAL_SIM_PATH = NOMINAL_PATHS[0]
    NOMINAL_MAIN_CSV_PATH = NOMINAL_PATHS[1]
    NOMINAL_HEATMAP_PARENT_FOLDER_PATH = NOMINAL_PATHS[2]
    NOMINAL_HEATMAP_FOLDER_PATH = NOMINAL_PATHS[3]
    NOMINAL_HEATMAP_CSV_PATH = NOMINAL_PATHS[4]
    NOMINAL_HEATMAP_IMG_PATH = NOMINAL_PATHS[5]

    ANOMALOUS_SIM_PATH = ANOMALOUS_PATHS[0]
    ANOMALOUS_MAIN_CSV_PATH = ANOMALOUS_PATHS[1]
    ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH = ANOMALOUS_PATHS[2]
    ANOMALOUS_HEATMAP_FOLDER_PATH = ANOMALOUS_PATHS[3]
    ANOMALOUS_HEATMAP_CSV_PATH = ANOMALOUS_PATHS[4]
    ANOMALOUS_HEATMAP_IMG_PATH = ANOMALOUS_PATHS[5]


    # 1. find the closest frame from anomalous sim to the frame in nominal sim (comparing car position)
    # nominal simulation
    cprintf(f"Path for data_df_nominal: {NOMINAL_HEATMAP_CSV_PATH}", 'white')
    data_df_nominal = pd.read_csv(NOMINAL_HEATMAP_CSV_PATH)
    nominal = pd.DataFrame(data_df_nominal['frameId'].copy(), columns=['frameId'])
    nominal['position'] = data_df_nominal['position'].copy()
    nominal['center'] = data_df_nominal['center'].copy()
    nominal['steering_angle'] = data_df_nominal['steering_angle'].copy()
    nominal['throttle'] = data_df_nominal['throttle'].copy()
    nominal['speed'] = data_df_nominal['speed'].copy()
    # total number of nominal frames
    num_nominal_frames = pd.Series.max(nominal['frameId']) + 1
    if NUM_FRAMES_NOM != num_nominal_frames:
        raise ValueError(Fore.RED + f'Mismatch in number of \"nominal\" frames: {NUM_FRAMES_NOM} != {num_nominal_frames}' + Fore.RESET)
    
    # anomalous simulation
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

    # Localization
    # path to csv file containing mapped positions
    POSITIONAL_MAPPING_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH, f'pos_mappings_{nominal_simulation_name}.csv')
      
    # check if positional mapping list file in csv format already exists 
    if not os.path.exists(POSITIONAL_MAPPING_PATH):
        cprintf(f"Positional mapping list for {anomalous_simulation_name} and {nominal_simulation_name} does not exist. Generating list ...", 'l_blue')
        pos_mappings = np.zeros(num_anomalous_frames, dtype=float)
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
            closest, _ = pairwise_distances_argmin_min(sample_point, nominal_positions)
            pos_mappings[anomalous_frame] = closest
        # save list of positional mappings
        cprintf(f"Saving positional mappings to CSV file at {POSITIONAL_MAPPING_PATH} ...", 'l_yellow')
        np.savetxt(POSITIONAL_MAPPING_PATH, pos_mappings, delimiter=",")
    else:
        cprintf(f"Positional mapping list exists.", 'l_green')
        # load list of mapped positions
        cprintf(f"Loading CSV file from {POSITIONAL_MAPPING_PATH} ...", 'l_yellow')
        pos_mappings = np.loadtxt(POSITIONAL_MAPPING_PATH, dtype='int')

    # np.set_printoptions(threshold=sys.maxsize)
    print(pos_mappings)
    if len(pos_mappings) != num_anomalous_frames:
        raise ValueError(Fore.RED + f"Length of oaded positional mapping array (from CSV file above) \"{len(pos_mappings)}\" does not match the number of anomalous frames: {num_anomalous_frames}" + Fore.RESET)
    # np.set_printoptions(threshold=False)

    # 2. Principal component analysis to project heat-maps to a point in a lower dim space
    pca = PCA(n_components=pca_dimension)

    # initialize arrays
    ht_height, ht_width = get_heatmaps(0, anomalous, nominal, pos_mappings, return_size=True, return_IMAGE=False)
    x_ano_all_frames = np.zeros((num_anomalous_frames, ht_height*ht_width))
    x_nom_all_frames = np.zeros((num_anomalous_frames, ht_height*ht_width))

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
    for anomalous_frame in tqdm(range(num_anomalous_frames)):
        # load the centeral camera heatmaps of this anomalous frame and the closest nominal frame in terms of position
        ano_heatmap, closest_nom_heatmap = get_heatmaps(anomalous_frame, anomalous, nominal, pos_mappings, return_size=False, return_IMAGE=False)

        # convert to grayscale
        x_ano = cv2.cvtColor(ano_heatmap, cv2.COLOR_BGR2GRAY)
        x_nom = cv2.cvtColor(closest_nom_heatmap, cv2.COLOR_BGR2GRAY)

        ano_std_scale = preprocessing.StandardScaler().fit(x_ano)
        x_ano_std = ano_std_scale.transform(x_ano)
        nom_std_scale = preprocessing.StandardScaler().fit(x_nom)
        x_nom_std = nom_std_scale.transform(x_nom)

        x_ano_all_frames[anomalous_frame,:] = x_ano_std.flatten()
        x_nom_all_frames[anomalous_frame,:] = x_nom_std.flatten()

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

            ano_sa = anomalous['steering_angle'].iloc[anomalous_frame]
            nom_sa = nominal['steering_angle'].iloc[int(pos_mappings[anomalous_frame])]
            ano_throttle = anomalous['throttle'].iloc[anomalous_frame]
            nom_throttle = nominal['throttle'].iloc[int(pos_mappings[anomalous_frame])]
            ano_speed = anomalous['speed'].iloc[anomalous_frame]
            nom_speed = nominal['speed'].iloc[int(pos_mappings[anomalous_frame])]

            font = ImageFont.truetype(os.path.join(cfg.TESTING_DATA_DIR,"arial.ttf"), 18)
            draw_ano = ImageDraw.Draw(ano_img)
            draw_ano.text((0, 0),f"Anomalous: steering_angle:{float(ano_sa):.2f} // throttle:{float(ano_throttle):.2f} // speed:{float(ano_speed):.2f}",(255,255,255),font=font)
            draw_nom = ImageDraw.Draw(closest_nom_img)
            draw_nom.text((10, 0),f"Nominal: steering_angle:{float(nom_sa):.2f} // throttle:{float(nom_throttle):.2f} // speed:{float(nom_speed):.2f}",(255,255,255),font=font)

            # create and save collage
            base_collage = Image.new("RGBA", (1280,640))
            base_collage.paste(ano_img, (0,0))
            base_collage.paste(closest_nom_img, (640,0))
            base_collage.paste(ano_heatmap, (0,320))
            base_collage.paste(closest_nom_heatmap, (640,320))
            base_collage.save(os.path.join(COLLAGES_BASE_FOLDER_PATH, f'FID_{anomalous_frame}.png'))
            
    if cfg.GENERATE_SUMMARY_COLLAGES:
        if missing_collages > 0 and missing_collages != num_anomalous_frames:
            cprintf(f'There are missing collages. Deleting base folder ...', 'l_red')
            shutil.rmtree(COLLAGES_BASE_FOLDER_PATH)
            raise ValueError(Fore.RED + "Error in number of saved collages. Removing all the collages! Please rerun the code." + Fore.RESET)
        elif missing_collages == 0:
            cprintf(f"All base collages already exist.", 'l_green')
        
    print(f'x_ano_std.shape is {x_ano_std.shape}')
    print(f'x_nom_std.shape is {x_nom_std.shape}')
    print(f'x_ano_all_frames.shape is {x_ano_all_frames.shape}')
    print(f'x_nom_all_frames.shape is {x_nom_all_frames.shape}')

    if cfg.SAVE_PCA:
        cprintf(f"WARNING: It's best to calculate PCA instead of saving it. Loading method is not tested", 'yellow')
        # path to pca csv file
        PCA_ANO_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                    f'pca_ano_{pca_dimension}d.csv')
        PCA_NOM_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                    f'pca_nom_{pca_dimension}d.csv')
        
        # check if positional mapping list file in csv format already exists 
        if (not os.path.exists(PCA_ANO_PATH)) or (not os.path.exists(PCA_NOM_PATH)):
            cprintf(f"PCA array for PCA dimension \"{pca_dimension}\" does not exist. Generating ...", 'l_blue')
            cprintf(f"Performing PCA conversion with {pca_dimension} dimensions ...", 'l_cyan')
            # PCA conversion
            pca_ano = pca.fit_transform(x_ano_all_frames)
            pca_nom = pca.fit_transform(x_nom_all_frames)
            # save list of positional mappings
            cprintf(f"Saving ano and nom PCA arrays to CSV files ..." ,'l_yellow')
            np.savetxt(PCA_ANO_PATH, pca_ano, delimiter=",")
            np.savetxt(PCA_NOM_PATH, pca_nom, delimiter=",")
        else:
            cprintf(f"PCA CSV files already exist.", 'l_green')
            # load list of mapped positions
            cprintf(f"Loading CSV file {PCA_ANO_PATH} ...", 'l_yellow')
            pca_ano = np.genfromtxt(PCA_ANO_PATH, delimiter=',', dtype='float')[:,:-1]
            cprintf(f"Loading CSV file {PCA_NOM_PATH} ...", 'l_yellow')
            pca_nom = np.genfromtxt(PCA_NOM_PATH, delimiter=',', dtype='float')[:,:-1]
    else:
        cprintf(f"Performing PCA conversion using {pca_dimension} dimensions ...", 'l_cyan')
        pca_ano = pca.fit_transform(x_ano_all_frames)
        pca_nom = pca.fit_transform(x_nom_all_frames)
        cprintf(f'pca_ano.shape is {pca_ano.shape}', 'l_magenta')
        cprintf(f'pca_nom.shape is {pca_nom.shape}', 'l_magenta')

    # path to csv file containing distance vector
    DISTANCE_VECTOR_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                        f'dist_vect_{distance_method}_{distance_type}_{pca_dimension}d.csv')
      
    # check if positional mapping list file in csv format already exists 
    if not os.path.exists(DISTANCE_VECTOR_PATH):
        cprintf(f"Distance vector list for \"{distance_method}\" of type \"{distance_type}\" and PCA dimension \"{pca_dimension}\" does not exist. Generating list ...", 'l_blue')
        if distance_method == 'pairwise_distance':
            distance_vector = pairwise.paired_distances(pca_ano, pca_nom, metric=distance_type)
        else:
            raise ValueError(f"Distance method \"{distance_method}\" is not defined.")
        # save list of positional mappings
        cprintf(f"Saving distance vector to CSV file at {DISTANCE_VECTOR_PATH} ...", 'l_yellow')
        np.savetxt(DISTANCE_VECTOR_PATH, distance_vector, delimiter=",")
    else:
        cprintf(f"Distance vector list already exists.", 'l_green')
        # load list of mapped positions
        cprintf(f"Loading CSV file from {DISTANCE_VECTOR_PATH} ...", 'l_yellow')
        distance_vector = np.loadtxt(DISTANCE_VECTOR_PATH, dtype='float')

    fig = plt.figure(figsize=(20,15), constrained_layout=False)
    spec = fig.add_gridspec(nrows=3, ncols=1, width_ratios= [1], height_ratios=[3, 1, 1])

    # Plot pca cluster
    if pca_dimension == 2:
        ax = fig.add_subplot(spec[0, :])
        ax.scatter(pca_ano[:,0], pca_ano[:,1], color = 'r')
        ax.scatter(pca_nom[:,0], pca_nom[:,1], color = 'b')  
    else:
        ax = fig.add_subplot(spec[0, :], projection='3d')
        ax.scatter(pca_ano[:,0], pca_ano[:,1], pca_ano[:,2], color = 'r')
        ax.scatter(pca_nom[:,0], pca_nom[:,1], pca_nom[:,2], color = 'b')       

    # Plot distance vector/ranges
    ax = fig.add_subplot(spec[1, :])

    # plot preprocessing
    # anomalous cross track errors
    cte_anomalous = data_df_anomalous['cte']
    # car speed in anomaluos mode
    speed_anomalous = data_df_anomalous['speed']
    YELLOW_BORDER = 3.6
    ORANGE_BORDER = 5.0
    RED_BORDER = 7.0
    plot_ranges(ax, cte_anomalous, alpha=0.2, YELLOW_BORDER=YELLOW_BORDER,
                ORANGE_BORDER=ORANGE_BORDER, RED_BORDER=RED_BORDER)
    plot_crash_ranges(ax, speed_anomalous)

    # Plot distance scores
    color = 'blue'
    ax.set_xlabel('Frame ID', color=color)
    ax.set_ylabel(f'{distance_type} distance scores', color=color)
    ax.plot(distance_vector, label=distance_method, linewidth= 0.5, linestyle = '-', color=color)
    title = f"{heatmap_type} && {pca_dimension}d"
    ax.set_title(title)
    ax.legend(loc='upper left')

    if cfg.GENERATE_SUMMARY_COLLAGES:
        missing_collages = 0
        # path to collages folder
        COLLAGES_FOLDER_PATH = os.path.join(COLLAGES_PARENT_FOLDER_PATH, f"{distance_type}_{distance_method}_{pca_dimension}d")

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
            if distance_method in cfg.SUMMARY_COLLAGE_DIST_METHODS:
                collage_name = f'_{distance_method}'
            else:
                cprintf(f' No summary collages for distance method: {distance_method} ...', 'yellow')
                cprintf(f'Removing completed collages folder ...', 'yellow')
                shutil.rmtree(COLLAGES_FOLDER_PATH)
                break
            if (pca_dimension == num_anomalous_frames) or (pca_dimension in cfg.SUMMARY_COLLAGE_PCA_DIMS):
                collage_name = collage_name + f'_{pca_dimension}d_FID_{anomalous_frame}.png'
            else:
                cprintf(f' No summary collages for PCA dimesion: {pca_dimension} ...', 'yellow')
                cprintf(f'Removing completed collages folder ...', 'yellow')
                shutil.rmtree(COLLAGES_FOLDER_PATH)
                break

            create_collage = True
            if anomalous_frame == 0:
                cprintf(f' Generating complete summary collages for {distance_type}_{distance_method}_{pca_dimension}d ...', 'l_cyan')

            # double-check if every collage actually exists
            if collage_name in os.listdir(COLLAGES_FOLDER_PATH):
                continue
            else:
                missing_collages += 1 
            # load saved base collage
            base_collage_img = Image.open(os.path.join(COLLAGES_BASE_FOLDER_PATH, f'FID_{anomalous_frame}.png'))
            ax.axvline(x = anomalous_frame, color = 'black', linestyle = '-')
            distance_plot_img = save_ax_nosave(ax)
            distance_plot_img = distance_plot_img.resize((1280,320))
            ax.lines.pop(-1)
            collage = Image.new("RGBA", (1280,960))
            collage.paste(base_collage_img, (0,0))
            collage.paste(distance_plot_img, (0,640))
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

        if cfg.CREATE_GIF:
            VIDEO_FOLDER_PATH = os.path.join(COLLAGES_PARENT_FOLDER_PATH, 'video')
            make_avi(COLLAGES_FOLDER_PATH, VIDEO_FOLDER_PATH, f"{distance_type}_{distance_method}_{pca_dimension}d")
        
    # Plot cross track error
    ax = fig.add_subplot(spec[2, :])
    color = 'red'
    ax.set_xlabel('Frame ID', color=color)
    ax.set_ylabel('cross-track error', color=color)
    ax.plot(cte_anomalous, label='cross-track error', linewidth= 0.5, linestyle = '-', color=color)

    ax.axhline(y = YELLOW_BORDER, color = 'yellow', linestyle = '--')
    ax.axhline(y = -YELLOW_BORDER, color = 'yellow', linestyle = '--')
    ax.axhline(y = ORANGE_BORDER, color = 'orange', linestyle = '--')
    ax.axhline(y = -ORANGE_BORDER, color = 'orange', linestyle = '--')
    ax.axhline(y = RED_BORDER, color = 'red', linestyle = '--')
    ax.axhline(y = -RED_BORDER, color = 'red', linestyle = '--')    

    title = f"{heatmap_type} && {pca_dimension}d"
    ax.set_title(title)
    ax.legend(loc='upper left')
 
    # path to plotted figure images folder
    FIGURES_FOLDER_PATH = os.path.join(ANOMALOUS_HEATMAP_FOLDER_PATH,
                                       f"figures_{anomalous_simulation_name}_{nominal_simulation_name}")
    if not os.path.exists(FIGURES_FOLDER_PATH):
        os.makedirs(FIGURES_FOLDER_PATH)

    ALL_RUNS_FIGURES_FOLDER_PATH = os.path.join(ANOMALOUS_HEATMAP_PARENT_FOLDER_PATH, "figures_all_runs")
    ALL_RUNS_FIGURE_PATH = os.path.join(ALL_RUNS_FIGURES_FOLDER_PATH,
                                        f"{anomalous_simulation_name}_{nominal_simulation_name}",
                                        f"pca_{pca_dimension}d")
    if not os.path.exists(ALL_RUNS_FIGURE_PATH):
        cprintf(f'All runs\' figures folder does not exist. Creating folder ...', 'l_blue')
        os.makedirs(ALL_RUNS_FIGURE_PATH)


    cprintf(f'\nSaving plotted figure ...', 'l_yellow')
    plt.savefig(os.path.join(FIGURES_FOLDER_PATH, f"{distance_type}_{pca_dimension}d.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(ALL_RUNS_FIGURE_PATH, f"run_id_{run_id}_{distance_type}.png"), bbox_inches='tight', dpi=300)


    # plt.show()
    # a = ScrollableWindow(fig)
    # b = ScrollableGraph(fig, ax)
