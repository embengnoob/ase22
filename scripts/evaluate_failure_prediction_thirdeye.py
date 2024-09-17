import utils
from utils import *
from utils_models import *

######################################################################################
################### EVALUATION FUNCTION FOR THE THIRDEYE METHOD ######################
######################################################################################


def evaluate_failure_prediction_thirdeye(cfg, NOMINAL_PATHS, ANOMALOUS_PATHS, seconds_to_anticipate, anomalous_simulation_name, nominal_simulation_name,
                                         heatmap_type, summary_types, aggregation_methods):
    
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

    return results_csv_path


