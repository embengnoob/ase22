import sys
sys.path.append("..")
import utils
from utils import *


# scores of different weather and lighting conditions for different nominal sim types

DISTANCE_TYPES = ['sobolev-norm', 'euclidean'] #'euclidean', 'EMD', 'moran', 'mutual-info', 'sobolev-norm'
HEATMAP_TYPES = ['SmoothGrad', 'RectGrad'] #'SmoothGrad', 'GradCam++', 'RectGrad', 'RectGrad_PRR', 'Saliency', 'Guided_BP', 'SmoothGrad_2', 'Gradient-Input', 'IntegGrad', 'Epsilon_LRP'
ANO_SIMULATIONS =   ['track1-night-fog-100',
                    'track1-night-snow-100']
RUN_ID_NUMBERS = [[1, 2],
                 [1, 2]]
SECONDS_TO_ANTICIPATE = [1, 2, 3]

hm_nf_1 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-fog-100\results\1\results_ano_track1-night-fog-100_nom_track1-sunny-nominal_total_scores_heatmaps.csv"
dt_nf_1 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-fog-100\results\1\results_ano_track1-night-fog-100_nom_track1-sunny-nominal_total_scores_distance_types.csv"
hm_nf_2 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-fog-100\results\2\results_ano_track1-night-fog-100_nom_track1-sunny-nominal_total_scores_heatmaps.csv"
dt_nf_2 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-fog-100\results\2\results_ano_track1-night-fog-100_nom_track1-sunny-nominal_total_scores_distance_types.csv"

hm_ns_1 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-snow-100\results\1\results_ano_track1-night-snow-100_nom_track1-sunny-nominal_total_scores_heatmaps.csv"
dt_ns_1 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-snow-100\results\1\results_ano_track1-night-snow-100_nom_track1-sunny-nominal_total_scores_distance_types.csv"
hm_ns_2 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-snow-100\results\2\results_ano_track1-night-snow-100_nom_track1-sunny-nominal_total_scores_heatmaps.csv"
dt_ns_2 = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-snow-100\results\2\results_ano_track1-night-snow-100_nom_track1-sunny-nominal_total_scores_distance_types.csv"

hm_total_scores_paths = [[hm_nf_1, hm_nf_2], [hm_ns_1, hm_ns_2]]
dt_total_scores_paths = [[dt_nf_1, dt_nf_2], [dt_ns_1, dt_ns_2]]

def f_beta_score(precision, recall, beta=3):
    numerator = (1 + beta ** 2) * (precision * recall)
    denominator = (beta ** 2 * precision) + recall
    f_beta_score = numerator / denominator
    return f_beta_score

def heatmap_type_scores(hm_total_scores_paths, HEATMAP_TYPES, sim_idx, sim_name, number_of_runs,
                        seconds_to_anticipate, ht_scores_df, last_index, print_results=False):
    if ht_scores_df is None:
        raise ValueError("Received None as dataframe")
    
    # add simulation name row
    sim_name_row = ['Simulation']
    for col_idx in range((len(seconds_to_anticipate)+1)*4):
        if col_idx == 0:
            sim_name_row.append(sim_name)
        else:
            sim_name_row.append('-')
    ht_scores_df.iloc[last_index] = sim_name_row

    for heatmap_type in HEATMAP_TYPES:
        last_index += 1
        if print_results:
            print(heatmap_type)
        # replace 0th column with heatmap names
        ht_scores_df.at[last_index, 'Window Size'] = heatmap_type
        # reset column number to 1 for after each heatmap type
        col_ctr = 1
        # reset sum arrays including sum vars for each sta for after each heatmap type
        avg_prec_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        avg_re_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        f3_scores_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        avg_acc_sum = np.zeros((len(seconds_to_anticipate)), dtype=float)
        
        avg_prec_all_sum = 0.0
        avg_rec_all_sta_sum = 0.0
        f3_score_all_sta_sum = 0.0
        avg_acc_all_sta_sum = 0.0

        for run_idx in range(number_of_runs):
            for sta_idx, sta in enumerate(seconds_to_anticipate):
                if print_results:
                    print('sta:' + str(sta))

                # read heatmap results csv file for this run_id and filter by heatmap type
                ht_results_df = pd.read_csv(hm_total_scores_paths[sim_idx][run_idx])
                filter_by_sta = ht_results_df[(ht_results_df['sta'] == sta) & (ht_results_df['heatmap_type'] == heatmap_type)]
                # precision
                precision = filter_by_sta['precision'].values
                avg_precision = np.average(precision)
                avg_prec_sum[sta_idx] += avg_precision
                if print_results:
                    print(f'avg_prec_sum[sta_idx:{sta_idx}]: {avg_prec_sum}')
                # recall
                recall = filter_by_sta['recall'].values
                avg_recall = np.average(recall)
                avg_re_sum[sta_idx] += avg_recall
                if print_results:
                    print(f'avg_re_sum[sta_idx:{sta_idx}]: {avg_re_sum}')
                # f3
                f3_score = f_beta_score(avg_precision, avg_recall, beta=3)
                f3_scores_sum[sta_idx] += f3_score
                if print_results:
                    print(f'f3_scores_sum[sta_idx:{sta_idx}]: {f3_scores_sum}')
                # accuracy
                accuracy = filter_by_sta['accuracy'].values
                avg_accuracy = np.average(accuracy)
                avg_acc_sum[sta_idx] += avg_accuracy
                if print_results:
                    print(f'avg_acc_sum[sta_idx:{sta_idx}]: {avg_acc_sum}')

                if print_results:
                    print('run_number:' + str(run_idx))
                    cprintf(f'sta: avg_precision: {sta}: {avg_precision*100}', 'l_green')
                    cprintf(f'sta: avg_recall: {sta}: {avg_recall*100}', 'l_yellow')
                    cprintf(f'sta: f3_score: {sta}: {f3_score*100}', 'l_red')    
                    cprintf(f'sta: avg_accuracy: {sta}: {avg_accuracy*100}', 'l_blue')
            
                # save average scores between multiple runs to dataframe
                if run_idx == number_of_runs-1:
                    if print_results:
                        print(avg_prec_sum[sta_idx]/(number_of_runs))
                        print(avg_re_sum[sta_idx]/(number_of_runs))
                        print(f3_scores_sum[sta_idx]/(number_of_runs))
                        print(avg_acc_sum[sta_idx]/(number_of_runs))
                        print(last_index, col_ctr)
                    ht_scores_df.iat[last_index, col_ctr] = avg_prec_sum[sta_idx]/(number_of_runs)
                    ht_scores_df.iat[last_index, col_ctr+1] = avg_re_sum[sta_idx]/(number_of_runs)
                    ht_scores_df.iat[last_index, col_ctr+2] = f3_scores_sum[sta_idx]/(number_of_runs)
                    ht_scores_df.iat[last_index, col_ctr+3] = avg_acc_sum[sta_idx]/(number_of_runs)
                    col_ctr += 4
                    if print_results:
                        print(last_index, col_ctr)
                        print('------------------------------------')
                    
            # avg of all stas
            filter_by_sta = ht_results_df[(ht_results_df['sta'] == 1)] # the ..._all value for sta of 1, 2, or 3 is the same.
            # precision
            precision_all_sta = filter_by_sta['precision_all'].values
            avg_precision_all_sta = np.average(precision_all_sta)
            avg_prec_all_sum += avg_precision_all_sta

            # recall
            recall_all_sta = filter_by_sta['recall_all'].values
            avg_recall_all_sta = np.average(recall_all_sta)
            avg_rec_all_sta_sum += avg_recall_all_sta
            # f3
            f3_score_all_sta = f_beta_score(avg_precision_all_sta, avg_recall_all_sta, beta=3)
            f3_score_all_sta_sum += f3_score_all_sta
            # accuracy
            accuracy_all_sta = filter_by_sta['accuracy_all'].values
            avg_accuracy_all_sta = np.average(accuracy_all_sta)
            avg_acc_all_sta_sum += avg_accuracy_all_sta

            if run_idx == number_of_runs-1:
                if print_results:
                    print(avg_prec_all_sum/(number_of_runs))
                    print(avg_rec_all_sta_sum/(number_of_runs))
                    print(f3_score_all_sta_sum/(number_of_runs))
                    print(avg_acc_all_sta_sum/(number_of_runs))
                    print(last_index, col_ctr)
                ht_scores_df.iat[last_index, col_ctr] = avg_prec_all_sum/(number_of_runs)
                ht_scores_df.iat[last_index, col_ctr+1] = avg_rec_all_sta_sum/(number_of_runs)
                ht_scores_df.iat[last_index, col_ctr+2] = f3_score_all_sta_sum/(number_of_runs)
                ht_scores_df.iat[last_index, col_ctr+3] = avg_acc_all_sta_sum/(number_of_runs)
                col_ctr += 4
                if print_results:
                    print(last_index, col_ctr)

            if print_results:
                print('run_number:' + str(run_idx))
                cprintf(f'sta: avg_precision_all_sta: all: {avg_precision_all_sta*100}', 'l_green')
                cprintf(f'sta: avg_recall_all_sta: all: {avg_recall_all_sta*100}', 'l_yellow')
                cprintf(f'sta: f3_score_all_sta: all: {f3_score_all_sta*100}', 'l_red')
                cprintf(f'sta: avg_accuracy_all_sta: all: {avg_accuracy_all_sta*100}', 'l_blue')
                print('------------------------------')
    last_index += 1
    return ht_scores_df, last_index
    
def create_result_df(hm_total_scores_paths, HEATMAP_TYPES, ANO_SIMULATIONS, sim_idx, number_of_runs, seconds_to_anticipate):
    # test if the number of paths and runs are the same:
    if not (len(hm_total_scores_paths[sim_idx]) == number_of_runs):
            raise ValueError(Fore.RED + f"Mismatch in number of runs per simlation and number of heatmap result score csv paths: {len(hm_total_scores_paths[sim_idx])} != {number_of_runs}" + Fore.RESET)
    # build the top rows of column names
    seconds_to_anticipate_str = []
    for sta in seconds_to_anticipate:
        seconds_to_anticipate_str.append(str(sta)+'s')
    seconds_to_anticipate_str.append('All')
    scores_df = pd.DataFrame(np.zeros(((len(HEATMAP_TYPES)+1)*len(ANO_SIMULATIONS), 17)))
    col_names_row_1 = ['Window Size']
    col_names_row_2 = ['Criteria']
    for sta_str in seconds_to_anticipate_str:
        for col_idx in range(4):
            col_names_row_1.append(sta_str)
        col_names_row_2.append('Precision')
        col_names_row_2.append('Recall')
        col_names_row_2.append('F3')
        col_names_row_2.append('Accuracy')
    scores_df.columns = pd.MultiIndex.from_arrays([col_names_row_1, col_names_row_2])
    return scores_df

NUMBER_OF_RUNS = len(RUN_ID_NUMBERS[0])
last_index = 0
for sim_idx, sim_name in enumerate(ANO_SIMULATIONS):
    if sim_idx == 0:
        ht_scores_df = create_result_df(hm_total_scores_paths, HEATMAP_TYPES, ANO_SIMULATIONS, sim_idx, NUMBER_OF_RUNS, SECONDS_TO_ANTICIPATE)
    ht_scores_df, last_index = heatmap_type_scores(hm_total_scores_paths, HEATMAP_TYPES, sim_idx, sim_name, NUMBER_OF_RUNS, SECONDS_TO_ANTICIPATE, ht_scores_df, last_index, print_results=False)
# print(ht_scores_df)
ht_scores_df.to_csv('out.csv', index=False)
ht_scores_df.to_excel("output.xlsx") 



#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************#*************


# # Threshold nom sim comparison plot
# # Data
# simulations = ['Night Moon', 'Night Snow', 'Night Rain', 'Night Fog', 'Day Rain', 'Day Fog', 'Day Snow', 'Average Night', 'Average Day', 'Average All']
# similar_nom = [75, 71, 67, 47, 83, 83, 85, 65, 83.7, 63.9]
# sunny_nom = [82, 89.5, 88, 92, 90, 78, 85, 87.9, 84.4, 75.6]
# avg_threshold = [75, 80, 73, 84, 86, 78, 85, 78, 83, 70.2]

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(simulations, similar_nom, marker='o', label='Similar Nom')
# plt.plot(simulations, sunny_nom, marker='s', label='Sunny Nom')
# plt.plot(simulations, avg_threshold, marker='^', label='Avg Threshold')

# # Customize plot
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Lighting/Weather Conditions')
# plt.ylabel('Accuracy')
# plt.title('Performance Comparison of Different Nominal Data')
# plt.legend()
# plt.grid(True)

# # Show plot
# plt.tight_layout()
# plt.show()


# # Heatmap based evaluation
# DISTANCE_TYPES = ['euclidean', 'EMD', 'sobolev-norm']
# HEATMAP_TYPES = ['SmoothGrad', 'GradCam++', 'RectGrad', 'RectGrad_PRR', 'Saliency', 'Guided_BP', 'Gradient-Input', 'IntegGrad', 'Epsilon_LRP']

# def f_beta_score(precision, recall, beta=3):
#     if precision == recall == 0:
#         return -1
#     else:
#         numerator = (1 + beta ** 2) * (precision * recall)
#         denominator = (beta ** 2 * precision) + recall
#         f_beta_score = numerator / denominator
#         return f_beta_score

# results_csv_path = r"D:\ThirdEye\ase22\simulations\track1-night-snow-100-anomalous-2\1\results_ano_track1-night-snow-100-anomalous-2_nom_track1-sunny-positioned-nominal_total_scores_heatmaps.csv"
# # results_csv_path = results_csv_path.replace("\\", "\\\\")
# print(results_csv_path)
# results_df = pd.read_csv(results_csv_path)
# seconds_to_anticipate_list = [1, 2, 3]

# for heatmap_type in HEATMAP_TYPES:
#     print(heatmap_type)
#     for sta in seconds_to_anticipate_list:
#         filter_by_sta_and_heatmap = results_df[(results_df['sta'] == sta) & (results_df['heatmap_type'] == heatmap_type)]

#         precision = filter_by_sta_and_heatmap['precision'].values[0]
#         cprintf(f'{round(precision*100)}', 'l_green')

#         recall = filter_by_sta_and_heatmap['recall'].values[0]
#         cprintf(f'{round(recall*100)}', 'l_yellow')

#         f3_score = f_beta_score(precision, recall, beta=3)
#         cprintf(f'{round(f3_score*100)}', 'l_red')

#         accuracy = filter_by_sta_and_heatmap['accuracy'].values[0]
#         cprintf(f'{round(accuracy*100)}', 'l_blue')

#     filter_by_sta_and_heatmap = results_df[(results_df['sta'] == 1) & (results_df['heatmap_type'] == heatmap_type)]

#     precision_all = filter_by_sta_and_heatmap['precision_all'].values[0]
#     cprintf(f'{round(precision_all*100)}', 'l_green')

#     recall_all = filter_by_sta_and_heatmap['recall_all'].values[0]
#     cprintf(f'{round(recall_all*100)}', 'l_yellow')

#     f3_score_all = f_beta_score(precision_all, recall_all, beta=3)
#     cprintf(f'{round(f3_score_all*100)}', 'l_red')

#     accuracy_all = filter_by_sta_and_heatmap['accuracy_all'].values[0]
#     cprintf(f'{round(accuracy_all*100)}', 'l_blue')





# DISTANCE_TYPES = ['euclidean', 'EMD', 'sobolev-norm']

# def f_beta_score(precision, recall, beta=3):
#     if precision == recall == 0:
#         return -1
#     else:
#         numerator = (1 + beta ** 2) * (precision * recall)
#         denominator = (beta ** 2 * precision) + recall
#         f_beta_score = numerator / denominator
#         return f_beta_score

# results_csv_path = r"D:\ThirdEye\ase22\simulations\test1\1\results_ano_test1_nom_track1-sunny-positioned-nominal_total_scores_distance_types.csv"
# # results_csv_path = results_csv_path.replace("\\", "\\\\")
# print(results_csv_path)
# results_df = pd.read_csv(results_csv_path)
# seconds_to_anticipate_list = [1, 2, 3]

# for distance_type in DISTANCE_TYPES:
#     print(distance_type)
#     for sta in seconds_to_anticipate_list:
#         filter_by_sta_and_heatmap = results_df[(results_df['sta'] == sta) & (results_df['distance_type'] == distance_type)]

#         precision = filter_by_sta_and_heatmap['precision'].values[0]
#         cprintf(f'{round(precision*100)}', 'l_green')

#         recall = filter_by_sta_and_heatmap['recall'].values[0]
#         cprintf(f'{round(recall*100)}', 'l_yellow')

#         f3_score = f_beta_score(precision, recall, beta=3)
#         cprintf(f'{round(f3_score*100)}', 'l_red')

#         accuracy = filter_by_sta_and_heatmap['accuracy'].values[0]
#         cprintf(f'{round(accuracy*100)}', 'l_blue')

#     filter_by_sta_and_heatmap = results_df[(results_df['sta'] == 1) & (results_df['distance_type'] == distance_type)]

#     precision_all = filter_by_sta_and_heatmap['precision_all'].values[0]
#     cprintf(f'{round(precision_all*100)}', 'l_green')

#     recall_all = filter_by_sta_and_heatmap['recall_all'].values[0]
#     cprintf(f'{round(recall_all*100)}', 'l_yellow')

#     f3_score_all = f_beta_score(precision_all, recall_all, beta=3)
#     cprintf(f'{round(f3_score_all*100)}', 'l_red')

#     accuracy_all = filter_by_sta_and_heatmap['accuracy_all'].values[0]
#     cprintf(f'{round(accuracy_all*100)}', 'l_blue')



# import matplotlib.pyplot as plt

# # Data
# simulations = [
#     "Night Moon", "Night Snow", "Night Rain", "Night Fog", "Day Rain", 
#     "Day Fog", "Day Snow", "Day Sun", "Average Night", "Average Day", "Average All"
# ]
# smoothgrad = [86, 95, 96.5, 100, 84, 93, 80, 80, 94.4, 84.3, 89.4]
# gradcam = [71, 51, 46, 33, 99, 79, 99, 44, 50.3, 80.3, 65.3]
# rectgrad = [84, 94, 93, 99, 89, 74, 75, 71, 92.5, 77.3, 84.9]
# rectgrad_prr = [82, 94, 92, 99.5, 97, 75, 82, 76, 91.9, 82.5, 87.2]
# saliency = [88, 95.5, 93.5, 98, 82, 80, 82, 75, 93.8, 79.8, 86.8]
# guided_bp = [86, 95, 93.5, 98.5, 94, 71, 90, 71, 93.3, 81.5, 87.4]
# gradient_input = [79, 90.5, 88.5, 99, 88, 79, 88, 69, 89.3, 81, 85.2]
# integgrad = [80, 95, 91.5, 99.5, 95, 76, 88, 73, 91.5, 80, 85.8]
# epsilon_lrp = [86, 95.5, 96, 99.5, 85, 72, 99, 74, 94.3, 80, 87.2]

# labels = ['SmoothGrad',	'GradCam++', 'RectGrad', 'RectGrad_PRR',	'Saliency', 'Guided_BP', 'Gradient*Input', 'IntegGrad', 'Epsilon_LRP'
# ]
# # Plot
# plt.figure(figsize=(12, 8))

# # Marker styles and colors
# marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X']
# colors = plt.cm.tab10.colors

# # Plot each heatmap type
# for i, (heatmap, marker, color) in enumerate(zip(
#     [smoothgrad, gradcam, rectgrad, rectgrad_prr, saliency, guided_bp, gradient_input, integgrad, epsilon_lrp],
#     marker_styles,
#     colors
# )):
#     plt.scatter(simulations, heatmap, marker=marker, color=color, label=labels[i])

# plt.title('Accuracy Comparison of Different Simulations')
# plt.xlabel('Simulation')
# plt.ylabel('Accuracy (%)')
# plt.xticks(rotation=45, ha='right')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()









# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# simulations = [
#     "Night Moon", "Night Snow", "Night Rain", "Night Fog", "Day Rain", 
#     "Day Fog", "Day Snow", "Day Sun", "Average Night", "Average Day", "Average All"
# ]
# smoothgrad = [86, 95, 96.5, 100, 84, 93, 80, 80, 94.4, 84.3, 89.4]
# gradcam = [71, 51, 46, 33, 99, 79, 99, 44, 50.3, 80.3, 65.3]
# rectgrad = [84, 94, 93, 99, 89, 74, 75, 71, 92.5, 77.3, 84.9]
# rectgrad_prr = [82, 94, 92, 99.5, 97, 75, 82, 76, 91.9, 82.5, 87.2]
# saliency = [88, 95.5, 93.5, 98, 82, 80, 82, 75, 93.8, 79.8, 86.8]
# guided_bp = [86, 95, 93.5, 98.5, 94, 71, 90, 71, 93.3, 81.5, 87.4]
# gradient_input = [79, 90.5, 88.5, 99, 88, 79, 88, 69, 89.3, 81, 85.2]
# integgrad = [80, 95, 91.5, 99.5, 95, 76, 88, 73, 91.5, 80, 85.8]
# epsilon_lrp = [86, 95.5, 96, 99.5, 85, 72, 99, 74, 94.3, 80, 87.2]

# labels = ['SmoothGrad',	'GradCam++', 'RectGrad', 'RectGrad_PRR',	'Saliency', 'Guided_BP', 'Gradient-Input', 'IntegGrad', 'Epsilon_LRP'
# ]

# # Plot
# plt.figure(figsize=(12, 8))

# # Marker styles and colors
# marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X']
# colors = plt.cm.tab10.colors

# # Plot each heatmap type
# for i, (heatmap, marker, color) in enumerate(zip(
#     [smoothgrad, gradcam, rectgrad, rectgrad_prr, saliency, guided_bp, gradient_input, integgrad, epsilon_lrp],
#     marker_styles,
#     colors
# )):
#     plt.scatter(simulations, heatmap, marker=marker, color=color, label=labels[i])

# plt.title('Accuracy Comparison of Different Simulations')
# plt.xlabel('Simulation')
# plt.ylabel('Accuracy (%)')
# plt.xticks(rotation=45, ha='right')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()

# # Save plot as LaTeX code
# plt.savefig('heatmap_comparison.tex', backend='pgf')


























# def splitall(path):
#     allparts = []
#     while 1:
#         parts = os.path.split(path)
#         if parts[0] == path:  # sentinel for absolute paths
#             allparts.insert(0, parts[0])
#             break
#         elif parts[1] == path: # sentinel for relative paths
#             allparts.insert(0, parts[1])
#             break
#         else:
#             path = parts[0]
#             allparts.insert(0, parts[1])
#     return allparts

# def fix_escape_sequences(img_addr):
#     if "\\\\" in img_addr:
#         img_addr = img_addr.replace("\\\\", "/")
#     elif "\\" in img_addr:
#         img_addr = img_addr.replace("\\", "/")
#     elif "\\a" in img_addr:
#         img_addr = img_addr.replace("\\a", "/a")
#     elif "\\t" in img_addr:
#         img_addr = img_addr.replace("\\t", "/t")
#     elif "\\n" in img_addr:
#         img_addr = img_addr.replace("\\n", "")
#     elif "\\b" in img_addr:
#         img_addr = img_addr.replace("\\b", "")
#     return img_addr

# def correct_img_address(img_addr, csv_dir):
#     img_name = Path(img_addr).stem
#     corrected_path = os.path.join(csv_dir, 'IMG', img_name + '.jpg')
#     return corrected_path


# def check_addresses(center_img_addresses, csv_dir):
#     first_center_img_address = fix_escape_sequences(center_img_addresses[0])
#     if not os.path.exists(first_center_img_address):
#         corrected_path = correct_img_address(first_center_img_address, csv_dir)
#         if not os.path.exists(corrected_path):
#             print(corrected_path)
#             raise ValueError(Fore.RED + f"The provided img path in the csv file is not in the same dir or does not exist." + Fore.RESET)
#         return False
#     else:
#         return True


# run_id = 1
# CSV_PATH = r"D:\ThirdEye\ase22\simulations\track1\anomalous\track1-night-moon\heatmaps\heatmaps-smoothgrad\1\driving_log_Copy.csv"
# csv_type = 'heatmap'

# csv_dir = os.path.dirname(CSV_PATH)
# csv_file = pd.read_csv(CSV_PATH)
# center_img_addresses = csv_file["center"]
# if csv_type == 'main':
#     left_img_addresses = csv_file["left"]
#     right_img_addresses = csv_file["right"]

# # if the img exists in the correct path but the path in the csv file is wrong:
# if not check_addresses(center_img_addresses, csv_dir):
#     # center images
#     for idx, img_addr in enumerate(tqdm(center_img_addresses)):
#         fixed_img_addr = fix_escape_sequences(img_addr)
#         corrected_path = correct_img_address(fixed_img_addr, csv_dir)
#         csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
#     if csv_type == 'main':
#         # left images
#         for idx, img_addr in enumerate(tqdm(left_img_addresses)):
#             fixed_img_addr = fix_escape_sequences(img_addr)
#             corrected_path = correct_img_address(fixed_img_addr, csv_dir)
#             csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
#         # right images
#         for idx, img_addr in enumerate(tqdm(right_img_addresses)):
#             fixed_img_addr = fix_escape_sequences(img_addr)
#             corrected_path = correct_img_address(fixed_img_addr, csv_dir)
#             csv_file.replace(to_replace=img_addr, value=corrected_path, inplace=True)
    
#     csv_file.to_csv(CSV_PATH, index=False)

# #     # cprintf(f'{img_addr}', 'l_yellow')
# #     # replace drive letter if database is copied
# #     if os.path.splitdrive(img_addr)[0] != os.path.splitdrive(os.getcwd())[0]:
# #         if '\\ThirdEye\\ase22\\' in img_addr:
# #             img_addr = img_addr.replace(os.path.splitdrive(img_addr)[0], os.path.splitdrive(os.getcwd())[0])
# #     # change the img addresses accordingly if file placement is in the newer cleaner format ({simulation_name}/src/{run_id}) 
# #     # print(splitall(img_addr))
# #     for idx, addr_part in enumerate(splitall(img_addr)):
# #         if addr_part == 'IMG':
# #             addr_part = os.path.join('src', str(run_id), 'IMG')
# #         if idx == 0:
# #             corrected_address = addr_part
# #         else:
# #             corrected_address = os.path.join(prev_addr_chunk, addr_part)

# #         prev_addr_chunk = corrected_address

# #     if idx == 1:
# #         break

# # print(corrected_address)
# # if os.path.exists(corrected_address):
# #     print('true')
    