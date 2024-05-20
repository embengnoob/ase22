import sys
sys.path.append("..")
import utils
from utils import *


# # scores of different weather and lighting conditions for different nominal sim types

# DISTANCE_TYPES = ['sobolev-norm'] #'euclidean', 'EMD', 'moran', 'mutual-info', 'sobolev-norm'
# HEATMAP_TYPES = ['SmoothGrad'] #'SmoothGrad', 'GradCam++', 'RectGrad', 'RectGrad_PRR', 'Saliency', 'Guided_BP', 'SmoothGrad_2', 'Gradient-Input', 'IntegGrad', 'Epsilon_LRP'

# def f_beta_score(precision, recall, beta=3):
#     numerator = (1 + beta ** 2) * (precision * recall)
#     denominator = (beta ** 2 * precision) + recall
#     f_beta_score = numerator / denominator
#     return f_beta_score

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


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def fix_escape_sequences(img_addr):
    if "\\\\" in img_addr:
        img_addr = img_addr.replace("\\\\", "/")
    elif "\\" in img_addr:
        img_addr = img_addr.replace("\\", "/")
    elif "\\a" in img_addr:
        img_addr = img_addr.replace("\\a", "/a")
    elif "\\t" in img_addr:
        img_addr = img_addr.replace("\\t", "/t")
    elif "\\n" in img_addr:
        img_addr = img_addr.replace("\\n", "")
    elif "\\b" in img_addr:
        img_addr = img_addr.replace("\\b", "")
    return img_addr


run_id = 1
MAIN_CSV_PATH = r"D:\ThirdEye\ase22\simulations\track1-night-fog-100-anomalous-2\src\1\driving_log.csv"
main_data = pd.read_csv(MAIN_CSV_PATH)
center_addresses = main_data["center"]

for idx, img_addr in enumerate(tqdm(center_addresses)):
    cprintf(f'{img_addr}', 'l_red')
    img_addr = fix_escape_sequences(img_addr)


    cprintf(f'{img_addr}', 'l_yellow')
    # replace drive letter if database is copied
    if os.path.splitdrive(img_addr)[0] != os.path.splitdrive(os.getcwd())[0]:
        if '\\ThirdEye\\ase22\\' in img_addr:
            img_addr = img_addr.replace(os.path.splitdrive(img_addr)[0], os.path.splitdrive(os.getcwd())[0])
    # change the img addresses accordingly if file placement is in the newer cleaner format ({simulation_name}/src/{run_id}) 
    print(splitall(img_addr))
    for idx, addr_part in enumerate(splitall(img_addr)):
        if addr_part == 'IMG':
            addr_part = os.path.join('src', str(run_id), 'IMG')
        if idx == 0:
            corrected_address = addr_part
        else:
            corrected_address = os.path.join(prev_addr_chunk, addr_part)

        prev_addr_chunk = corrected_address

    if idx == 1:
        break

print(corrected_address)
if os.path.exists(corrected_address):
    print('true')
    