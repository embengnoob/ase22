from pathlib import Path

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm
import sys
sys.path.append("..")
import utils
from utils import *
from utils_models import *
from deep_explain import DeepExplain

def score_when_decrease(output):
    return -1.0 * output[:, 0]


def compute_heatmap(cfg, nominal, simulation_name, NUM_OF_FRAMES, MODE, run_id, attention_type, SIM_PATH, MAIN_CSV_PATH, HEATMAP_FOLDER_PATH, HEATMAP_CSV_PATH, HEATMAP_IMG_PATH):
    """
    Given a simulation by Udacity, the script reads the corresponding image paths from the csv and creates a heatmap for
    each driving image. The heatmap is created with the SmoothGrad algorithm available from tf-keras-vis
    (https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#SmoothGrad). The scripts generates a separate
    IMG/ folder and csv file.

    """

    cprintf(f"Computing attention heatmaps for simulation \"{simulation_name}\" using \"{attention_type}\" of run id \"{run_id}\"", 'l_cyan')

    # load the image file paths from main csv
    main_data = pd.read_csv(MAIN_CSV_PATH)
    data = main_data["center"]
    print("Read %d images from file\n" % len(data))

    if len(data) != NUM_OF_FRAMES:
        raise ValueError(Fore.RED + f'Length of loaded data:{len(data)} is not the same as number of frames: {NUM_OF_FRAMES}' + Fore.RESET)

    # load self-driving car model
    cprintf(f'Loading self-driving car model: {cfg.SDC_MODEL_NAME}', 'l_yellow')
    self_driving_car_model = tf.keras.models.load_model(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))

    # load attention model
    saliency = None
    if attention_type == "SmoothGrad":
        saliency = Saliency(self_driving_car_model, model_modifier=None)
    elif attention_type == "GradCam++":
        saliency = GradcamPlusPlus(self_driving_car_model, model_modifier=None)
    attribution_method_list = [
        ('RectGrad', 'rectgrad'),
        ('RectGrad_PRR', 'rectgradprr'),
        ('Saliency', 'saliency'),
        ('Guided_BP', 'guidedbp'),
        ('SmoothGrad_2', 'smoothgrad'),
        ('Gradient*Input', 'grad*input'),
        ('IntegGrad', 'intgrad'),
        ('Epsilon_LRP', 'elrp'),
        ('DeepLIFT', 'deeplift')
    ]
    attribution_methods = collections.OrderedDict(attribution_method_list)
    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    prev_hm = gradient = np.zeros((80, 160))

    missing_heatmaps = 0

    cprintf(f'Generating heatmaps and loss scores/plots...', 'l_cyan')
    for idx, img in enumerate(tqdm(data)):
        if MODE != 'new_calc':
            # double-check if every heatmap actually exists
            img_address = img.split('/')[-1]
            img_name = os.path.basename(os.path.normpath(img_address))
            hm_name = "htm-" + attention_type.lower() + '-' + img_name
            if hm_name in os.listdir(HEATMAP_IMG_PATH):
                # store the addresses for the heatmap csv file
                heatmap_path = os.path.join(HEATMAP_IMG_PATH, hm_name)
                list_of_image_paths.append(heatmap_path)
                continue
            else:
                missing_heatmaps += 1 
        # convert Windows path, if needed
        if "\\\\" in img:
            img = img.replace("\\\\", "/")
        elif "\\" in img:
            img = img.replace("\\", "/")

        # load image
        x = mpimg.imread(img)

        # preprocess image
        x = utils.resize(x).astype('float32')

        # compute heatmap image
        saliency_map = None
        if attention_type == "SmoothGrad":
            saliency_map = saliency(score_when_decrease, x, smooth_samples=20, smooth_noise=0.20)
        elif attention_type == "GradCam++":
            saliency_map = saliency(score_when_decrease, x, penultimate_layer=-1)
        else:
            attributions_orig = collections.OrderedDict()
            attributions_sparse = collections.OrderedDict()
            if attention_type not in attribution_methods:
                raise ValueError(Fore.RED + f'Unknown heatmap computation method {attention_type} given.' + Fore.RESET)
            else:
                xs = x
                xs = xs[np.newaxis is None,:,:,:]
                xs_tensor = tensor(xs)

                T = self_driving_car_model(xs_tensor)

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(xs_tensor)
                    y_pred = self_driving_car_model(xs_tensor)

                # DeepExplain initialization
                with DeepExplain() as de:
                    for k, v in attribution_methods.items():

                        # Explanation using DeepExplain
                        attribution = de.explain(v, T, xs, xs_tensor, y_pred, tape, self_driving_car_model)
                        
                        # Preprocessing based on the method used
                        if 'RectGrad' in k:
                            attributions_orig[k] = preprocess(attribution, 0.5, 99.5)
                            saliency_map = attributions_orig[k]
                            if cfg.SPARSE_ATTRIBUTION:
                                attributions_sparse[k] = preprocess(attribution, 0.5, 99.5)
                                saliency_map = attributions_sparse[k]
                        else:
                            attributions_orig[k] = preprocess(attribution, 0.5, 99.5)
                            saliency_map = attributions_orig[k]
                            if cfg.SPARSE_ATTRIBUTION:
                                attributions_sparse[k] = preprocess(attribution, 95, 99.5)
                                saliency_map = attributions_sparse[k]


        if cfg.METHOD == 'thirdeye':
            # compute average of the heatmap
            average = np.average(saliency_map)

            # compute gradient of the heatmap
            if idx == 0:
                gradient = 0
            else:
                gradient = abs(prev_hm - saliency_map)
            average_gradient = np.average(gradient)
            prev_hm = saliency_map

        # store the heatmaps
        file_name = img.split('/')[-1]
        file_name = "htm-" + attention_type.lower() + '-' + file_name
        path_name = os.path.join(HEATMAP_IMG_PATH, file_name)
        mpimg.imsave(path_name, np.squeeze(saliency_map))

        list_of_image_paths.append(path_name)

        if cfg.METHOD == 'thirdeye':
            avg_heatmaps.append(average)
            avg_gradient_heatmaps.append(average_gradient)

    if cfg.METHOD == 'thirdeye':
        # score and their plot paths
        if not nominal:
            SCORES_FOLDER_PATH = os.path.join(SIM_PATH, run_id)
            if not os.path.exists(SCORES_FOLDER_PATH):
                cprintf(f'Loss avg/avg-grad scores folder does not exist. Creating folder ...' ,'l_blue')
                os.makedirs(SCORES_FOLDER_PATH)
        else:
            SCORES_FOLDER_PATH = SIM_PATH

        cprintf(f'Saving loss avg/avg-grad scores and their plots to {SCORES_FOLDER_PATH}' ,'l_yellow')
        file_name = "htm-" + attention_type.lower() + '-scores'
        AVG_SCORE_PATH = os.path.join(SCORES_FOLDER_PATH, file_name + '-avg')
        AVG_PLOT_PATH = os.path.join(SCORES_FOLDER_PATH, 'plot-' + file_name + '-avg.png')
        AVG_GRAD_SCORE_PATH = os.path.join(SCORES_FOLDER_PATH, file_name + '-avg-grad')
        AVG_GRAD_PLOT_PATH = os.path.join(SCORES_FOLDER_PATH, 'plot-' + file_name + '-avg-grad.png')

    if MODE == 'new_calc':
        if cfg.METHOD == 'thirdeye':
            # save scores as numpy arrays
            np.save(AVG_SCORE_PATH, avg_heatmaps)
            # plot scores as histograms
            plt.hist(avg_heatmaps)
            plt.title("average attention heatmaps")
            plt.savefig(AVG_PLOT_PATH)
            np.save(AVG_GRAD_SCORE_PATH, avg_gradient_heatmaps)
            plt.clf()
            plt.hist(avg_gradient_heatmaps)
            plt.title("average gradient attention heatmaps")
            plt.savefig(AVG_GRAD_PLOT_PATH)
            #plt.show()
    else:
        if missing_heatmaps > 0:
            # remove hm folder
            shutil.rmtree(HEATMAP_FOLDER_PATH)
            if cfg.METHOD == 'thirdeye':
                # remove scores and plots
                os.remove(AVG_SCORE_PATH)
                os.remove(AVG_PLOT_PATH)
                os.remove(AVG_GRAD_SCORE_PATH)
                os.remove(AVG_GRAD_PLOT_PATH)
            raise ValueError("Error in number of frames of aved heatmaps. Removing all the heatmaps! Please rerun the code.")

    # save as csv in heatmap folder
    try:
        # autonomous mode simulation data
        data = main_data[["frameId", "time", "crashed", "cte", "speed", "car_position", "steering_angle", "throttle"]]
    except:
        # manual mode simulation data
        data = main_data[["frameId", "cte", "speed", "car_position", "steeringAngle", "throttle"]]

    # copy frame id, simulation time and crashed information from simulation's csv
    # if 'frameId' in data.columns:
    df = pd.DataFrame(data['frameId'].copy(), columns=['frameId'])
    # else:
    #     df = pd.DataFrame(data.index.copy(), columns=['frameId'])
    df_img = pd.DataFrame(list_of_image_paths, columns=['center'])
    df['center'] = df_img['center'].copy()
    if 'time' in data.columns:
        df['time'] = data['time'].copy()
    if 'crashed' in data.columns:
        df['crashed'] = data['crashed'].copy()
    df['cte'] = data['cte'].copy()
    df['speed'] = data['speed'].copy()
    df['position'] = data['car_position'].copy()
    df['throttle'] = data['throttle'].copy()
    try:
        df['steering_angle'] = data['steering_angle'].copy()
    except:
        df['steering_angle'] = data['steeringAngle'].copy()
    
    # save it as a separate csv
    cprintf(f'Saving CSV file to: {HEATMAP_CSV_PATH}', 'l_yellow')
    df.to_csv(HEATMAP_CSV_PATH, index=False)




    


# # filenames_valid = []
# # images = []
# for idx, img_adr in enumerate(tqdm(data)):
#     if idx != 0:
#         break
#     # convert Windows path, if needed
#     if "\\\\" in img_adr:
#         img_adr = img_adr.replace("\\\\", "/")
#     elif "\\" in img_adr:
#         img_adr = img_adr.replace("\\", "/")

#     # load image
#     xs = mpimg.imread(img_adr)

#     # preprocess image
#     xs = utils.resize(xs).astype('float32')
#     xs = xs[np.newaxis is None,:,:,:]
#     xs_tensor = tensor(xs)

#     T = self_driving_car_model(xs_tensor)

#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(xs_tensor)
#         y_pred = self_driving_car_model(xs_tensor)

#     # gradients = tape.gradient(y_pred, xs_tensor)

#     # attribution_methods = [
#     #     ('RectGrad', 'rectgrad'),
#     #     ('RectGrad PRR', 'rectgradprr'),
#     #     ('Saliency Map', 'saliency'),
#     #     ('Guided BP', 'guidedbp'),
#     #     ('SmoothGrad', 'smoothgrad'),
#     #     ('Gradient * Input', 'grad*input'),
#     #     ('IntegGrad', 'intgrad'),
#     #     ('Epsilon-LRP', 'elrp'),
#     #     ('DeepLIFT', 'deeplift')
#     # ]
#     attribution_methods = [
#         ('RectGrad', 'rectgrad'),
#         ('RectGrad_PRR', 'rectgradprr'),
#         ('Saliency Map', 'saliency'),
#         ('Guided_BP', 'guidedbp'),
#         ('SmoothGrad', 'smoothgrad'),
#         ('Gradient * Input', 'grad*input'),
#         ('IntegGrad', 'intgrad'),
#         ('Epsilon_LRP', 'elrp')
#     ]

#     attribution_methods = collections.OrderedDict(attribution_methods)

#     attributions_orig = collections.OrderedDict()
#     attributions_sparse = collections.OrderedDict()

#     # DeepExplain initialization
#     with DeepExplain() as de:
#         for k, v in attribution_methods.items():
#             cprintf(f'Running {k} explanation method', 'l_cyan')
            
#             # Explanation using DeepExplain
#             attribution = de.explain(v, T, xs, xs_tensor, y_pred, tape, self_driving_car_model)
            
#             # Preprocessing based on the method used
#             if 'RectGrad' in k:
#                 attributions_orig[k] = preprocess(attribution, 0.5, 99.5)
#                 attributions_sparse[k] = preprocess(attribution, 0.5, 99.5)
#             else:
#                 attributions_orig[k] = preprocess(attribution, 0.5, 99.5)
#                 attributions_sparse[k] = preprocess(attribution, 95, 99.5)

#     print('Done!')

#     # print(type(attributions_orig))
#     # print(type(attributions_sparse))

#     ### Attribution Map Comparison W/O Baseline Final Thresholding

# for i, xi in enumerate(xs):
#     if i != 0:
#         break
#     plt.figure(figsize=(13,6))
    
#     xi = (xi - np.min(xi))
#     xi /= np.max(xi)
    
#     plt.subplot(2, 5, 1)
#     plt.imshow(xi)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.title(label_map[labels[i] - 1].split(',')[0].capitalize(), fontsize=20, pad=10)
#     # 2023_11_07_12_39_56_257_FID_0.jpg
#     plt.title('nominal_FID_0.jpg')
    
#     for j, a in enumerate(attribution_methods):
        
#         plt.subplot(2, 5, j + 2)
#         v, cmap = pixel_range(attributions_orig[a][i])
#         plt.imshow(attributions_orig[a][i], vmin=v[0], vmax=v[1], cmap=cmap)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(a, fontsize=20, pad=10)
    
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.05, hspace=0.2)

#     png_folder = os.path.join('png_results', '5')
#     png_file = os.path.join(png_folder, f'5_Attribution_Map_NO_Baseline_final_thresholding{i}.png')
#     if not os.path.exists(png_folder):
#         os.makedirs(png_folder)  
#     plt.savefig(png_file, bbox_inches='tight', dpi=300)

#     plt.show()    
    
#     plt.close()

# ### Attribution Map Comparison with Baseline Final Thresholding

# for i, xi in enumerate(xs):
#     if i != 0:
#         break    
#     plt.figure(figsize=(13,6))
    
#     xi = (xi - np.min(xi))
#     xi /= np.max(xi)
    
#     plt.subplot(2, 5, 1)
#     plt.imshow(xi)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.title(label_map[labels[i] - 1].split(',')[0].capitalize(), fontsize=20, pad=10)
#     # 2023_11_07_12_39_56_257_FID_0.jpg
#     plt.title('nominal_FID_0.jpg')

#     for j, a in enumerate(attribution_methods):
        
#         plt.subplot(2, 5, j + 2)
#         v, cmap = pixel_range(attributions_sparse[a][i])
#         plt.imshow(attributions_sparse[a][i], vmin=v[0], vmax=v[1], cmap=cmap)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(a, fontsize=20, pad=10)
    
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.05, hspace=0.2)

#     results_folder = os.path.join(c_dir, 'png_results')
#     png_folder = os.path.join(results_folder, '5')
#     png_file = os.path.join(png_folder, f'5_Attribution_Map_WITH_Baseline_final_thresholding{i}.png')
#     if not os.path.exists(results_folder):
#         os.mkdir(results_folder)
#     if not os.path.exists(png_folder):
#         os.mkdir(png_folder)  
#     plt.savefig(png_file, bbox_inches='tight', dpi=300)

#     plt.show()    
    
#     plt.close()
