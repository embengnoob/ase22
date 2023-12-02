from pathlib import Path

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm
import sys
sys.path.append("..")
import utils
from utils import *
from utils_models import *

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
    self_driving_car_model = tensorflow.keras.models.load_model(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))
    # model = build_model(cfg.SDC_MODEL_NAME, cfg.USE_PREDICTIVE_UNCERTAINTY)
    # self_driving_car_model = model.load_weights(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))

    # load attention model
    saliency = None
    if attention_type == "SmoothGrad":
        saliency = Saliency(self_driving_car_model, model_modifier=None)
    elif attention_type == "GradCam++":
        saliency = GradcamPlusPlus(self_driving_car_model, model_modifier=None)

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
            raise ValueError(Fore.RED + f'Unknown heatmap computation method {attention_type} given.' + Fore.RESET)
        
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

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)

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