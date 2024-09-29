from sanic import Sanic
from sanic.response import html
import asyncio
import pathlib
import pandas as pd
from src.pipeline.pipeline_components.data_loader import DataLoader
from src.pipeline.pipeline_components.preprocessor import Preprocessor
from src.pipeline.pipeline_components.dynamic_samples_builder import DynamicSamplesBuilder
from src.pipeline.pipeline_components.classifier import Classifier
from src.pipeline.pipeline_components.gesture_fire_controller import GestureFireController
from src.pipeline.pipeline import Pipeline
from src.machine_learning_framework.TrainingModel import *

slideshow_root_path = pathlib.Path(__file__).parent.joinpath("util/slideshow/slideshow")

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)


#with open(os.path.join(os.getcwd(), 'resource/models/gesicherte_gewichte_TestModel.pkl'), 'rb') as file:
#    loaded_list = pickle.load(file)
#
#loaded_weights, loaded_bias = loaded_list
## create model
#layer_1 = ClassicLayer(421, 20, loaded_weights[0], loaded_bias[0])
#layer_2 = ClassicLayer(20, 20, loaded_weights[1], loaded_bias[1])
#layer_3 = ClassicLayer(20, 20, loaded_weights[2], loaded_bias[2])
#layer_4 = SoftmaxLayer(20, 4, loaded_weights[3], loaded_bias[3])
#
#model = NeuronalNetworkModel([layer_1, layer_2, layer_3, layer_4])
#
#preprocessor = Preprocessor(False)
#dynamic_sample_builder = DynamicSamplesBuilder()
#classifier = Classifier(model)
#gesture_fire_controller = GestureFireController()
#
#pipeline = Pipeline(preprocessor, dynamic_sample_builder, classifier, gesture_fire_controller)
pipeline = Pipeline(mode="live_mandatory")



@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow.html"), "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    ##################################### insert out pipeline ##########################################################

    ###################################### copied from live video feed #################################################

    import cv2
    import mediapipe as mp
    import yaml
    import pathlib

    # This script uses mediapipe to parse videos to extract coordinates of
    # the user's joints. You find documentation about mediapipe here:
    #  https://google.github.io/mediapipe/solutions/pose.html

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    script_dir = pathlib.Path(__file__).parent

    # ===========================================================
    # ======================= SETTINGS ==========================
    show_video = True
    show_data = True
    flip_image = True  # when your webcam flips your image, you may need to re-flip it by setting this to True

    #cap = cv2.VideoCapture(filename=str(script_dir.joinpath("../resource/misc/video_rotate.mp4")))  # Video
    #cap = cv2.VideoCapture(filename=str(script_dir.joinpath("../resource/misc/video_test_all_julius.mp4")))  # Video
    cap = cv2.VideoCapture(index=0) # Live from camera (change index if you have more than one camera)

    # lower framerate to 30 fps -> working and necessary???
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ===========================================================

    # the names of each joint ("keypoint") are defined in this yaml file:
    with open(script_dir.joinpath("util/process_videos/keypoint_mapping.yml"), "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        KEYPOINT_NAMES = mappings["face"]
        KEYPOINT_NAMES += mappings["body"]

    print(KEYPOINT_NAMES)

    success = True
    # find parameters for Pose here: https://google.github.io/mediapipe/solutions/pose.html#solution-apis
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and success:
            success, image = cap.read()
            if not success:
                break

            if flip_image:
                image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image
            if show_video:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imshow('MediaPipe Pose', image)

            # press ESC to stop the loop
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # =================================
            # ===== read and process data =====
            df_data = {}
            if show_data and results.pose_landmarks is not None:
                result = f"timestamp: {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.1f} seconds\n"
                df_data['timestamp'] = cap.get(cv2.CAP_PROP_POS_MSEC)

                for joint_name in KEYPOINT_NAMES:  # you can choose any joint listed in `KEYPOINT_NAMES`
                    joint_data = results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)]
                    df_data[f'{joint_name}_x'] = joint_data.x
                    df_data[f'{joint_name}_y'] = joint_data.y
                    df_data[f'{joint_name}_z'] = joint_data.z
                    # visibility = confidence?
                    df_data[f'{joint_name}_confidence'] = joint_data.visibility

                    #for joint_name in ["nose", "right_wrist", "left_wrist"]:  # you can choose any joint listed in `KEYPOINT_NAMES`
                    #joint_data = results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)]
                    #result += f"   {joint_name:<12s} > (x: {joint_data.x:.2f}, y: {joint_data.y:.2f}, z: {joint_data.z:.2f}) [{joint_data.visibility * 100:3.0f}% visible]\n"
                #print(result)

                frame = pd.DataFrame(df_data, index=[0])
                gesture = pipeline.run(frame)
                match gesture:
                    case "idle":
                        pass
                    case "swipe_left":
                        await ws.send("right")
                        await asyncio.sleep(0.05)
                    case "swipe_right":
                        await ws.send("left")
                        await asyncio.sleep(0.05)
                    case "rotate_right":
                        await ws.send("rotate")
                        await asyncio.sleep(0.05)
                    case "rotate_left":
                        await ws.send("rotate_left")
                        await asyncio.sleep(0.05)
                    case "flip_table":
                        await ws.send("rotate_reset")
                        await asyncio.sleep(0.05)
                    case "spread":
                        await ws.send("zoom_in")
                        await asyncio.sleep(0.05)
                    case "pinch":
                        await ws.send("zoom_out")
                        await asyncio.sleep(0.05)
                    case _:
                        print("Unknown gesture")
                        await asyncio.sleep(0.05)
            # ==================================
    cap.release()

#@app.websocket("/events")
#async def emitter(_request, ws):
#    print("websocket connection opened")
#
#    # ======================== add calls to your model here ======================
#    # uncomment for event emitting demo: the following loop will alternate
#    # emitting events and pausing
#    #
#    while True:
#        print("emitting 'right'")
#        # app.add_signal(event="right")
#        await ws.send("right")
#        await asyncio.sleep(2)
#    #
#        print("emitting 'rotate'")
#        await ws.send("rotate")
#        await asyncio.sleep(2)
#    #
#        print("emitting 'left'")
#        await ws.send("left")
#        await asyncio.sleep(2)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
