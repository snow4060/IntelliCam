import cv2
import os
import sys
import select
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import pytesseract
import speech_recognition as sr
import pyttsx3

def init():
    global paths, files, category_index, configs
    CUSTOM_MODEL_NAME = "objects" 
    PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
    PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
    TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
    LABEL_MAP_NAME = "label_map.pbtxt"
    paths = {
        "WORKSPACE_PATH": os.path.join("object_detection", "Tensorflow", "workspace"),
        "SCRIPTS_PATH": os.path.join("object_detection", "Tensorflow","scripts"),
        "APIMODEL_PATH": os.path.join("object_detection", "Tensorflow","models"),
        "ANNOTATION_PATH": os.path.join("object_detection", "Tensorflow", "workspace","annotations"),
        "IMAGE_PATH": os.path.join("object_detection", "Tensorflow", "workspace","images"),
        "MODEL_PATH": os.path.join("object_detection", "Tensorflow", "workspace","models"),
        "PRETRAINED_MODEL_PATH": os.path.join("object_detection", "Tensorflow", "workspace","pre-trained-models"),
        "CHECKPOINT_PATH": os.path.join("object_detection", "Tensorflow", "workspace","models",CUSTOM_MODEL_NAME), 
        "OUTPUT_PATH": os.path.join("object_detection", "Tensorflow", "workspace","models",CUSTOM_MODEL_NAME, "export"), 
        "TFJS_PATH":os.path.join("object_detection", "Tensorflow", "workspace","models",CUSTOM_MODEL_NAME, "tfjsexport"), 
        "TFLITE_PATH":os.path.join("object_detection", "Tensorflow", "workspace","models",CUSTOM_MODEL_NAME, "tfliteexport"), 
        "PROTOC_PATH":os.path.join("object_detection", "Tensorflow","protoc")
    }
    files = {
        "PIPELINE_CONFIG":os.path.join("object_detection", "Tensorflow", "workspace","models", CUSTOM_MODEL_NAME, "pipeline.config"),
        "TF_RECORD_SCRIPT": os.path.join(paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME), 
        "LABELMAP": os.path.join(paths["ANNOTATION_PATH"], LABEL_MAP_NAME)
    }
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    configs = r"--psm 11 --oem 3"

    
def load_model():
    global detection_model
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-7')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def annotate(frame):
    #object detection: 
    frame_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame_np, 0), dtype=tf.float32) 
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = frame_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    
    #text detection
    height, width, _ = frame.shape
    boxes = pytesseract.image_to_boxes(frame, config=configs)
    for box in boxes.splitlines():
        box = box.split(" ")
        image_np_with_detections = cv2.rectangle(image_np_with_detections, (int(box[1]), height-int(box[2])), (int(box[3]), height-int(box[4])), (255, 0, 0))
    
    return (image_np_with_detections, detections["detection_boxes"][0])

def get_direction(center):
    #center
    if center[0] < 0.6 and center[0] > 0.4 and center[1] < 0.6 and center[1] > 0.4:
        return "middle"
    #down
    if center[0] < 0.6 and center[0] > 0.4 and center[1] > 0.6:
        return "bottom"
    #up
    if center[0] < 0.6 and center[0] > 0.4 and center[1] < 0.4:
        return "top"
    #left
    if center[1] < 0.6 and center[1] > 0.4 and center[0] < 0.4:
        return "left"
    #right
    if center[1] < 0.6 and center[1] > 0.4 and center[0] > 0.6:
        return "right"
    
    
    #topleft
    if center[0] < 0.4 and center[1] < 0.4:
        return "top left"
    #top right
    if center[0] > 0.6 and center[1] < 0.4:
        return "top right"
    #bottom left
    if center[0] < 0.4 and center[1] > 0.6:
        return "bottom left"
    #bottom right
    if center[0] > 0.6 and center[1] > 0.6:
        return "bottom right"
    
    
def speak_text(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def run():
    k = None
    video = cv2.VideoCapture(2)
    
    while True: 
        ret, frame = video.read()
        user_input = ""
        annotated = annotate(frame)

        
        #take user input without waiting
        if select.select([sys.stdin], [], [], 0)[0]:
            user_input = sys.stdin.readline().strip()
            
        #no interactions, continue
        if not user_input:
            cv2.imshow("image", annotated[0])
            k = cv2.waitKey(1) & 0xFF
            continue
        
        
        if "ball" in user_input:
            center = (annotated[1][3]-annotated[1][1], annotated[1][2]-annotated[1][0])
            message = "the ball is to the " + get_direction(center)
            print(message)
            speak_text(message)
        elif "mouse" in user_input:
            center = (annotated[1][3]-annotated[1][1], annotated[1][2]-annotated[1][0])
            message = "the mouse is to the " + get_direction(center)
            print(message)
            speak_text(message)
        elif "marker" in user_input:
            center = (annotated[1][3]-annotated[1][1], annotated[1][2]-annotated[1][0])
            message = "the marker is to the " + get_direction(center)
            print(message)
            speak_text(message)
        elif "charger" in user_input:
            center = (annotated[1][3]-annotated[1][1], annotated[1][2]-annotated[1][0])
            message = "the charger is to the " + get_direction(center)
            print(message)
            speak_text(message)
        elif "text" in user_input:
            text = pytesseract.image_to_string(frame, config=configs)
            print(text)
            speak_text(text)
            
        
        elif "exit" in user_input or k == 27:
            video.release()
            cv2.destroyAllWindows()
            break
        
def conditions():
    import tkinter as tk
    window = tk.Tk()
    window.title("Privacy Statement")
    #information is not collected or saved
    label = tk.Label(window, text="Intellicam will not save or collect any of the camera footage or interactions either locally or on the cloud. All interactions are local to this client and deleted after runtime unless selected otherwise by the user.")
    label.pack(padx=20, pady=20)
    
    button = tk.Button(window, text="Accept", command=window.destroy)
    button.pack(side="bottom", padx=20, pady=20)
    
    window.mainloop()
    
conditions()

init()
load_model()    
run()