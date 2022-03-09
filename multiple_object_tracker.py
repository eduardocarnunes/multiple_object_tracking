# import libraries
import time
from datetime import datetime
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from absl.flags import FLAGS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import util
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# load params from .yml
print('[INFO]: Load params from config.yml')
yaml_content = util.yaml_load()

# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0

print('[INFO]: Initialize Deep Sort')
# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

print('[INFO]: Load configuration for object detector')
# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config('yolov4')
input_size = yaml_content['input_size_yolo']

print('[INFO]: Load model and weights YOLOv4')
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# begin video capture
if yaml_content['is_mode_camera'] == True:
    print('[INFO]: Mode Camera ON. Camera Real Time!')
    vid = cv2.VideoCapture(yaml_content['id_camera'])
else:
    print('[INFO]: Modelo Video File ON!')
    video_path = yaml_content['video_path_test']
    vid = cv2.VideoCapture(video_path)
    
    
# set camera resolution     
if yaml_content['set_resolution'] == 'make_1080p':
    vid.set(3, 1920)
    vid.set(4, 1080)
elif yaml_content['set_resolution'] == 'make_720p':
    vid.set(3, 1280)
    vid.set(4, 720)
elif yaml_content['set_resolution'] == 'make_480p':
    vid.set(3, 640)
    vid.set(4, 480)
elif yaml_content['set_resolution'] == 'custom':
    vid.set(3, yaml_content['resolution_custom_width'])
    vid.set(4, yaml_content['resolution_custom_height'])
   


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# get video ready to save locally if flag is set
if yaml_content['is_save_result_video'] == True:
    print('[INFO]: Settings params video output')
    out = None
    # by default VideoCapture returns float instead of int    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(vid.get(cv2.CAP_PROP_FPS))
    fps = yaml_content['fps_save_result']
    codec = cv2.VideoWriter_fourcc(*'MPEG')
    video_result_path = yaml_content['video_path_result']
    out = cv2.VideoWriter(video_result_path, codec, fps, (width, height))


# need this for show points for object tracking
from _collections import deque
pts = [deque(maxlen=20) for _ in range(800)]

# load allowed_classes
allowed_classes = yaml_content['allowed_classes'] 
print('[INFO]: Allowed Classes: ' + str(allowed_classes))

# create dataframe for result in .csv
if yaml_content['is_save_result_csv'] == True:    
    df_result = pd.DataFrame(columns=['Time', 'Tracker_ID', 'Class_Name', 'Box_Coords', 'Center'])

frame_num = 0

# while video is running
while True:
    # read frame
    return_value, frame = vid.read()
    
    # image conversions
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        print('[INFO]: Video has ended or failed, try a different video format!')
        break
    
    frame_num +=1
    
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)    
    
    start_time = time.time()
    
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class = 50,
        max_total_size = 50,
        iou_threshold = 0.45,
        score_threshold = 0.5
    )

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)    

    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)
    
    if yaml_content['is_show_count_object'] == True:
        cv2.putText(frame, "Objects being tracked: {}".format(count), (0, 35), 0, 1, (0, 0, 255), 2)            
        print("Objects being tracked: {}".format(count))
        
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]       

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        class_name = track.get_class()
        
        # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)        
        
        # get object center
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)
        
        # draw center points
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(frame, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)        

        # if enable info flag then print details about each track
        if yaml_content['is_show_result_cmd'] == True:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            print('Center: ' + str(center))
            
        if yaml_content['is_save_result_csv'] == True:
            time_now_result = datetime.now()
            tracker_id_result = str(track.track_id)
            class_name_result = class_name
            box_coords = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            center_result = str(center)
            
            new_row = {'Time' : time_now_result, 
                        'Tracker_ID' : tracker_id_result, 
                        'Class_Name' : class_name_result,
                        'Box_Coords' : box_coords,
                        'Center' : center_result}
            
            df_result = df_result.append(new_row, ignore_index=True)

    # calculate frames per second of running detections
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (0,65), 0, 1, (0,0,255), 2)
    print("FPS: %.2f" % fps)
    result = np.asarray(frame)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # preview result
    if yaml_content['is_preview_result'] == True:
        cv2.imshow("Output Video", result)
    
    # if output flag is set, save video file
    if yaml_content['is_save_result_video'] == True:
        out.write(result)
    # exit        
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()

# save result video
if yaml_content['is_save_result_video'] == True:
    print('[INFO]: Result save in ' + str(video_result_path))
    out.release()

# close preview object    
if yaml_content['is_preview_result'] == True:
    vid.release()
    
# save to .csv
if yaml_content['is_save_result_csv'] == True:
    df_result.to_csv(yaml_content['path_file_csv'], index=False)
    print('[INFO]: Result .csv save in' + yaml_content['path_file_csv'])    
