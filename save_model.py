import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

import util
print('[INFO]: Load params from config.yml')
yaml_content = util.yaml_load()


# path to weights file
yolov4_weights_path = './data/weights/yolov4.weights'
# path to output
output_weights = './checkpoints/yolov4-416'
# define input size of export model
input_size = yaml_content['input_size_yolo']
# define score threshold
score_thres = 0.2


def save_tf():
  """save model. (darknet to tensorflow)
  """
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([input_size, input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, 'yolov4', False)
  
  bbox_tensors = []
  prob_tensors = []
    
  for i, fm in enumerate(feature_maps):
    if i == 0:
      output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tf')
    elif i == 1:
      output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tf')
    else:
      output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tf')
    bbox_tensors.append(output_tensors[0])
    prob_tensors.append(output_tensors[1])
    
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  
  boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_thres, input_shape=tf.constant([input_size, input_size]))
  pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  utils.load_weights(model, yolov4_weights_path, 'yolov4', False)  
  model.summary()
  print('[INFO]: Save in ' + str(output_weights))
  model.save(output_weights)


if __name__ == '__main__':
  save_tf()   
  print('[INFO]: Finished')