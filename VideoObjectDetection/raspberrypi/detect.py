# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import gpiozero as gz

import openpyxl


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """
  fps_time_list = [["FPS", "Latency (ms)","Time","Temperature"]]

  # Variables to calculate FPS and Latency
  counter, fps, latency, end_time, cpu_temp= 0, 0, 0, 0,0

  start_time, initial_start, prev_end_time = time.time(), time.time(), time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 1

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=1, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  
  #my_detection_result = processor.DetectionResult

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    ''' Extract the first confidence score (probability) for each detected object
    #confidence_scores = [detection.score[0] for detection in detection_result]
    
    #Extract the first confidence score for the FIRST detected object (most probabilty)
    #confidence_score = my_detection_result.detection.categories[0]
    
    #confidence_scores = detection_result.get_scores()
    #confidence_score = confidence_scores[0]

    #    num_detections = detection_result.num_detections
    #   if num_detections > 0:
    #      confidence_score = detection_result.scores[0]
    #    else:
    #      confidence_score = None
    #detection = detection_result.detections[0]
    #confidence_score = detection.scores[0].numpy()
    '''

    # Calculate the FPS
    #if counter % fps_avg_frame_count == 0:
    #Skips the first iteration to set everything up
    if counter != 1:
        end_time = time.time()
        total_time = end_time - start_time  # Calculate the total time elapsed
        fps = fps_avg_frame_count / total_time
        start_time = time.time()  # Update the start time for the next FPS calculation
    
    # Calculate the latency in IF loop
        latency = (end_time - prev_end_time)*1000
        prev_end_time = end_time
        
    # Get temp
        cpu_temp = gz.CPUTemperature().temperature

      
    #Append time list
    #fps_time_list.append([fps,latency,confidence_scores,end_time])
        if fps != 0 or latency != 0 or end_time != 0:
                fps_time_list.append([fps, latency, (end_time-initial_start),cpu_temp])



    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    fps_text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, fps_text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    
    # Show the Latency
    # Draw the latency on the image
    latency_text = 'Latency: {:.2f} ms'.format(latency)
    latency_text_location = (left_margin, height-row_size)
    cv2.putText(image, latency_text, latency_text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)


    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  

  output_file = 'fps_results.xlsx'  # Specify the filename for the Excel file
  workbook = openpyxl.Workbook()
  sheet = workbook.active
  
  for row_index, row_data in enumerate(fps_time_list, start=1):
      for column_index, value in enumerate(row_data, start=1):
          cell = sheet.cell(row=row_index, column=column_index)
          cell.value = value

  # Save the workbook to the Excel file
  workbook.save(output_file)
 
  cap.release()
  cv2.destroyAllWindows()




def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='doormodel.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
