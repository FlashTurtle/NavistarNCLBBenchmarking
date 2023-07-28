import tensorflow as tf
import os
import cv2
import numpy as np
import time
import openpyxl

final_start_time = time.time()
#folder_path = 'E:\\Storage\\Code\\Navistar\\BenchmarkProject\\Log'#Edit to your path
#door_model_path = 'E:\\Storage\\Code\\Navistar\\BenchmarkProject\\doormodel.tflite' #Edit to your path

folder_path = '/home/flashturtle/Desktop/BenchmarkProject/RaspberryPi/LogTest'#Edit to your path
door_model_path = '/home/flashturtle/Desktop/BenchmarkProject/RaspberryPi/doormodel.tflite'#Edit to your path
interpreter = tf.lite.Interpreter(model_path=door_model_path)
interpreter.allocate_tensors()
# Loads model, setting interpreter based on the model path.

# Obtain input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the indices of the input and output tensors
input_tensor_index = input_details[0]['index']
output_tensor_index = output_details[0]['index']
confidence_index = output_details[0]['index']

def get_time(start_time, end_time):
    #return ("%10f" % (end_time-start_time))
    return(end_time-start_time)

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = int(milliseconds * 1000)
    return("%fm %fs %fms" %(minutes,seconds,milliseconds))




image_files = [file for file in os.listdir(folder_path) 
               if os.path.isfile(os.path.join(folder_path, file))
               ]
# Imports the files
# For all the files in the folder path, get all the names from the storage by joining them together  

# Process images to get them into standardized 320x320 for Tensor flow model
image_shape = (320,320)
procesed_image_list = []
print("\nProcessing Images")
for image in image_files:
    # Load the image
    print(image)
    image_path = os.path.join(folder_path, image)
    
    image = cv2.imread(image_path)
    processed_image = cv2.resize(image,image_shape)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    procesed_image_list.append(processed_image)
    
final_end_time = time.time()
get_time(final_start_time,final_end_time)
print("Total processing image time: %f seconds" % (get_time(final_start_time,final_end_time)))

# Main start for tensor flow
print("\nStarting TensorFlow")
count = 0
total_time = 0
image_time_list = [["Image No.","Time To Complete","Total Time","Max_Confidence"]]
for image in procesed_image_list:
    count+=1
    start_time = time.time()
    
    # Run inference
    print("Starting Image #%.0f"%(count))
    interpreter.set_tensor(input_tensor_index,image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_tensor_index)
    output_tensor = interpreter.get_tensor(confidence_index)
    #confidence = output_tensor[0].item()
    confidences = output_tensor.tolist()
    print("\n Output:",output)
    #print("\n confidences:",confidences)
    flattened_confidences = [confidence for sublist in confidences for confidence in sublist]
    #total_confidence = sum(flattened_confidences)
    #mean_confidence = sum(flattened_confidences) / len(flattened_confidences)
    max_confidence = max(flattened_confidences)
    #print (total_confidence,mean_confidence,max_confidence)
    
    end_time = time.time()
    total_time+=(end_time-start_time)
    print("Image #%.0f Complete \nTime: %f seconds\n" % (count,get_time(start_time,end_time)))
    
    image_time_list.append(["Image #%.0f"%(count),get_time(start_time,end_time),total_time,max_confidence])

# Finish with what needs to be outputted for debugging
print(image_time_list)

print("Average Time: %f seconds" % (total_time/count))    

final_end_time = time.time()
get_time(final_start_time,final_end_time)
print("\nTotal time: %f seconds" % (get_time(final_start_time,final_end_time)))

# Create a new workbook
workbook = openpyxl.Workbook()
sheet = workbook.active

# Iterate over the 2D array and input the values into Excel cells
for row_index, row_data in enumerate(image_time_list, start=1):
    for column_index, value in enumerate(row_data, start=1):
        cell = sheet.cell(row=row_index, column=column_index)
        cell.value = value

# Save the workbook to a file
#workbook.save("E:\\Storage\\Code\\Navistar\\BenchmarkProject\\"+"PCoutput.xlsx")
workbook.save("/home/flashturtle/Desktop/BenchmarkProject/RaspberryPi/"+"PIoutput.xlsx")

print("Saved to workbook!")
