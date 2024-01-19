# from ultralytics import YOLO
# import cv2
# import numpy as np
# from copy import deepcopy
# from components.proposal_box_yolo import proposal_box_yolo
# import time
# import torch
# from torch.quantization import quantize_dynamic, convert

# start_time = time.time()
# model = YOLO(r'C:\Users\TKD01A-1\Desktop\Projects\nachi_prog\PROGRAM_SOURCE\ComputerVision\train_sl-20240117T101706Z-001\train_sl\last.pt')  # load a custom model

# origin_img = cv2.imread(r"C:\Users\TKD01A-1\Downloads\31201927_01_12_16_23_35_617.jpg")
# print(origin_img.shape)
# pred_img = model(origin_img, save = False)
# print("result detect: ",len(pred_img[0].boxes.xywh))
# for element in pred_img[0].boxes.xywh:
#     if element[1] < 1450:
#         print("element: ",element)
#         cv2.rectangle(pred_img, (), (), color, thickness=2)
    

# print("time process to get number: ",time.time() - start_time)


# from ultralytics import YOLO
# import time
# # start_time = time.time()
# # Load a YOLOv8n PyTorch model
# model = YOLO(r'C:\Users\TKD01A-1\Desktop\Projects\nachi_prog\PROGRAM_SOURCE\ComputerVision\nhandienvat-main\train_sl-20240117T110559Z-001\train_sl\last.pt')

# # Export the model
# # model.export(format='openvino', imgsz=(480,640))  # creates 'yolov8n_openvino_model/'

# # Load the exported OpenVINO model
# ov_model = YOLO('yolov8n_openvino_model/')

# # Run inference
# results = model(r"C:\Users\TKD01A-1\Downloads\31201927_01_12_16_24_10_498.jpg")
# results = ov_model(r"C:\Users\TKD01A-1\Downloads\31201927_01_12_16_24_10_498.jpg",imgsz=(480, 640))

# # print("time process to get number: ",time.time() - start_time)


from ultralytics import YOLO
import time
import torch


# Load a YOLOv8n PyTorch model
model = YOLO(r'C:\Users\TKD01A-1\Desktop\Projects\nachi_prog\PROGRAM_SOURCE\ComputerVision\nhandienvat-main\train_sl-20240117T110559Z-001\train_sl\last.pt')
# Export the model
# model.export(format='openvino',half=False, int8=True,dynamic=False,data=r"C:\Users\TKD01A-1\Desktop\Projects\nachi_prog\PROGRAM_SOURCE\ComputerVision\nhandienvat-main\get_num.v1i.yolov8-20240117T124943Z-001\get_num.v1i.yolov8\data.yaml")
# Load the exported OpenVINO model
ov_model = YOLO('yolov8n_openvino_model/')
start_time = time.time()
results_openvino = ov_model(r"C:\Users\TKD01A-1\Desktop\Projects\final_img\ANH\anh sl\31201927_01_11_18_43_39_840.jpg")
print("time process to get number: ",time.time() - start_time)
# ov_model.info()

# da toi uu
for i in range(10):
    index=1
    start_time = time.time()
    # results_pytorch = model(rf"C:\Users\TKD01A-1\Desktop\Projects\final_img\ANH\anh sl\{index}.jpg")
    results_openvino = ov_model(rf"C:\Users\TKD01A-1\Desktop\Projects\final_img\ANH\anh sl\{index}.jpg")
    print("time process to get number: ",time.time() - start_time)
    index=index+1
    time.sleep(1)

# chua toi uu
# results_pytorch = model(r"C:\Users\TKD01A-1\Desktop\Projects\final_img\ANH\anh sl\31201927_01_11_20_41_30_010.jpg")

# Run inference using OpenVINO model



# from deepsparse import Pipeline

# # Specify the path to your YOLOv8 ONNX model
# model_path = "path/to/yolov8n.onnx"

# # Set up the DeepSparse Pipeline
# yolo_pipeline = Pipeline.create(
#     task="yolov8",
#     model_path=model_path
# )

# # Run the model on your images
# images = ["path/to/image.jpg"]
# pipeline_outputs = yolo_pipeline(images=images)
    
