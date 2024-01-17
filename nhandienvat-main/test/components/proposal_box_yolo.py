import numpy as np
from ultis.apply_min_area import apply_min_area
from ultis.extractCSV import extractCSV
from ultralytics import YOLO
import cv2
import time

class YOLODetect:
    def __init__(self,model):
        self.model = YOLO(model)
    def predict(self,img):
        pred_img = self.model(img)
        print("length: ",len(pred_img[0].boxes.xywh))
        length = len(pred_img[0].boxes.xywh)
        return length


class YOLOSegmentation:
    def __init__(self,model):
        self.model =YOLO(model)  # load a custom model
    def predict(self,img,img_size,configScore):

        #Thực hiện nhận diện
        pred_img = self.model(img, save=False, conf=configScore, imgsz=img_size)
        #lấy danh sách các bounding box
        bboxes = np.array(pred_img[0].boxes.xyxy, dtype="int")
        len_obj=len(bboxes)
        # lấy danh sách mặt nạ masks của mỗi đối tượng đcược nhận diện
        masks = np.array(pred_img[0].masks.xy,dtype=object)
        # lấy danh sách các loại đối tượng đã được nhận diện
        class_ids = np.array(pred_img[0].boxes.cls,dtype="int")
        # print(class_ids)
        # lấy danh sách độ chính xác của các đối tượng đã được nhận diện
        score = np.array(pred_img[0].boxes.conf,dtype="float").round(2)
        return bboxes, masks, class_ids, score,len_obj
    
    @staticmethod
    def filter_boxes(bboxes,masks,class_ids,score):
        # chỉ lấy các đối tượng với chỉ số là 3(mat_dung)
        obj_detect = class_ids == 0
        # object thông tin bounding box, masks, score của các dối tượng đúng
        object_true = bboxes[obj_detect,:], masks[obj_detect], score[obj_detect]
        # print(object_true)
        # object thông tin bounding box, masks, score của các dối tượng sai
        object_fail = bboxes[~obj_detect,:],masks[~obj_detect],score[~obj_detect]
        # print("detect: ",object_true)
        return object_true,object_fail

    @staticmethod
    def create_angle(mask_true):  
         #truy cập vào từng điểm của mask và tính toán
         angle = list(map(lambda x: apply_min_area(x), mask_true))
         return angle
    @staticmethod
    def convert_xywh(boxes):
    #    tạo thông tin bounding box từ dang xyxy chuyển sang dạng xywh(xy ở đây là điểm x_min y_min chứ ko phải là center)
        boxes[:, 2], boxes[:, 3] = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        return boxes
    
        

def proposal_box_yolo(img,model,image_size,configScore,outputImgLink,csvLink):
    
    ys = YOLOSegmentation(model)
    try:
        bboxes,masks,class_ids, score,len_obj = ys.predict(img,image_size, configScore)
        obj,_ = ys.filter_boxes(bboxes,masks,class_ids,score)
        # tính toán góc xoay dựa vào các điểm masks(obj[1] là list các điểm masks đúng)
        angle_test = ys.create_angle(obj[1])
        xywh_boxes = ys.convert_xywh(obj[0])
        number_items = np.full(len(xywh_boxes),len(obj[0]),dtype=float)
        return len_obj,list(zip(angle_test, xywh_boxes,number_items))
    except Exception as e:
        print("Yolo detection error or no detection: ",e)
        result = [[0,0,0,0]]
        score = []
        outputImgLink = outputImgLink + '/output.jpg'
        cv2.imwrite(outputImgLink,img)
        extractCSV(csvLink,result,score)

# def proposal_box_yolo_get_total(img,model,image_size,configScore,outputImgLink):
#     ys = YOLOSegmentation(model)
#     try:
#         _,_,_, _,len_obj = ys.predict(img,image_size, configScore)
#         # obj,_ = ys.filter_boxes(bboxes,masks,class_ids,score)
#         # angle_test = ys.create_angle(obj[1])
#         # xywh_boxes = ys.convert_xywh(obj[0])
#         # number_items = np.full(len(xywh_boxes),len(obj[0]),dtype=float)
        
#         return len_obj
#     except Exception as e:
#         print("Yolo detection error or no detection: ",e)
#         outputImgLink = outputImgLink + '/output.jpg'
#         cv2.imwrite(outputImgLink,img)    
#         # return []

    

def proposal_box_yolo_get_total(img,model,image_size,configScore,outputImgLink):
    try:
        start_time = time.time()
        ys = YOLODetect(model)
        len_obj = ys.predict(img)
        print("time process: ",time.time() - start_time)
        return len_obj
        
    except Exception as e:
        print("Yolo detection error or no detection: ",e)
        outputImgLink = outputImgLink + '/output.jpg'
        cv2.imwrite(outputImgLink,img)    
        # return []
    