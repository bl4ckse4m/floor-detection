from pathlib import Path

import torch
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 # OpenCV
import easyocr
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from skimage import color, data, restoration
from scipy.signal import convolve2d
from ultralytics import YOLO

model = YOLO('weights/best_l.pt')
def model_predict(uploaded_image):
    #uploaded_image = Image.open(uploaded_file)
    res = model.predict(uploaded_image, agnostic_nms = True, augment = False, iou = 0.7, save = True)
    column_names = ['img_name'] + list(res[0].names.values())
    table = get_rooms_info(res)
    marked_img = show_predicted_img(res)
    return table, marked_img

def show_predicted_img(res, n=0):
    res_img = Image.open(os.path.join(res[n].save_dir,  Path(res[n].path).name))
    return res_img


def preprocess(img_path):
    img = cv2.imread(img_path).astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(gray, blur, scale=255)

    return divide


def cutout(res, img, k):
    crop_sides = np.rint(res.boxes.xyxy.cpu().numpy()).astype(int)
    crop = img[crop_sides[k][1]:crop_sides[k][3], crop_sides[k][0]:crop_sides[k][2]]
    show_img = cv2.resize(crop, (0, 0), fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC)

    return show_img


def process_ocr(ocr_obj):
    processed = []
    for obj in ocr_obj:
        #print(obj)
        conf = obj[2]
        if conf > 0.6:
            text = obj[1].lower()
            pattern1 = r'\d+(?:\.\d+)?\s*m2'
            pattern2 = r"\d*\.?\d+"
            if re.match(pattern1, text):
                return float(re.findall(pattern2, text)[0]), conf
            else:
                replaced = text.replace(',', '.')
                numbers = re.findall(pattern2, replaced)
                processed.extend(map(float,numbers))
    res_num = max(processed)
    #print(processed)
    #res_num = '/'.join(processed)
    return res_num, conf

def estimate_footage(read_data, confs, pix_footage):
    mat = []
    for i in range(len(read_data)):
        if confs[i]>0.7 and read_data[i] > 0:
            arr = np.zeros(len(read_data))
            for j in range(len(arr)):
                arr[j] = pix_footage[j]*read_data[i]/pix_footage[i]
            mat.append(arr)
    if not mat:
        return [[0]*len(read_data),[0]*len(read_data)] #so that later median would return [0,0,...] and not 0
    else:
        return np.vstack(mat)


def get_rooms_info(results):
    res_data = []
    column_names = ['img_name'] + list(results[0].names.values())
    easyOcr = easyocr.Reader(['en'])
    for result in tqdm(results):
        combined_data = {}
        classes = result.boxes.cls.cpu().numpy().astype(int)

        if classes.size == 0:
            for col in column_names[1:]:
                combined_data[col] = ['ND']
            res_data.append(combined_data)
            continue

        # print('cl', classes)
        wh = result.boxes.xywh[:, 2:].cpu().numpy().astype(float)
        pixel_footages = wh[:, 0] * wh[:, 1]
        image = preprocess(result.path)
        # print(image)
        ocr_list = []
        foot_list = [] * max(classes)
        conf_list = [] * max(classes)
        conf = -1
        for k in range(len(classes)):
            cut_img = cutout(result, image, k)
            ocr_out = easyOcr.readtext(cut_img)

            try:
                area_value, conf = process_ocr(ocr_out)
            except:
                area_value = 0
            conf_list.append(conf)
            foot_list.append(area_value)
            ocr_list.append({column_names[classes[k] + 1]: area_value})
        foot_list = [x for _, x in sorted(zip(classes, foot_list), key=lambda pair: pair[0])]
        conf_list = [x for _, x in sorted(zip(classes, conf_list), key=lambda pair: pair[0])]
        pixel_footages = [x for _, x in sorted(zip(classes, pixel_footages), key=lambda pair: pair[0])]


        for col in column_names[1:]:
            combined_data[col] = list(set([d[col] for d in ocr_list if col in d])) or ['ND']
        combined_data['img_name'] = Path(result.path).name

        est = np.round(np.median(estimate_footage(foot_list, conf_list, pixel_footages), axis=0), 3)
        cnt = 0
        for k in list(combined_data.keys())[:-1]:
            if combined_data[k] != ['ND']:
                for j in range(len(combined_data[k])):
                    if conf_list[cnt] > 0.8 and foot_list[cnt] != 0 and 0.5 < foot_list[cnt] / est[cnt] < 1.5:
                        combined_data[k][j] = str(foot_list[cnt])
                    elif est.all() == 0:
                        combined_data[k][j] = '0'
                    else:
                        combined_data[k][j] = (str(est[cnt])+' est.')
                    cnt += 1
        #print(combined_data)
        res_data.append(combined_data)
    df = pd.DataFrame(res_data, columns=column_names)
    return df