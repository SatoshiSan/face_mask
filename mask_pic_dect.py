# -*- coding: utf-8 -*-
# @File : mask_pic_dect.py
# @Author : kangkang_LaoShi
# @Time : 2022/08/25 17:03:13
import paddlehub as hub
import cv2

mask_detector = hub.Module(name="pyramidbox_lite_server_mask")
img_path = '1.png'
img = cv2.imread(img_path)

input_dict = {"data": [img]}
result = mask_detector.face_detection(data=input_dict)

count = len(result[0]['data'])
if count < 1:
    print('There is no face detected!')
else:
    for i in range(0, count):

        # print(result[0]['data'][i])
        label = result[0]['data'][i].get('label')
        score = float(result[0]['data'][i].get('confidence'))
        x1 = int(result[0]['data'][i].get('left'))
        y1 = int(result[0]['data'][i].get('top'))
        x2 = int(result[0]['data'][i].get('right'))
        y2 = int(result[0]['data'][i].get('bottom'))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        if label == 'NO MASK':
            cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 255, 0), 2)

cv2.imwrite('result.jpg', img)
cv2.imshow('mask-detection', img)
cv2.waitKey()
cv2.destroyAllWindows()
print('Done!')
