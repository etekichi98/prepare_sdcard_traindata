import os
import cv2
import numpy as np
import shutil

def extract_contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,5)
    ret,img_bin = cv2.threshold(img_gray,140,255,cv2.THRESH_BINARY) #110
    img_bin = cv2.bitwise_not(img_bin) # 白黒反転
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def make_rotated_images(data_path, image_path, annotation_path, img_name, class_name):
    img_orig = cv2.imread(img_name+'.jpg')
    contours = extract_contours(img_orig)
    # 輪郭を描画
    img_con = img_orig.copy()
    dw, dh = (300, 200)
    rot_img_size = 416 # 1200
    center = (int(rot_img_size/2), int(rot_img_size/2))
    count = 0
    for i in range(len(contours)):
        # 外接矩形（正立）
        x, y, w, h = cv2.boundingRect(contours[i])
        if w>440 and h>580:
            count = count + 1
            #with open('./train.txt', 'a') as ft:
            if True:
                cv2.rectangle(img_con, (x, y), (x + w, y + h), (255, 0, 0), 7)
                img_obj = img_orig[y-dh:y+h+dh,x-dw:x+w+dw]
                
                obj_w = w + dw + dw
                obj_h = h + dh + dh
                obj_rate = rot_img_size / max(obj_w, obj_h)
                img_obj = cv2.resize(img_obj, dsize=(int(obj_w*obj_rate), int(obj_h*obj_rate)) )
        
                image_for_rotate = np.zeros((rot_img_size, rot_img_size, 3), np.uint8)
                image_for_rotate[:] = img_obj[1,1]
                dx = int((rot_img_size - img_obj.shape[1])/2.0)
                dy = int((rot_img_size - img_obj.shape[0])/2.0)
                image_for_rotate[dy:img_obj.shape[0]+dy, dx:img_obj.shape[1]+dx] = img_obj
                
                image_6x6 = np.zeros((rot_img_size*6, rot_img_size*6, 3), np.uint8)
                image_6x6[:] = 0
    
                for angle in range(0, 360, 10):
                    trans = cv2.getRotationMatrix2D(center, angle , 1.0)
                    img_obj = cv2.warpAffine(image_for_rotate, trans, (rot_img_size, rot_img_size),
                                             borderMode=cv2.BORDER_REPLICATE)
                    file_name = '{}_{}_{}'.format(img_name, count, angle)
                    # 教師画像書き出し
                    cv2.imwrite(image_path+'{}.jpg'.format(file_name), img_obj)
                    # アノテーション書き出し
                    obj_contours = extract_contours(img_obj)
                    num_contours = 0
                    for j in range(len(obj_contours)):
                        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(obj_contours[j])
                        if obj_w>440*obj_rate and obj_h>440*obj_rate:
                            num_contours = num_contours + 1
                            cv2.rectangle(img_obj, (obj_x, obj_y), (obj_x+obj_w, obj_y+obj_h), (0, 0, 255), 7)
                    if num_contours!=1:
                        print('Error in {} num_contours={}'.format(file_name, num_contours))
                    ix = int((angle%60)/10)
                    iy = int(angle/60)
                    image_6x6[iy*rot_img_size:(iy+1)*rot_img_size,
                              ix*rot_img_size:(ix+1)*rot_img_size] = img_obj
                    # アノテーション書き出し
                    annotation_text = make_yolo_annotation(image_path, file_name,
                                        rot_img_size, rot_img_size, class_name[1],
                                        obj_x, obj_y, obj_x+obj_w, obj_y+obj_h)
                    with open(annotation_path+'{}.txt'.format(file_name), 'w') as fa:
                        fa.write(annotation_text)
                    #ft.write('{}{}.jpg {},{},{},{},{}\n'.format(
                    #    image_path, file_name, obj_x, obj_y, obj_x+obj_w, obj_y+obj_h, class_name[1]))
                
                # 確認用のアノテーション付き教師画像書き出し
                #cv2.imwrite(data_path+'{}_{}.jpg'.format(img_name, count), image_6x6)
    #cv2.imwrite(data_path+img_name+'_contours1.jpg', img_con)

def make_yolo_annotation(image_path, img_name, img_width, img_height, class_id, xmin, ymin, xmax, ymax):
    '''
    YOLO形式のアノテーション(yolov4)
    '''
    xcenter = (xmax + xmin)/img_width/2.0
    ycenter = (ymax + ymin)/img_height/2.0
    width = (xmax - xmin)/img_width
    height = (ymax - ymin)/img_height
    annotation_text = '{} {} {} {} {}\n'.format(class_id, xcenter, ycenter, width, height)
    return annotation_text

def make_classes_txt(path):
    with open(path+'classes.txt', mode='w') as f:
       f.write('sd_front\n') 
       f.write('sd_back\n') 
    
def main():
    os.makedirs('./training/train/images/', exist_ok=True)
    os.makedirs('./training/train/labels/', exist_ok=True)
    make_classes_txt('./training/train/labels/')
    os.makedirs('./training/test/images/', exist_ok=True)
    os.makedirs('./training/test/labels/', exist_ok=True)
    shutil.copy2('./SD_TEST.jpg', './training/test/images/')
    shutil.copy2('./SD_TEST.txt', './training/test/labels/')
    make_classes_txt('./training/test/labels/')
    os.makedirs('./training/val/images/', exist_ok=True)
    os.makedirs('./training/val/labels/', exist_ok=True)
    
    data_path = './training/train/'
    annotation_path = data_path + 'labels/'
    image_path = data_path + 'images/'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(annotation_path, exist_ok=True)
    img_names = ['SD_FRONT_1', 'SD_FRONT_5', 'SD_BACK_1', 'SD_BACK_5']
    class_names = [('sd_front',0), ('sd_front',0), ('sd_back',1), ('sd_back',1)]
    for img_name, class_name in zip(img_names, class_names):
        make_rotated_images(data_path, image_path, annotation_path, img_name, class_name)

if __name__ == '__main__':
    main()
