import os
import cv2
import numpy as np
import shutil
import random

rot_img_size = 416 # 1200
dw, dh = (300, 200)

def extract_contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,5)
    ret,img_bin = cv2.threshold(img_gray,140,255,cv2.THRESH_BINARY) #110
    img_bin = cv2.bitwise_not(img_bin) # 白黒反転
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_sdcard_images(sdcard_images, sdcard_classs, img_name, class_name):
    img_orig = cv2.imread(img_name+'.jpg')
    contours = extract_contours(img_orig)
    # 輪郭を描画
    img_con = img_orig.copy()
    #center = (int(rot_img_size/2), int(rot_img_size/2))
    count = 0
    for i in range(len(contours)):
        # 外接矩形（正立）
        x, y, w, h = cv2.boundingRect(contours[i])
        if w>440 and h>580:
            count = count + 1
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
            sdcard_images.append(image_for_rotate)
            sdcard_classs.append(class_name)
    return sdcard_images, sdcard_classs

def make_mask(img_obj):
    kernel = np.ones((3,3),np.uint8)
    img_gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(img_bin)
    contours,_ = cv2.findContours(img_bin, 1, 2)
    img_mask = np.zeros([img_bin.shape[0], img_bin.shape[1]], dtype="uint8")
    for contour in contours:
        cv2.fillPoly(img_mask, [contour], (255, 255, 255))
    img_mask = cv2.dilate(img_mask, kernel, iterations = 1)
    #cv2.imwrite('{}-{}.jpg'.format(i,j), img_mask)
    return img_mask

def calc_extent(img_obj):
    obj_contours = extract_contours(img_obj)
    num_contours = 0
    for j in range(len(obj_contours)):
        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(obj_contours[j])
        if obj_w>170 and obj_h>170:
            num_contours = num_contours + 1
            return obj_x, obj_y, obj_w, obj_h
    return 0, 0, 0, 0

def make_training_images(image_path, sdcard_images, sdcard_classes):
    with open('./train.txt', 'w') as ft:
        pass
    with open('./train.txt', 'a') as ft:
        for i in range(100):
            ft.write('{}{}.jpg'.format(image_path, str(i)))
            train_image = np.zeros((rot_img_size*3, rot_img_size*3, 3), np.uint8)
            train_image[:] = 225
            for j in range(9):
                angle = random.randrange(360)
                img_id = random.randrange(len(sdcard_images))
                sdcard_image = sdcard_images[img_id]
                sdcard_class = sdcard_classes[img_id]
                center = (int(rot_img_size/2), int(rot_img_size/2))
                trans = cv2.getRotationMatrix2D(center, angle , 1.0)
                img_obj = cv2.warpAffine(sdcard_image, trans, (rot_img_size, rot_img_size), borderMode=cv2.BORDER_REPLICATE)
                img_mask = make_mask(img_obj)
    
                ix = int(j%3)
                iy = int(j/3)
                if ix==0:   pos_x = int(ix*rot_img_size + random.randrange(rot_img_size/2))
                elif ix==1: pos_x = int(ix*rot_img_size + random.randrange(rot_img_size) - rot_img_size/2)
                else:       pos_x = int(ix*rot_img_size - random.randrange(rot_img_size/2))
                    
                if iy==0:   pos_y = int(iy*rot_img_size + random.randrange(rot_img_size/2))
                elif iy==1: pos_y = int(iy*rot_img_size + random.randrange(rot_img_size) - rot_img_size/2)
                else:       pos_y = int(iy*rot_img_size - random.randrange(rot_img_size/2))
    
                bg_roi = train_image[pos_y:pos_y+rot_img_size,pos_x:pos_x+rot_img_size]
                train_image[pos_y:pos_y+rot_img_size,pos_x:pos_x+rot_img_size] \
                    = np.where(img_mask[:,:,np.newaxis]==0, bg_roi, img_obj)
    
                # アノテーション
                obj_x, obj_y, obj_w, obj_h = calc_extent(img_obj)
                obj_x = obj_x + pos_x
                obj_y = obj_y + pos_y
                #cv2.rectangle(train_image, (obj_x, obj_y), (obj_x+obj_w, obj_y+obj_h), (0, 0, 255), 7)
                ft.write(' {},{},{},{},{}'.format(
                    obj_x, obj_y, obj_x+obj_w, obj_y+obj_h, sdcard_class[1]))
            ft.write('\n')
            cv2.imwrite(image_path+'{}.jpg'.format(i), train_image)


'''
def make_rotated_images(data_path, image_path, img_name, class_name):
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
            with open('./train.txt', 'a') as ft:
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
                    ft.write('{}{}.jpg {},{},{},{},{}\n'.format(
                        image_path, file_name, obj_x, obj_y, obj_x+obj_w, obj_y+obj_h, class_name[1]))
                
                # 確認用のアノテーション付き教師画像書き出し
                #cv2.imwrite(data_path+'{}_{}.jpg'.format(img_name, count), image_6x6)
    #cv2.imwrite(data_path+img_name+'_contours1.jpg', img_con)
'''

def main():
    data_path = './data/'
    image_path = data_path + 'Images/'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    img_names = ['SD_FRONT_1', 'SD_FRONT_5', 'SD_BACK_1', 'SD_BACK_5']
    class_names = [('sd_front',0), ('sd_front',0), ('sd_back',1), ('sd_back',1)]
    sdcard_images = []
    sdcard_classs = []
    for img_name, class_name in zip(img_names, class_names):
        sdcard_images, sdcard_classs = extract_sdcard_images(sdcard_images, sdcard_classs, img_name, class_name)
    make_training_images(image_path, sdcard_images, sdcard_classs) 

if __name__ == '__main__':
    main()
