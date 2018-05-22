# -*- coding: utf-8 -*
import time
import math
import cv2
import scipy.io as sio
from pylab import *
from statistics import *
import numpy as np
from skimage import measure
from matplotlib import colors
from PIL import Image

def find(matrix,x):
    row = []
    col = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == x:
                row.append(i)
                col.append(j)
    return row,col

def autoroi(bwimage):
    row,col = find(bwimage,1)
    y_roi = min(col,default=0)
    x_roi = min(row,default=0)
    try:
        width_roi = max(col)-min(col,default=0)
    except ValueError:
        width_roi = 0
    try:
        height_roi = max(row)-min(row,default=0)
    except ValueError:
        height_roi = 0
    if len(row) == 0:
        hand_roi = bwimage
        w = len(bwimage[0])
        h = len(bwimage)
    else:
        hand_roi = bwimage[x_roi:x_roi + height_roi,y_roi:y_roi + width_roi]
        w = width_roi
        h = height_roi
    return hand_roi,w,h

def  Comp_effluent(IM,I_t,thresh,x,y):
    IM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
    I_t = cv2.cvtColor(I_t,cv2.COLOR_BGR2GRAY)
    x_point = int(x[0])
    y_point = int(y[0])
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]- x[0]) + 1
    I_tc = I_t[x_point:x_point + height,y_point:y_point + width]
    box = IM[x_point:x_point + height,y_point:y_point + width]
    diff = double((abs(double(box)-double(I_tc))))
    retval,bw = cv2.threshold(diff,thresh*255,1, cv2.THRESH_BINARY)
    bw = np.array(bw)
    score = np.sum(bw==1)/bw.size
    return score

def Judge_effluent(score,result_total,j,mint,maxt,ef_et):
    if score>mint and score<maxt:
        result = 1
    elif score>=maxt:
        if j <= ef_et:
            result = mode(result_total[:j])
        else:
            result = mode(result_total[(j-ef_et-1):(j-1)])
    else:
        result = 0
    return result

def illumination_correct(im):
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = np.array(im)/255
    HSV = colors.rgb_to_hsv(im)
    H,S,V = cv2.split(HSV)
    HSIZE = min(len(im),len(im[0]))
    # 卷积核的大小必须为奇数
    if HSIZE % 2 == 0:
        HSIZE -=1
    q=sqrt(2)
    SIGMA1 = 15
    SIGMA2 = 80
    SIGMA3 = 250
    gaus1 = cv2.GaussianBlur(V, (HSIZE,HSIZE), SIGMA1 / q)
    gaus2 = cv2.GaussianBlur(V, (HSIZE,HSIZE), SIGMA2 / q)
    gaus3 = cv2.GaussianBlur(V, (HSIZE,HSIZE), SIGMA3 / q)
    gaus = (np.array(gaus1) + np.array(gaus2) + np.array(gaus3))/3
    m = np.mean(np.array(gaus))
    gama = np.power(0.5,((m-np.array(gaus))/m))
    out = np.power(np.array(V),np.array(gama))
    newHSV = cv2.merge([H,S,out])
    rgb = colors.hsv_to_rgb(newHSV)
    rgb = uint8(np.array(rgb)*255)
    rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    return rgb

def skindetect2(BGR):
    B,G,R = cv2.split(BGR)
    YCrCb = cv2.cvtColor(BGR, cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb = cv2.split(YCrCb)
    I = BGR
    rows = len(Y)
    columns = len(Y[0])
    k = (2.53/180)*math.pi
    m = math.sin(k)
    n = math.cos(k)
    x,y = 0,0
    cx,cy,ecx,ecy,a,b = 109.38,152.02,1.60,2.41,25.39,14.03
    for i in range(rows):
        for j in range(columns):
            if Y[i,j] < 80:
                I[i,j,:] = 0
            elif Y[i,j] <= 230 and Y[i,j] >= 80:
                x=(double(Cb[i,j])-cx)*n+(double(Cr[i,j])-cy)*m
                y=(double(Cr[i,j])-cy)*n-(double(Cb[i,j])-cx)*m
            elif Y[i,j] > 230:
                x=(double(Cb[i,j])-cx)*n+(double(Cr[i,j])-cy)*m
                y=(double(Cr[i,j])-cy)*n-(double(Cb[i,j])-cx)*m
                if ((x-ecx)**2/a**2+(y-ecy)**2/b**2) <= 1:
                    I[i,j,:] = 255
                else:
                    I[i,j,:] = 0
    I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    retval,I = cv2.threshold(I,0.5*255,1, cv2.THRESH_BINARY)
    return I

def Comp_wash(IM,x,y):
    x_point = int(x[0])
    y_point = int(y[0])
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    box = IM[x_point:x_point+height,y_point:y_point+width]
    box = cv2.resize(box,(math.ceil(width*0.3),math.ceil(height*0.3)),interpolation=cv2.INTER_NEAREST)
    box = illumination_correct(box)
    hand = skindetect2(box)
    return hand

def Judge_wash(c,j,ef):
    if j <= ef:
        try:
            if mode(c[:j]) == 1:
                result = 1
            else:
                result = 0
        except StatisticsError:
            return 0
    else:
        if mode(c[j-ef-1:j]) == 1:
            result = 1
        else:
            result = 0
    return result

def Comp_soap_new(IM,x,y):
    x_point = int(x[0])
    y_point = int(y[0])
    width = int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    box = IM[x_point:x_point+height,y_point:y_point+width]
    hand = skindetect2(box)
    hand = np.array(hand)
    hll = np.sum(hand == 0) / hand.size
    return hll,hand

def Judge_soap(c_sp,j,fps,tt):
    if c_sp[j] ==0:
        if (len([i for i in c_sp[:j] if i ==1])) / fps <tt:
            result = 0
        else:
            result = 3
    else:
        if (len([i for i in c_sp[:j] if i ==1])) / fps <tt:
            result=4
        else:
            result=5
    return result

def Comp_foam_new(IM,bwt,x,y):
    x_point = int(np.round(x[0]))
    y_point = int(np.round(y[0]))
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    box = IM[x_point:x_point+height,y_point:y_point+width]
    box_small = cv2.resize(box,(math.ceil(width*0.3),math.ceil(height*0.3)),interpolation=cv2.INTER_NEAREST)
    box_c = illumination_correct(box_small)
    R, G, B = cv2.split(box_c)
    box_c = cv2.merge([B, G, R])
    I = double(cv2.cvtColor(box_c,cv2.COLOR_BGR2GRAY))
    retval, bw = cv2.threshold(I, bwt * 255, 1, cv2.THRESH_BINARY)
    bw = np.array(bw)
    s = np.sum(bw == 1)/bw.size
    return s,bw

def Judge_foam(num,j,ef,crnt,pfnt,tpft):
    if j<=ef:
        pfnt[j-1] = sum(num[:j]>crnt)
    else:
        pfnt[j-1] = sum(num[j-ef-1:j]>crnt)
    length = 0
    for i in range(len(pfnt)):
        for j in range(len(pfnt[0])):
            if pfnt[i][j] > pfnt:
                length += 1
    if length > tpft:
        result = 1
    else:
        result = 0
    return result

def ShowResult(result):
    if result == 1:
        word = 'yes'
    elif result == 2:
        word = 'cannot judge'
    elif result == 3:
        word = 'no(done)'
    elif result == 4:
        word = 'yes(ing)'
    elif result == 5:
        word = 'yes(done)'
    else:
        word = 'no'
    return word

def Judge_washcurrent(hand,result):
    col = []
    for i in range(len(hand)):
        for j in range(len(hand[0])):
            if hand[i][j] == 1:
                col.append(j)
    if result == 1:
        if min(col,default=0)<(0.2*len(hand[0])):
            c = 1
        else:
            c = 0
    else:
         c = 0
    return c

if __name__ == "__main__":
    time_start = time.time()

    video_full_path = "./Video/yy2.mp4"
    capture = cv2.VideoCapture(video_full_path)

    # opencv2.4.9用cv2.cv.CV_CAP_PROP_FPS；如果是3用cv2.CAP_PROP_FPS
    (major_ver,minor_ver,subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = capture.get(cv2.CAP_PROP_FPS)

    #获取所有帧
    frame_count = 0
    all_frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1

    #I_t为第一帧也就是背景
    I_t = all_frames[0]

    #设置参数
    stf = 1
    ovf = frame_count
    min_et = 0.16
    max_et = 0.3
    bwt_et = 30 / 255
    ef_et = 4
    # ef_et = 5
    bwt_ws = 50 / 255
    dlt_ws = 10
    ef_ws = 20
    # lr_sp = 0.1
    lr_sp = 0.94
    tt_sp = 0.5
    lt_sp = 100
    ef_fm = math.ceil(fps) * 2
    bwt_fm = 140 / 255
    dlt_fm = 15
    rtt_fm = 0.07
    prtt_fm = 15
    tpft_fm = 1

    #获取ROI区域中的八个坐标点
    data = sio.loadmat('ROI_yy2_fm.mat')
    x_et = data['x_et']
    y_et = data['y_et']
    x_ws = data['x_ws']
    y_ws = data['y_ws']
    x_sp = data['x_sp']
    y_sp = data['y_sp']
    x_fm = data['x_fm']
    y_fm = data['y_fm']
	
    #初始化数组长度为所有帧的个数
    s_et = [0 for i in range(len(all_frames))]
    result_ws = [0 for i in range(len(all_frames))]
    result_et = [0 for i in range(len(all_frames))]
    result_fm = [0 for i in range(len(all_frames))]
    result_sp = [0 for i in range(len(all_frames))]
    c_w = [0 for i in range(len(all_frames))]
    c_sp = [0 for i in range(len(all_frames))]
    rt_fm = [0 for i in range(len(all_frames))]
    prt_fm = [0 for i in range(len(all_frames))]
    hl = [0 for i in range(len(all_frames))]

    cv2.namedWindow("Result")
    for i in range(len(all_frames)):
        IM = all_frames[i]
        B,G,R = cv2.split(IM)
        j = i-(stf-1)

        ##### 水流判断
        s_et[j] = Comp_effluent(IM,I_t,bwt_et,x_et,y_et)
        result_et[j] = Judge_effluent(s_et[j],result_et,j,min_et,max_et,ef_et)

        #### 洗手判断
        hand = Comp_wash(IM,x_ws,y_ws)
        c_w[j] = Judge_washcurrent(hand,result_et[j])
        result_ws[j] = Judge_wash(c_w,j,ef_ws)
        time_ws = result_ws.count(1)/fps

        ##### 洗手液判断
        hl = np.array(hl).astype(float)
        hl[j],hand_sp = Comp_soap_new(IM,x_sp,y_sp)
        if (hl[j]) > lr_sp: # lr_sp = 0.1
            c_sp[j] = 1
        else:
            c_sp[j] = 0
        result_sp[j] = Judge_soap(c_sp, j, fps, tt_sp)

        #### 泡沫判断
        rt_fm[j], bw = Comp_foam_new(IM, bwt_fm,x_fm,y_fm)
        rt_fm = np.array(rt_fm)
        if j<=ef_fm:
            prt_fm[j] = len([i for i in rt_fm[:j] if i > rtt_fm])
        else:
            prt_fm[j] = len([i for i in rt_fm[j - ef_fm - 1:j] if i > rtt_fm])
        prt_fm = np.array(prt_fm)
        if np.sum(prt_fm[:j] > prtt_fm) > tpft_fm:
            result_fm[j] = 1
        else:
            result_fm[j] = 0

        word_et = ShowResult(result_et[j])
        word_ws = ShowResult(result_ws[j])
        word_sp = ShowResult(result_sp[j])
        word_fm = ShowResult(result_fm[j])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(IM, 'W:%s'%(str(word_et))+
                    ' '+'H:%s'%str(word_ws) +
                    ' ' + 'S:%s'%str(word_sp) +
                    ' '+ 'F:%s'%str(word_fm),
                    (30, 800), font, 1, (255, 0, 255), 2)#str(j)+' '+

        cv2.imshow("Result", IM)
        c = cv2.waitKey(10)

    time_end = time.time()
    time_cost = time_end - time_start
    print(time_cost)