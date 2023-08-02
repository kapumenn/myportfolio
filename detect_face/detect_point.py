#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 #OpenCV:画像処理系ライブラリ
import dlib #機械学習系ライブラリ
import imutils #OpenCVの補助
from imutils import face_utils
import numpy as np


# VideoCapture オブジェクトを取得します
DEVICE_ID = 0 #ID 0は標準web cam
capture = cv2.VideoCapture(DEVICE_ID)#dlibの学習済みデータの読み込み
predictor_path = "C:\detect_face\shape_predictor_68_face_landmarks.dat"

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し。ただ顔だけを検出する。
predictor = dlib.shape_predictor(predictor_path) #顔から目鼻などランドマークを出力する

while(True): #カメラから連続で画像を取得する
    ret, frame = capture.read() #カメラからキャプチャしてframeに１コマ分の画像データを入れる

    frame = imutils.resize(frame, width=2000) #frameの画像の表示サイズを整える
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray scaleに変換する
    rects = detector(gray, 0) #grayから顔を検出
    image_points = None

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #print(shape[30])#鼻の座標
        cal = shape-shape[30]
        print("######[X,Y]#######",
              "\n point18=",cal[17],
              "\n point22=",cal[21],
              "\n point37=",cal[36],
              "\n point40=",cal[39],
              "\n point28=",cal[27],
              "\n point31=",cal[30],
              "\n point32=",cal[31],
              "\n point49=",cal[48],
              "\n point58=",cal[57],
              "\n point9=",cal[8])

        for (x, y) in shape: #顔全体の68箇所のランドマークをプロット
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
            cv2.putText(frame,str((x, y)-shape[30]),(x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)


    cv2.imshow('frame',frame) # 画像を表示する
    if cv2.waitKey(1) & 0xFF == ord('q'): #qを押すとbreakしてwhileから抜ける
        break

capture.release() #video captureを終了する
cv2.destroyAllWindows() #windowを閉じる