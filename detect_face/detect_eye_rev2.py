import cv2
import numpy as np
import dlib
import datetime
import csv
import imutils #OpenCVの補助
from imutils import face_utils

DEVICE_ID = 0 #ID 0は標準web cam
capture = cv2.VideoCapture(DEVICE_ID)#dlibの学習済みデータの読み込み
predictor_path = "C:\detect_face\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し。ただ顔だけを検出する。
predictor = dlib.shape_predictor(predictor_path) #顔から目鼻などランドマークを出力する
i = 0
idnumber=0
left_eye_list=[]
right_eye_list=[]
face_degree_list=[]
while(True):
    ret, frame = capture.read() #カメラからキャプチャしてframeに１コマ分の画像データを入れる

    frame = imutils.resize(frame, width=1000) #frameの画像の表示サイズを整える
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray scaleに変換する
    rects = detector(gray, 0) #grayから顔を検出
    image_points = None

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape: #顔全体の68箇所のランドマークをプロット
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        image_points = np.array([
                tuple(shape[30]),#鼻頭
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ],dtype='double')

    if len(rects) > 0:
        model_points = np.array([
                (0.0,0.0,0.0), # 30
                (-30.0,-125.0,-30.0), # 21
                (30.0,-125.0,-30.0), # 22
                (-60.0,-70.0,-60.0), # 39
                (60.0,-70.0,-60.0), # 42
                (-40.0,40.0,-50.0), # 31
                (40.0,40.0,-50.0), # 35
                (-70.0,130.0,-100.0), # 48
                (70.0,130.0,-100.0), # 54
                (0.0,158.0,-10.0), # 57
                (0.0,250.0,-50.0) # 8
                ])
                #モデルの点を決めるここを人によって決定する。

        size = frame.shape

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2) #顔の中心座標

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #回転行列とヤコビアン
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))

        #yaw,pitch,rollの取り出し
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]

        #ヨー、ピッチ、ロールの出力
        print("yaw",int(yaw),"pitch",int(pitch),"roll",int(roll))#頭部姿勢データの取り出し
        idnumber=idnumber+1;
        dt=datetime.datetime.now()
        face_degree_list.append([idnumber,dt,yaw,pitch,roll])
        cv2.putText(frame, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)
        #計算に使用した点のプロット/顔方向のベクトルの表示
        for p in image_points:
            cv2.drawMarker(frame, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)
    
    cv2.imshow('frame',frame) # 画像を表示する
    face = open("face.csv", 'w', newline='')
    writer = csv.writer(face)
    writer.writerows(face_degree_list)
    face.close()

    
    _, frame = capture.read()
    
    #グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #ランドマーク
    faces = detector(gray)
    if(len(faces)==0):
        print("顔がカメラに移っていないです。")
    else:
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predictor(gray, face)

            # for n in range(0,68):
            #     x = landmarks.part(n).x
            #     y = landmarks.part(n).y
            #     cv2.circle(frame, (x,y), 2, (255,0,0), -1)
        # 瞳のトリミング処理
        # 右目：[36,,37,39, 40]　左目：[42, 43, 45, 46]
        idnumber=idnumber+1;
        # Right eye
        r_x1,r_y1 = landmarks.part(36).x,landmarks.part(36).y
        r_x2,r_y2 = landmarks.part(37).x,landmarks.part(37).y
        r_x3,r_y3 = landmarks.part(39).x,landmarks.part(39).y
        r_x4,r_y4 = landmarks.part(40).x,landmarks.part(40).y
        # Left eye
        l_x1,l_y1 = landmarks.part(42).x,landmarks.part(42).y
        l_x2,l_y2 = landmarks.part(43).x,landmarks.part(43).y
        l_x3,l_y3 = landmarks.part(45).x,landmarks.part(45).y
        l_x4,l_y4 = landmarks.part(46).x,landmarks.part(46).y

        #　トリミング範囲補正
        trim_val = 2
        r_frame_trim = frame[r_y2-trim_val:r_y4+trim_val, r_x1:r_x3]
        l_frame_trim = frame[l_y2-trim_val:l_y4+trim_val, l_x1:l_x3]

        # 拡大処理（5倍）
        r_height,r_width = r_frame_trim.shape[0],r_frame_trim.shape[1]
        l_height,l_width = l_frame_trim.shape[0],l_frame_trim.shape[1]
        r_frame_trim_resize = cv2.resize(r_frame_trim , (int(r_width*7.0), int(r_height*7.0)))
        l_frame_trim_resize = cv2.resize(l_frame_trim , (int(l_width*7.0), int(l_height*7.0)))

        # グレースケール処理
        r_frame_gray = cv2.cvtColor(r_frame_trim_resize, cv2.COLOR_BGR2GRAY)
        l_frame_gray = cv2.cvtColor(l_frame_trim_resize, cv2.COLOR_BGR2GRAY)

        #平滑化（ぼかし）
        r_frame_gray = cv2.GaussianBlur(r_frame_gray,(7,7),0)
        l_frame_gray = cv2.GaussianBlur(l_frame_gray,(7,7),0)

        # 2値化処理
        thresh = 80
        maxval = 255
        e_th,r_frame_black_white = cv2.threshold(r_frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV)
        l_th,l_frame_black_white = cv2.threshold(l_frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV)

        #輪郭の表示
        r_eye_contours, _ = cv2.findContours(r_frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        r_eye_contours = sorted(r_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる
    
        for cnt in r_eye_contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # cv2.drawContours(r_frame_trim_resize, [cnt], -1, (0,0,255),3) #輪郭の表示
            # cv2.rectangle(r_frame_trim_resize, (x, y), ((x + w, y + h)), (255, 0, 0), 2)#矩形で表示
            cv2.circle(r_frame_trim_resize, (int(x+w/2), int(y+h/2)), int((w+h)/4), (255, 0, 0), 2) #円で表示
            cv2.circle(frame, (int(r_x1+(x+w)/10), int(r_y2-3+(y+h)/10)), int((w+h)/20), (0, 255, 0), 1)    #元画像に表示
            coordinate_r= (int(x+w/2),int(y+h/2))
            dt_r=datetime.datetime.now()
            right_eye_list.append([idnumber,dt_r,coordinate_r])

        l_eye_contours, _ = cv2.findContours(l_frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l_eye_contours = sorted(l_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる

        for cnt in l_eye_contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # cv2.drawContours(l_frame_trim_resize, [cnt], -1, (0,0,255),3) #輪郭の表示
            # cv2.rectangle(l_frame_trim_resize, (x, y), ((x + w, y + h)), (255, 0, 0), 2)#矩形で表示
            cv2.circle(l_frame_trim_resize, (int(x+w/2), int(y+h/2)), int((w+h)/4), (255, 0, 0), 2) #円で表示
            cv2.circle(frame, (int(l_x1+(x+w)/10), int(l_y2-3+(y+h)/10)), int((w+h)/20), (0, 255, 0), 1)    #元画像に表示
            coordinate_l= (int(x+w/2),int(y+h/2))
            dt_l=datetime.datetime.now()
            left_eye_list.append([idnumber,dt_l,coordinate_l])
        
        right = open("righteye.csv", 'w', newline='')
        writer = csv.writer(right)
        writer.writerows(right_eye_list)
        right.close()

        left = open("lefteye.csv", 'w', newline='')
        writer = csv.writer(left)
        writer.writerows(left_eye_list)
        left.close()


        #画像の表示    
        cv2.imshow("frame",frame)
        
        cv2.imshow("right eye trim",r_frame_trim_resize)
        cv2.imshow("left eye trim",l_frame_trim_resize)

        cv2.imshow("right eye black white",r_frame_black_white)
        cv2.imshow("left eye black white",l_frame_black_white)

        #ウィンドウの配置変更
        cv2.moveWindow('frame', 200,0)
        cv2.moveWindow('right eye trim', 100,100)
        cv2.moveWindow('left eye trim', 240,100)
        cv2.moveWindow('right eye black white', 100,250)
        cv2.moveWindow('left eye black white', 240,250)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): #qを押すとbreakしてwhileから抜ける
        break
