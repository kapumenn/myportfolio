import cv2
import copy
import sys
import numpy as np
from matplotlib import pyplot as plt


# 動画ファイルに対する処理の基本形
import cv2   # 使用するライブラリ（OpenCV）をインポート

#カメラ接続状態フラグ
flag = True

#　USBカメラ接続
capture = cv2.VideoCapture("bird.mp4")        ### USBカメラはID(0) ### 動画ファイルは"ファイル名"
if capture.isOpened() is False:
    flag = False

# 動画像処理の無限ループ（画像を取得できる限り繰り返す）
while flag:
    
    # 画像のキャプチャ（取得）
    ret, img1 = capture.read()   # キャプチャ実行（その瞬間のカメラ画像を取得しimg1に格納）
    if ret == False:             # キャプチャ失敗したらループを抜けて終了
        break            


     # 線形フィルタ
    filter_kernel0 = np.array([
                  [-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]
                  ], np.float32)

    # 画像処理の例（ぼかし処理）
    img2 = cv2.filter2D(img1, -1, filter_kernel0)  # img1にぼかし処理をかけて結果をimg2に格納
    
   


    # 画像の画面表示
    cv2.imshow("Capture", img1)   # Captureという名前の画像表示ウィンドウにimg1を表示
    cv2.imshow("Result", img2)    # Resultという名前の画像表示ウィンドウにimg2を表示
    
    


    ### 終了処理 ###
    k = cv2.waitKey(20) & 0xff   # 操作待ち時間をmsec（ミリ秒）で指定（画像表示更新のためにも必要）
    if k == 27:                  # ESCキーが押されたことをチェック
        cv2.destroyAllWindows()  # 画像表示ウィンドウを破棄
        capture.release()        # キャプチャを破棄
        break                    # ループを抜けて終了