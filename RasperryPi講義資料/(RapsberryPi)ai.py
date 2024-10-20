# -*- coding: utf-8 -*-
# ai.py
# An object detection sample with TensorFlow models.
# TensorFlowで作成された人工知能モデルを使用したサンプル
# This code is based on mobilenet_ssd_python.py from OpenCV3.4.1 (LICENSE_CV.txt) 
#
# Copyright 2018 Masami Yamakawa (MONOxIT Inc.)
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

# ライブラリ（ある機能を簡単にプログラミングできるように、あらかじめ作られたプログラム）をインポート（取り入れる）
# 画像処理用の多くの機能が入っているOpenCV(cv2)ライブラリ
import cv2

# 数値計算が楽になるnumpyをnpという名前で
import numpy as np
# コマンドのオプション指定を簡単に取り込めるargparse、乱数用のrandomや時刻取得用のtime
import argparse
import random
import time

# カメラバッファ読み飛ばし回数
CAMERA_BUF_FLUSH_NUM = 6

# 人工知能モデルへ入力する画像の調整パラメタ
IN_WIDTH = 300
IN_HEIGHT = 300

# Mobilenet SSD COCO学習済モデルのラベル一覧の定義
CLASS_LABELS = {0: 'background',
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
                5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
                10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
                23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
                38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
                57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
                65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
                73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
                81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                90: 'toothbrush'}

# 引数（コマンドラインのオプション指定）の定義
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--pbtxt', required=True,
                help='path to pbtxt file')
ap.add_argument('-w', '--weights', required=True,
                help='path to TensorFlow inference graph')
ap.add_argument('-c', '--confidence', type=float, default=0.3,
                help='minimum probability')
ap.add_argument('-i', '--interval', type=float, default=0,
                help='process interval to reduce CPU usage')
args = vars(ap.parse_args())

colors = {}
# ラベル毎の枠色をランダムにセット
random.seed()
for key in CLASS_LABELS.keys():
    colors[key] = (random.randrange(255),
                   random.randrange(255),
                   random.randrange(255))

# 人工知能モデルの読み込み
print('モデル読み込み...')
net = cv2.dnn.readNet(args['weights'], args['pbtxt'])

# ビデオカメラ開始
print('ビデオカメラ開始...')

cap = cv2.VideoCapture(0)

# OpenCVのチックメータ（ストップウオッチ）機能をtmという名前で使えるようにする
tm = cv2.TickMeter()

# 画像キャプチャと検出の永久ループ
while True:
    #バッファに滞留しているカメラ画像を指定回数読み飛ばし、最新画像をframeに読み込む
    for i in range(CAMERA_BUF_FLUSH_NUM):
        ret, frame = cap.read()
    
    # 取り込んだ画像の幅を縦横比を維持して500ピクセルに縮小
    ratio = 500 / frame.shape[1]
    frame = cv2.resize(frame, dsize=None, fx=ratio, fy=ratio)

    # 高さと幅情報を画像フレームから取り出す
    (frame_height, frame_width) = frame.shape[:2]

    # 画像フレームを調整しblob形式へ変換
    blob =  cv2.dnn.blobFromImage(frame, size=(IN_WIDTH, IN_HEIGHT), swapRB=False, crop=False)
            
    # blob形式の入力画像を人工知能にセット
    net.setInput(blob)
    
    # 画像を人工知能へ流す
    tm.reset()
    tm.start()
    detections = net.forward()
    tm.stop()

    # 検出数（mobilenet SSDでは100）繰り返し
    for i in range(detections.shape[2]):
        # i番目の検出オブジェクトの正答率を取り出す
        confidence = detections[0, 0, i, 2]

        # 正答率がしきい値を下回ったらなにもしない
        if confidence < args['confidence']:
            continue
            
        # 検出物体の種別と座標を取得
        class_id = int(detections[0, 0, i, 1])
        start_x = int(detections[0, 0, i, 3] * frame_width)
        start_y = int(detections[0, 0, i, 4] * frame_height)
        end_x = int(detections[0, 0, i, 5] * frame_width)
        end_y = int(detections[0, 0, i, 6] * frame_height)
        
        # 枠をフレームに描画
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                      colors[class_id], 2)
        
        # 物体の種別を示す personといったラベルと確信度をlabelにセット
        label = CLASS_LABELS[class_id]
        label += ': ' + str(round(confidence * 100, 2)) + '%'
        label_size, base_line = cv2.getTextSize(label,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, 1)

        cv2.rectangle(frame, (start_x, start_y - label_size[1]),
                      (start_x + label_size[0], start_y + base_line),
                      (255, 255, 255), cv2.FILLED)
                      
        cv2.putText(frame, label, (start_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    ai_time = tm.getTimeMilli()
    cv2.putText(frame, '{:.2f}(ms)'.format(ai_time),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    # フレームを画面に描画
    cv2.imshow('Live', frame)

    if cv2.waitKey(1) >= 0:
        break
    time.sleep(args['interval'])

# 終了処理
print('終了処理...')
cv2.destroyAllWindows()
cap.release()
time.sleep(3)

