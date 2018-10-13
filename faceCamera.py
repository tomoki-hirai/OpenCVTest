import cv2
import requests
import json
from time import sleep

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps
    ROOM_ID = 0 #ルーム指定
    USER_ID = 1 #ユーザ指定
    USER_NAME = "riku" #ユーザ名の指定

    ORG_WINDOW_NAME = "org"
    #GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    speaking_counter = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_default.xml"
    mouse_cascade_file = "haarcascade_mcs_mouth.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    mouse_cascade = cv2.CascadeClassifier(mouse_cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    #cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    #会議名とユーザ名の取得(json_dict[num]:numの数値でユーザー名の変更)
    headers = {"content-type": "application/json"}
    # json_str_room = requests.get('http://ec2-13-231-238-116.ap-northeast-1.compute.amazonaws.com:3000/meeting/room', headers=headers)
    # json_dict_room = json_str_room.json()
    # print(json_str_room)
    # json_str_user = requests.get('http://ec2-13-231-238-116.ap-northeast-1.compute.amazonaws.com:3000/users', headers=headers)
    #print(json_str_user)
    # json_dict_user = json_str_user.json()

    #起動時にuser_idとuser_nameをサーバに送信
    correct_USER_ID = USER_ID + 1
    # response = requests.post('http://ec2-13-231-238-116.ap-northeast-1.compute.amazonaws.com:3000/users', data={'user_id':correct_USER_ID, 'user_name':USER_NAME})

    # 変換処理ループ
    while end_flag == True:
        # 画像の取得と顔の検出
        img = c_frame
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        #文字の出力（会議名，ユーザ名）
        # cv2.rectangle(img, (0, 0), (350, 280), (255, 255, 255), thickness=-1)
        # cv2.putText(img, "aaa", (5,40), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0, 0, 0))
        # cv2.line(img,(2,56),(345,56),(0,0,0),2)
        # cv2.putText(img, '--participant--', (15,95), cv2.FONT_HERSHEY_COMPLEX, 1.0,(0, 0, 0))
        # cv2.putText(img, "a", (15,140), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0, 0, 255))
        # cv2.putText(img, "a", (15,180), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0, 0, 0))
        # cv2.putText(img, "a", (15,220), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0, 0, 0))
        # cv2.putText(img, "a", (15,260), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0, 0, 0))

        # 検出した顔に印を付ける
        for (x, y, w, h) in face_list:
            color = (0, 0, 225)
            pen_w = 3
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)

            #検出した口に印をつける
            face_lower = int(h/2)
            mouse_color = img[x:x+w, y+face_lower:y+h]
            mouse_gray = cv2.cvtColor(mouse_color, cv2.COLOR_BGR2GRAY)
            mouse = mouse_cascade.detectMultiScale(mouse_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 80))
            for (mx, my, mw, mh) in mouse:
                cv2.rectangle(mouse_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)

                # 口だけ切り出して二値化画像を保存
                threshold = 40 #二値化の閾値の設定
                for num, rect in enumerate(mouse):
                    x = rect[0]
                    y = rect[1]
                    width = rect[2]
                    height = rect[3]
                    dst = mouse_gray[y:y + height, x:x + width]
                    #二値化処理
                    ret,img_threshold = cv2.threshold(dst,threshold,255,cv2.THRESH_BINARY)
                    cv2.imwrite("./output/img_threshold.jpg",img_threshold)
                    binary_img = cv2.imread("./output/img_threshold.jpg")
                    cnt =0 #黒色領域の数を格納する変数
                    for val in binary_img.flat:
                        if val == 0:
                            cnt += 1
                    cv2.waitKey(1)

                    #二値化画像の黒色領域が何箇所あるかの判断→口が開いていれば黒いろ領域が多くなる＝発言している
                    if cnt > 600:
                        #print(speaking_counter, "Speaking!!")
                        speaking_counter += 1
                    #発言カウンターが5個貯まれば，サーバに秒数の送信
                    if speaking_counter == 5:
                        print("You spoke ", speaking_counter * 0.21, "second")
                        # response = requests.post('http://ec2-13-231-238-116.ap-northeast-1.compute.amazonaws.com:3000/meeting/speaking', data={'user_id':correct_USER_ID, 'start': speaking_counter * 0.21, 'end': 0, 'room_id':ROOM_ID})
                        #response = requests.post('http://requestbin.fullcontact.com/xxex5cxx', data={'user_id':USER_ID, 'start': speaking_counter * 0.21})
                        speaking_counter = 0
                    else:
                        continue

        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        #cv2.imshow(GAUSSIAN_WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
