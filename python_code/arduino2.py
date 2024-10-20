import serial
import time

# シリアルポートの設定 (ポート番号は適宜変更)
ser = serial.Serial('COM12', 115200)

while True:
    # データの有無を確認
    if ser.in_waiting > 0:
        # データを読み込む
        data = ser.readline().decode().rstrip()

        # データの表示 (デバッグ用)
        print(f"Received: {data}")

        # データの処理 (数値に変換する場合はtry-exceptでエラー処理)
        try:
            value = float(data)
            print(f"Converted value: {value}")
        except ValueError:
            print("Invalid data format")

    # 一定時間待つ (処理負荷を軽減)
    time.sleep(0.1)

# シリアルポートを閉じる
ser.close()
