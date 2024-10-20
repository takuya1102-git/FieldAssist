#Pythonプログラムを作成します。
#$ sudo nano client2.py


import socket

# サーバーのアドレスとポート番号
SERVER_ADDRESS = '0.0.0.0'
SERVER_PORT = 8080

# ソケットを作成
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # サーバーに接続
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    # 接続成功時の処理
except ConnectionRefusedError:
    print("接続に失敗しました。サーバーが起動していないか、ポートが閉じています。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:
    client_socket.close()

#保存（Ctrl-o 、リターン、 Ctrl-x)
#	↓
#$ python3 client2.py
