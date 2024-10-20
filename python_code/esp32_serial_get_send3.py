import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
esp32_data = 0

while True:
	esp32_data = ser.readline()
	print(esp32_data)
	if(esp32_data > 0):
		print("get end")
		ser.write(b'A')
		print("send end")
        esp32_data = 0#初期値に戻す。v3
        print("servo ON")#位置変更
ser.close()



