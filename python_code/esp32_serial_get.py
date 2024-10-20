import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)

while True:
	esp32_data = ser.readline()
	print(esp32_data)
ser.close()


