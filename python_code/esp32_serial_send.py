import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)

for i in range(3):
	ser.write(b'A')
	print("LED ON")
	time.sleep(0.5)
	ser.write(b'B')
	print("LED OFF")
	time.sleep(0.5)

ser.close()

print("blink end")


