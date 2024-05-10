import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt
 

sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 100
speed_data = []
time_data = []
start_time = None
plot_saved = False  # A flag to ensure the graph is saved only once
model_name = 'two_tracks_with_data_aug_no_drop_out'

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    global start_time, speed_data, time_data, plot_saved, model_name
    if start_time is None:
        start_time = time.time()
    
    speed = float(data['speed'])
    current_time = time.time() - start_time

    # Collect speed and time data
    speed_data.append(speed)
    time_data.append(current_time)

    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    
    # Check if the time has exceeded 30 seconds
    if current_time >= 30 and not plot_saved:
        plot_saved = True
        plot_speed_time_graph()  

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

def plot_speed_time_graph():
    global model_name
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, speed_data, label='Speed (mph) Over Time')
    plt.axhline(y=30, color='r', linestyle='--', label='Max Speed Limit (30 mph)')
    
    # Annotating possible behaviors
    plt.annotate('Speed Limit', xy=(max(time_data)*0.8, 30), xytext=(max(time_data)*0.8, 35),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    # Check for slow down due to obstacle
    slowed_down_times = [time_data[i] for i in range(150, len(speed_data)) if speed_data[i] < 29 and speed_data[i] > 0]
    if slowed_down_times:
        plt.annotate('Slowing down (Possible Obstacle)', 
                     xy=(slowed_down_times[0], speed_data[time_data.index(slowed_down_times[0])]),
                     xytext=(slowed_down_times[0], speed_data[time_data.index(slowed_down_times[0])] + 10),
                     arrowprops=dict(facecolor='orange', shrink=0.05))
    
    # Check for complete halt
    halted_times = [time_data[i] for i in range(100, len(speed_data)) if speed_data[i] <= 0.5]
    if halted_times:
        plt.annotate('Complete Halt (Obstruction)', 
                     xy=(halted_times[0], speed_data[time_data.index(halted_times[0])]),
                     xytext=(halted_times[0], 10),
                     arrowprops=dict(facecolor='blue', shrink=0.05))
        
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speed (miles per hour)')
    plt.title('Speed vs. Time Graph')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_fig/' + str(model_name) + '.png')  # Save the graph to a file
    plt.show()
    

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
    
if __name__ == '__main__':
    model = load_model('model/' + model_name + '.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)