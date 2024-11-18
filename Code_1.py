from pyfirmata import Arduino, SERVO
from time import sleep
import pandas as pd
from statistics import mean, stdev
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define the port and pin for the servo
port = 'COM3'  # Change this to your actual port
pin = 10       # Pin number where the servo is connected

# Initialize the board and configure the servo pin
board = Arduino(port)
board.digital[pin].mode = SERVO

# Function to rotate the servo to a specified angle
def rotateservo(pin, angle):
    board.digital[pin].write(angle)
    sleep(0.015)  # Wait for the servo to move to the desired angle

# Load the CSV data
df = pd.read_csv('user.csv')

# Function to get features from the data
def getChannelFeatures(af3):
    means = []
    std = []
    powers = []
    
    for i in range(1, 91):
        means.append(mean(af3[(i-1)*128:(i*128)]))
        std.append(stdev(af3[(i-1)*128:(i*128)]))
        mag = af3[(i-1)*128:(i*128)]
        arr_sqrd = np.power(mag, 2)
        signal_power = np.sum(np.abs(arr_sqrd)) / len(mag)
        powers.append(signal_power)
    
    return means, std, powers

# Extract features for all channels
channels = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
channels_data = []
header = []

for ch in channels:
    m, s, p = getChannelFeatures(df[ch])
    channels_data.extend([m, s, p])
    header.append(ch + '_MEAN')
    header.append(ch + '_STD')
    header.append(ch + '_POWER')

# Create a DataFrame for the new data
df_n = pd.DataFrame(channels_data)
df_new = df_n.transpose()
df_new.columns = header

# Assign classes
class_arr = []
for u in range(3):
    for i in range(0, 3):
        for j in range(10):
            class_arr.append(i)

df_new['Class'] = class_arr

# Split the data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(df_new.drop(['Class'], axis='columns'), df_new.Class, test_size=0.2)

# Train the model
model = RandomForestClassifier(n_estimators=130)
model.fit(X_train, Y_train)

# Make predictions
y_tree_predicts = model.predict(X_test)

# Extract the first 10 predictions
first_10_predictions = y_tree_predicts[:10]

# Initial position of the servo
current_angle = 90
rotateservo(pin, current_angle)
print(f"Initial position: {current_angle} degrees")

# Move the servo based on the first 10 predictions
for prediction in first_10_predictions:
    print(f"Prediction: {prediction}")  # Print each prediction before moving the servo
    
    if prediction == 0:
        print(f"Moving servo to the left from {current_angle} by 90 degrees...")  # Print movement direction
        new_angle = current_angle - 90
        if new_angle < 0:
            new_angle = 0  # Ensure the angle does not go below 0 degrees
        rotateservo(pin, new_angle)
        current_angle = new_angle

    elif prediction == 1:
        print(f"Moving servo to the right from {current_angle} by 90 degrees...")  # Print movement direction
        new_angle = current_angle + 90
        if new_angle > 180:
            new_angle = 180  # Ensure the angle does not exceed 180 degrees
        rotateservo(pin, new_angle)
        current_angle = new_angle

    elif prediction == 2:
        print(f"Servo remains at {current_angle} degrees. No movement.")  # Print no movement info
        # No update to current_angle

    else:
        print(f"Invalid prediction: {prediction}")

    sleep(2)  # Increased delay between movements to 2 seconds

# Output the result
print("Servo movements based on the first 10 predictions are complete.")
