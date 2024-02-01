import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from controller import Robot

# Get robot's heading in degree based on compass values
def getRobotHeading(compassValue):
    rad = math.atan2(compassValue[1], compassValue[0])
    bearing = (rad - 1.5708) / math.pi * 180.0
    if bearing < 0.0:
        bearing = bearing + 360.0
    
    heading = 360 - bearing
    return heading

def sign(x):
    return (x > 0) - (x < 0)

def clamp(value, low, high):
    return max(low, min(value, high))

model = load_model('CNN.h5')

def predictType(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold=100)
    keypoints = fast.detect(gray, None)
    topLeftCorner = (0, 0)
    nearestKeypoint = min(keypoints, key=lambda point: np.sqrt((point.pt[0] - topLeftCorner[0]) ** 2 + (point.pt[1] - topLeftCorner[1]) ** 2))
    xCorner, yCorner = nearestKeypoint.pt
    xCorner = int(xCorner)
    yCorner = int(yCorner)
    cropped = gray[yCorner:yCorner+86, xCorner:xCorner+86]
    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[0]):
            if cropped[i][j] < 90:
                cropped[i][j] = 0
    imageRe = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_NEAREST)
    reshapedMatrix = imageRe.reshape((1, 28, 28))
    predictions = model.predict(reshapedMatrix.astype('float32') / 255.0)
    predictedClass = np.argmax(predictions)
    return predictedClass

# Create an instance of robot
robot = Robot()

# get the time step of the current world.
TIME_STEP = int(robot.getBasicTimeStep())

# Load Devices such as sensors
camera = robot.getDevice("camera")
cameraRollMotor = robot.getDevice("camera roll")
cameraPitchMotor = robot.getDevice("camera pitch")
frontLeftLed = robot.getDevice("front left led")
frontRightLed = robot.getDevice("front right led")
imu = robot.getDevice("inertial unit")
gps = robot.getDevice("gps")
compass = robot.getDevice("compass")
gyro = robot.getDevice("gyro")
frontLeftMotor = robot.getDevice("front left propeller")
frontRightMotor = robot.getDevice("front right propeller")
rearLeftMotor = robot.getDevice("rear left propeller")
rearRightMotor = robot.getDevice("rear right propeller")
motors = [frontLeftMotor, frontRightMotor, rearLeftMotor, rearRightMotor]

# Enables the devices
camera.enable(TIME_STEP)
imu.enable(TIME_STEP)
gps.enable(TIME_STEP)
compass.enable(TIME_STEP)
gyro.enable(TIME_STEP)

for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(1.0)

while robot.step(TIME_STEP) != -1:
    if robot.getTime() > 1.0:
        break

# General parameters
kVerticalThrust = 68.5
kVerticalOffset = 0.6
kVerticalP = 3.0
kRollP = 50.0
kPitchP = 30.0

targetAltitude = 4
destinations = [(-3, -2), (3, -3), (5, 0), (2, 5), (-5, 4)]
finalDestination = destinations.pop(0)
toAngle = 0
mode = 'YAngle'
isX = False
stepsAfterFinish = 0

boxTypes = {0: 'T-Shirt', 1:'Pants', 2:'Sweater', 3:'Shoe', 4:'Bag'}
# Select the type
chosenType = 3
print('Chosen type is:', boxTypes[chosenType])

while robot.step(TIME_STEP) != -1:

    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    altitude = gps.getValues()[2]
    rollVelocity = gyro.getValues()[0]
    pitchVelocity = gyro.getValues()[1]

    cameraRollMotor.setPosition(-0.115 * rollVelocity)
    cameraPitchMotor.setPosition(-0.1 * pitchVelocity)

    rollDisturbance = 0.0
    pitchDisturbance = 0.0
    yawDisturbance = 0.0

    x, y, z = gps.getValues()

    if mode == 'XAngle':
        if x - finalDestination[0] > 0.1:
            toAngle = 180
        else:
            toAngle = 0
        mode = 'ToAngle'
        isX = True

    if mode == 'YAngle':
        if y - finalDestination[1] > 0.1:
            toAngle = 270
        else:
            toAngle = 90
        mode = 'ToAngle'
    
    if mode == 'ToAngle':
        robotHeading = getRobotHeading(compass.getValues())
        if toAngle == 0:
            condition = abs(robotHeading - toAngle) < 1.0 or abs(robotHeading - 360) < 1.0
        else:
            condition = abs(robotHeading - toAngle) < 1.0
        if condition:
            if isX:
                mode = 'moveToX'
            else:
                mode = 'moveToY'
        else:
            if toAngle == 0:
                if robotHeading > 180:
                    yawDisturbance = 0.7
                else:
                    yawDisturbance = -0.7
            else:
                yawDisturbance = 0.7 * sign(toAngle - robotHeading)
    
    if mode == 'moveToY':
        robotHeading = getRobotHeading(compass.getValues())
        if abs(robotHeading - toAngle) > 1.0:
            mode = 'ToAngle'
        else :
            if abs(y - finalDestination[1]) < 0.1:
                mode = 'XAngle'
            else:
                pitchDisturbance = -2.0

    if mode == 'moveToX':
        robotHeading = getRobotHeading(compass.getValues())
        if abs(robotHeading - toAngle) > 1.0:
            mode = 'ToAngle'
        else :
            if abs(x - finalDestination[0]) < 0.1:
                toAngle = 180
                mode = 'to180'
            else:
                pitchDisturbance = -2.0

    if mode == 'to180':
        robotHeading = getRobotHeading(compass.getValues())
        condition = abs(robotHeading - toAngle) < 1.0
        if condition:
            mode = 'takePhoto'
        else:
            yawDisturbance = 0.7 * sign(toAngle - robotHeading)

    if mode == 'takePhoto':
        img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        predictedClass = predictType(img)
        if predictedClass == chosenType:
            frontLeftLed.set(1)
            frontRightLed.set(1)
            mode = 'land'
            pitchDisturbance = -2.0
        else:
            finalDestination = destinations.pop(0)
            mode = 'YAngle'
            isX = False

    if mode == 'land':
        if stepsAfterFinish < 70:
            pitchDisturbance = -2.0
        elif stepsAfterFinish > 400:
            if targetAltitude > 0.005:
                targetAltitude -= 0.005
            else:
                targetAltitude = 0
                mode = 'finish'
        stepsAfterFinish += 1

    if mode == 'finish':
        if stepsAfterFinish >= 1500:
            frontLeftMotor.setVelocity(0)
            frontRightMotor.setVelocity(0)
            rearLeftMotor.setVelocity(0)
            rearRightMotor.setVelocity(0)
        if stepsAfterFinish >= 2400:
            break
        stepsAfterFinish += 1

    if stepsAfterFinish < 2000:
        rollInput = kRollP * clamp(roll, -1.0, 1.0) + rollVelocity + rollDisturbance
        pitchInput = kPitchP * clamp(pitch, -1.0, 1.0) + pitchVelocity + pitchDisturbance
        yawInput = yawDisturbance
        clampedDifferenceAltitude = clamp(targetAltitude - altitude + kVerticalOffset, -1.0, 1.0)
        verticalInput = kVerticalP * pow(clampedDifferenceAltitude, 3.0)

        frontLeftMotorInput = kVerticalThrust + verticalInput - rollInput + pitchInput - yawInput
        frontRightMotorInput = kVerticalThrust + verticalInput + rollInput + pitchInput + yawInput
        rearLeftMotorInput = kVerticalThrust + verticalInput - rollInput - pitchInput + yawInput
        rearRightMotorInput = kVerticalThrust + verticalInput + rollInput - pitchInput - yawInput

        frontLeftMotor.setVelocity(frontLeftMotorInput)
        frontRightMotor.setVelocity(-frontRightMotorInput)
        rearLeftMotor.setVelocity(-rearLeftMotorInput)
        rearRightMotor.setVelocity(rearRightMotorInput)