import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
import platform

# Global variables
ser = None

def list_available_ports():
    """List all available serial ports on the system"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Found port: {port.device}")
        print(f"Description: {port.description}")
        print(f"Hardware ID: {port.hwid}")
        print("---")
    return ports

def find_silicon_labs_port():
    """Find the Silicon Labs CP210x port"""
    ports = list_available_ports()
    for port in ports:
        if "CP210x" in port.description:
            return port.device
    return None

def get_port_name():
    """Get the appropriate port name based on the operating system"""
    system = platform.system()
    
    if system == "Windows":
        return 'COM1'
    elif system == "Linux":
        return '/dev/ttyUSB0'  # Common name for USB-Serial devices on Linux
    else:
        return None

def initialize_serial():
    global ser
    try:
        if ser is not None and ser.is_open:
            ser.close()
            ser = None
            
        # Try to find the correct port
        port = find_silicon_labs_port() or get_port_name()
        if not port:
            print("UART Debug: No suitable port found")
            return False
            
        print(f"UART Debug: Attempting to connect to {port}...")
        ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=1
        )
        
        if ser.is_open:
            print(f"UART Debug: Successfully connected to {port}")
            time.sleep(2)  # Wait for Arduino to initialize
            return True
        return False
            
    except serial.SerialException as e:
        print(f"UART Debug: Serial Error: {e}")
        return False
    except Exception as e:
        print(f"UART Debug: Unexpected error: {e}")
        return False

def send_to_serial(command):
    global ser
    try:
        if not ser or not ser.is_open:
            print("UART Debug: Attempting to reconnect...")
            if not initialize_serial():
                print("UART Debug: Could not establish connection")
                return False
        
        print(f"UART Debug: Sending command {command}")
        ser.write(command.encode('ascii'))
        ser.flush()
        print(f"UART Debug: Command {command} sent successfully")
        
        time.sleep(0.1)
        if ser.in_waiting:
            response = ser.read(ser.in_waiting)
            response_str = response.decode('ascii', errors='ignore').strip()
            print(f"UART Debug: Received response: {response_str}")
        return True
            
    except Exception as e:
        print(f"UART Debug: Error in send_to_serial: {e}")
        return False

def detect_objects():
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Load COCO class labels
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Open the webcam
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 120) 
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Processing camera feed...")
        return [], None
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Create a blob from the frame and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Extract class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                class_ids.append(class_id)
    
    return class_ids, frame

def main():
    print("UART Debug: Starting application...")
    print(f"Running on: {platform.system()}")
    
    # List all available ports
    print("Available ports:")
    ports = list_available_ports()
    
    # Initialize serial connection
    retry_count = 3
    while retry_count > 0:
        if initialize_serial():
            print("UART Debug: Initial serial connection successful")
            break
        else:
            print(f"UART Debug: Failed to establish initial serial connection, retries left: {retry_count-1}")
            retry_count -= 1
            time.sleep(1)
    
    if not ser or not ser.is_open:
        print("Warning: Starting without serial connection. Will try to connect when needed.")

    while True:
        # Detect objects and get the frame
        class_ids, frame = detect_objects()
        
        # Initialize variables to track if human, animal, or bird is detected
        bicycle_detected = False
        car_detected = False
        motorbike_detected = False
        
        # Determine if humans, animals, or birds are detected
        for class_id in class_ids:
            if class_id == 1:
                bicycle_detected = True
            elif class_id == 2:
                car_detected = True
            elif class_id == 3:
                motorbike_detected = True
        
        # Send detection messages through UART
        if bicycle_detected:
            print("Bicycle detected. Sending 2")
            send_to_serial('2')
        if car_detected:
            print("Car detected. Sending 1")
            send_to_serial('1')
        if motorbike_detected:
            print("Motorbike detected. Sending 3")
            send_to_serial('3')
        if not (bicycle_detected or car_detected or motorbike_detected):
            print("No bicycle, car, or bike detected.")
            #send_to_serial('0')
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Wait for 1 second or until a key is pressed
        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break
    
    # Cleanup
    if ser is not None and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

