import zmq
import json

def main():
    # Prepare the ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Connect to the publisher socket
    socket.connect("tcp://localhost:5556")  # Replace with your server's address
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    while True:
        # Receive a message from the ZeroMQ socket
        message = socket.recv_string()
        
        # Parse the JSON data
        event_data = json.loads(message)
        
        # Print out the event details (excluding the frame data)
        print(f"Class Name: {event_data['class_name']}")
        print(f"Bounding Boxes: {event_data['bounding_boxes']}")
        print(f"Confidence: {event_data['confidence']}")
        print(f"Timestamp: {event_data['timestamp']}")
        print("------------------------------------------------")

if __name__ == "__main__":
    main()

