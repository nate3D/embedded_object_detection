#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <zmq.hpp>

void detect_objects(cv::Mat& frame, cv::dnn::Net& net, const std::vector<std::string>& class_names, bool draw_boxes, const std::set<int>& desired_class_ids);
void draw_detection_box(cv::Mat& frame, int left, int top, int right, int bottom, const std::string& label);

int main(int argc, char** argv) {
    bool use_webcam = false; // Default to ZeroMQ stream
    std::set<int> desired_class_ids = {0, 14, 15, 16}; // Only detect these classes (person, bird, cat, dog)
    int fps = 60; // Desired frames per second to process for detection

    if (argc > 1 && strcmp(argv[1], "webcam") == 0) {
        use_webcam = true; // Use webcam if the first argument is "webcam"
    }

    // Setup timer for FPS
    std::chrono::milliseconds frame_interval(1000 / fps);
    std::chrono::steady_clock::time_point next_frame_time = std::chrono::steady_clock::now();

    // Setup the model path dynamically
    std::filesystem::path exe_path = std::filesystem::current_path();
    std::filesystem::path model_path = exe_path / "models";

    std::filesystem::path cfg_path = model_path / "yolov7-tiny.cfg";
    std::filesystem::path weights_path = model_path / "yolov7-tiny.weights";
    std::filesystem::path coco_path = model_path / "coco.names";

    // Load the pre-trained YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg_path.string(), weights_path.string());
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::vector<std::string> class_names;
    std::ifstream coco_file(coco_path.string());
    if (coco_file.is_open()) {
        std::string line;
        while (std::getline(coco_file, line)) {
            class_names.push_back(line);
        }
        coco_file.close();
    }

    // Open the video feed
    cv::VideoCapture cap;
    if (use_webcam) {
        cap.open(0); // Capture from the webcam
        if (!cap.isOpened()) {
            std::cerr << "Failed to open the video device" << std::endl;
            return 1;
        }
    }

    // Set up ZeroMQ context and socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);
    socket.connect("tcp://172.25.64.1:5555");
    socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    cv::namedWindow("Object Detection", cv::WINDOW_NORMAL);

    bool draw_boxes = true;

    while (true) {
        cv::Mat frame;
        if (use_webcam) {
            cap >> frame;
        } else {
            zmq::message_t message;
            socket.recv(&message);

            std::vector<uchar> data(message.data<unsigned char>(), message.data<unsigned char>() + message.size());
            cv::Mat buf(1, data.size(), CV_8UC1, data.data());
            frame = cv::imdecode(buf, 1);
        }

        if (frame.empty()) {
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= next_frame_time) {
            detect_objects(frame, net, class_names, draw_boxes, desired_class_ids);
            cv::imshow("Object Detection", frame);
            next_frame_time = now + frame_interval;
        }
        
        // Exit the loop if the user presses 'q'
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    if (use_webcam) {
        cap.release();
    }

    cv::destroyAllWindows();

    return 0;
}

void detect_objects(cv::Mat& frame, cv::dnn::Net& net, const std::vector<std::string>& class_names, bool draw_boxes, const std::set<int>& desired_class_ids) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> output_layers;
    net.forward(output_layers, net.getUnconnectedOutLayersNames());

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (auto& output_layer : output_layers) {
        float* data = (float*)output_layer.data;
        for (int j = 0; j < output_layer.rows; ++j, data += output_layer.cols) {
            cv::Mat scores = output_layer.row(j).colRange(5, output_layer.cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);

            if (confidence > 0.5) {
                // Check if the detected class ID is in the desired_class_ids set
                if (desired_class_ids.count(class_id_point.x) > 0) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    class_ids.push_back(class_id_point.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    for (int index : indices) {
        cv::Rect box = boxes[index];
        int class_id = class_ids[index];
        std::string label = class_names[class_id] + ": " + std::to_string(confidences[index]);

        if (draw_boxes) {
            draw_detection_box(frame, box.x, box.y, box.x + box.width, box.y + box.height, label);
        }
    }
}

void draw_detection_box(cv::Mat& frame, int left, int top, int right, int bottom, const std::string& label) {
    cv::rectangle(frame, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}
