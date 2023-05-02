#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <zmq.hpp>

struct DetectionEvent {
    cv::Mat frame;
    std::string class_name;
    float confidence;
    std::time_t timestamp;
    std::string video_clip_path;
};

bool detect_objects(cv::Mat& frame, cv::dnn::Net& net, const std::vector<std::string>& class_names, const std::set<int>& desired_class_ids, zmq::socket_t& event_socket, std::vector<DetectionEvent>& detected_events);
void draw_detection_box(cv::Mat& frame, int left, int top, int right, int bottom, const std::string& label);
void send_detection_event(zmq::socket_t& socket, const DetectionEvent& event);

int main(int argc, char** argv) {
    bool use_webcam = false; // Default to ZeroMQ stream
    bool is_recording = false;
    bool event_in_progress = false;

    std::set<int> desired_class_ids = {0, 14, 15, 16}; // Only detect these classes (person, bird, cat, dog)
    int fps = 10; // Desired frames per second to process for detection

    if (argc > 1 && strcmp(argv[1], "webcam") == 0) {
        use_webcam = true; // Use webcam if the first argument is "webcam"
    }

    // Setup timer for FPS
    std::chrono::milliseconds frame_interval(1000 / fps);
    std::chrono::steady_clock::time_point next_frame_time = std::chrono::steady_clock::now();

    // Setup timer for video clip timing
    std::chrono::steady_clock::time_point end_recording_time;
    cv::VideoWriter video_writer;
    std::chrono::steady_clock::time_point last_detection_time;
    std::vector<DetectionEvent> detected_events;

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

    // Initialize ZeroMQ context and socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);

    // Open the video feed
    cv::VideoCapture cap;
    if (use_webcam) {
        cap.open(0); // Capture from the webcam
        if (!cap.isOpened()) {
            std::cerr << "Failed to open the video device" << std::endl;
            return 1;
        }
    } else {
        socket.connect("tcp://172.25.64.1:5555");
        socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    }

    // Set up ZeroMQ socket for sending detection events
    zmq::socket_t event_socket(context, ZMQ_PUB);
    event_socket.bind("tcp://*:5556");


    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (!use_webcam) {
            zmq::message_t message;
            socket.recv(&message);

            std::vector<uchar> data(message.data<unsigned char>(), message.data<unsigned char>() + message.size());
            cv::Mat buf(1, data.size(), CV_8UC1, data.data());
            frame = cv::imdecode(buf, 1);
        }

        if (frame.empty()) {
            std::cerr << "Empty frame received" << std::endl;
            break;
        }

        // Limit the frame processing rate
        std::this_thread::sleep_until(next_frame_time);
        next_frame_time += frame_interval;

        // Detect objects
        bool detection_occurred = detect_objects(frame, net, class_names, desired_class_ids, event_socket, detected_events);

        // If an event is not in progress and a detection occurred, start recording
        if (!event_in_progress && detection_occurred) {
            event_in_progress = true;
            is_recording = true;
            end_recording_time = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            std::time_t timestamp = std::time(nullptr);
            std::stringstream filename_ss;
            filename_ss << "event_" << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S") << ".avi";
            std::string video_clip_path = "video_clips/" + filename_ss.str();
            video_writer.open(video_clip_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, frame.size());
        }

        // If an event is in progress, record the frame
        if (is_recording) {
            video_writer.write(frame);
        }

        // If an event is in progress and recording should stop, finalize and send events
        if (event_in_progress && std::chrono::steady_clock::now() >= end_recording_time) {
            event_in_progress = false;
            is_recording = false;
            video_writer.release();

            for (const DetectionEvent& event : detected_events) {
                send_detection_event(event_socket, event);
            }

            detected_events.clear();
        }
    }

    cap.release();
    event_socket.close();
    context.close();

    return 0;
}

std::string base64_encode(const unsigned char* src, size_t len) {
    static const char* base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len;) {
        uint32_t octet_a = i < len ? src[i++] : 0;
        uint32_t octet_b = i < len ? src[i++] : 0;
        uint32_t octet_c = i < len ? src[i++] : 0;
        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        result.push_back(base64_chars[(triple >> 3 * 6) & 0x3F]);
        result.push_back(base64_chars[(triple >> 2 * 6) & 0x3F]);
        result.push_back(base64_chars[(triple >> 1 * 6) & 0x3F]);
        result.push_back(base64_chars[(triple >> 0 * 6) & 0x3F]);
    }

    if (len % 3 != 0) {
        result[result.size() - 1] = '=';
        if (len % 3 == 1) {
            result[result.size() - 2] = '=';
        }
    }

    return result;
}

void send_detection_event(zmq::socket_t& socket, const DetectionEvent& event) {
    // Serialize the frame as a JPEG image
    std::vector<uchar> buffer;
    cv::imencode(".jpg", event.frame, buffer);

    // Create a JSON object to hold the event data
    std::stringstream json_data;
    json_data << R"({"class_name":")" << event.class_name << R"(",)";
    json_data << R"("confidence":)" << event.confidence << R"(,)";
    json_data << R"("timestamp":)" << event.timestamp << R"(,)";
    json_data << R"("video_clip_path":)" << event.video_clip_path << R"(,)";
    json_data << R"("frame":")" << base64_encode(buffer.data(), buffer.size()) << R"("})";

    // Send the JSON object as a string over ZeroMQ
    zmq::message_t zmq_message(json_data.str().c_str(), json_data.str().size());
    socket.send(zmq_message, zmq::send_flags::none);
}

bool detect_objects(cv::Mat& frame, cv::dnn::Net& net, const std::vector<std::string>& class_names, const std::set<int>& desired_class_ids, zmq::socket_t& event_socket, std::vector<DetectionEvent>& detected_events) {
    bool draw_boxes = true;
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

    bool object_detected = false;
    for (int index : indices) {
        cv::Rect box = boxes[index];
        int class_id = class_ids[index];
        std::string label = class_names[class_id] + ": " + std::to_string(confidences[index]);

        if (draw_boxes) {
            draw_detection_box(frame, box.x, box.y, box.x + box.width, box.y + box.height, label);
        }

        if (desired_class_ids.count(class_id) > 0) {
            DetectionEvent event{frame.clone(), class_names[class_id], confidences[index], std::time(nullptr), ""};
            detected_events.push_back(event);
            object_detected = true;
        }
    }

    return object_detected;
}

void draw_detection_box(cv::Mat& frame, int left, int top, int right, int bottom, const std::string& label) {
    cv::rectangle(frame, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}
