#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#include <iomanip>
#include <thread>
#include "/repos/darknet/include/darknet.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <zmq.hpp>

struct FrameData
{
    cv::Mat frame;
    std::vector<cv::Rect> bounding_boxes;
    std::string class_name;
    float confidence;
    std::time_t timestamp;
};

bool detect_objects(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &class_names, const std::set<int> &desired_class_ids, zmq::socket_t &event_socket, std::vector<FrameData> &detected_events, std::vector<int> &class_ids, std::vector<float> &confidences, std::vector<cv::Rect> &boxes);
void send_detection_event(zmq::socket_t &socket, const FrameData &event);

int main(int argc, char **argv)
{
    bool use_webcam = true;                            // Default to ZeroMQ stream
    std::set<int> desired_class_ids = {0, 14, 15, 16}; // Only detect these classes (person, bird, cat, dog)

    if (argc > 1 && strcmp(argv[1], "webcam") == 0)
    {
        use_webcam = true; // Use webcam if the first argument is "webcam"
    }

    std::vector<FrameData> detected_events;

    // Setup the model path dynamically
    fs::path exe_path = fs::current_path();
    fs::path model_path = exe_path / "models";

    fs::path cfg_path = model_path / "yolov7-tiny.cfg";
    fs::path weights_path = model_path / "yolov7-tiny.weights";
    fs::path coco_path = model_path / "coco.names";

    // Load the pre-trained YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg_path.string(), weights_path.string());
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::vector<std::string> class_names;
    std::ifstream coco_file(coco_path.string());
    if (coco_file.is_open())
    {
        std::string line;
        while (std::getline(coco_file, line))
        {
            class_names.push_back(line);
        }
        coco_file.close();
    }

    // Initialize ZeroMQ context and socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);

    // Open the video feed
    cv::VideoCapture cap;
    if (use_webcam)
    {
        cap.open(0); // Capture from the webcam
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Frame size : " << frame_width << " x " << frame_height << std::endl;
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);  // set the width to 640 pixels
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480); // set the height to 480 pixels
        if (!cap.isOpened())
        {
            std::cerr << "Failed to open the video device" << std::endl;
            return 1;
        }
    }
    else
    {
        socket.connect("tcp://172.25.64.1:5555");
        socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    }

    // Set up ZeroMQ socket for sending detection events
    zmq::socket_t event_socket(context, ZMQ_PUB);
    event_socket.bind("tcp://*:5556");

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (!use_webcam)
        {
            zmq::message_t message;
            socket.recv(&message);
            std::vector<uchar> data(message.data<unsigned char>(), message.data<unsigned char>() + message.size());
            cv::Mat buf(1, data.size(), CV_8UC1, data.data());
            frame = cv::imdecode(buf, 1);
        }

        if (frame.empty())
        {
            std::cerr << "Empty frame received" << std::endl;
            break;
        }

        // Clear the last detection results
        class_ids.clear();
        confidences.clear();
        boxes.clear();

        // Process detected objects in our detection desired_class_ids
        if (detect_objects(frame, net, class_names, desired_class_ids, event_socket, detected_events, class_ids, confidences, boxes))
        {
            // Multiple events may be detected in a single frame
            for (const FrameData &event : detected_events)
            {
                // Send the event over ZeroMQ
                send_detection_event(event_socket, event);
            }
        }
    }

    cap.release();
    event_socket.close();
    context.close();

    return 0;
}

std::string base64_encode(const unsigned char *src, size_t len)
{
    static const char *base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len;)
    {
        uint32_t octet_a = i < len ? src[i++] : 0;
        uint32_t octet_b = i < len ? src[i++] : 0;
        uint32_t octet_c = i < len ? src[i++] : 0;
        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        result.push_back(base64_chars[(triple >> 3 * 6) & 0x3F]);
        result.push_back(base64_chars[(triple >> 2 * 6) & 0x3F]);
        result.push_back(base64_chars[(triple >> 1 * 6) & 0x3F]);
        result.push_back(base64_chars[(triple >> 0 * 6) & 0x3F]);
    }

    if (len % 3 != 0)
    {
        result[result.size() - 1] = '=';
        if (len % 3 == 1)
        {
            result[result.size() - 2] = '=';
        }
    }

    return result;
}

void send_detection_event(zmq::socket_t &socket, const FrameData &event)
{
    // Serialize the frame as a JPEG image
    std::vector<uchar> buffer;
    cv::imencode(".jpg", event.frame, buffer);

    // Create a JSON object to hold the event data
    std::stringstream json_data;
    json_data << R"({"class_name":")" << event.class_name << R"(",)";
    json_data << R"("bounding_boxes":[)";
    for (size_t i = 0; i < event.bounding_boxes.size(); ++i)
    {
        const cv::Rect &box = event.bounding_boxes[i];
        json_data << R"({"x":)" << box.x << R"(,"y":)" << box.y << R"(,"width":)" << box.width << R"(,"height":)" << box.height << R"(})";
        if (i < event.bounding_boxes.size() - 1)
        {
            json_data << ",";
        }
    }
    json_data << R"(],)";
    json_data << R"("confidence":)" << event.confidence << R"(,)";
    json_data << R"("timestamp":)" << event.timestamp << R"(,)";
    json_data << R"("frame":")" << base64_encode(buffer.data(), buffer.size()) << R"("})";

    // Send the JSON object as a string over ZeroMQ
    zmq::message_t zmq_message(json_data.str().c_str(), json_data.str().size());
    socket.send(zmq_message);
}

bool detect_objects(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &class_names, const std::set<int> &desired_class_ids, zmq::socket_t &event_socket, std::vector<FrameData> &detected_events, std::vector<int> &class_ids, std::vector<float> &confidences, std::vector<cv::Rect> &boxes)
{
    bool draw_boxes = true;
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> output_layers;
    net.forward(output_layers, net.getUnconnectedOutLayersNames());

    for (auto &output_layer : output_layers)
    {
        float *data = (float *)output_layer.data;
        for (int j = 0; j < output_layer.rows; ++j, data += output_layer.cols)
        {
            cv::Mat scores = output_layer.row(j).colRange(5, output_layer.cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);

            if (confidence > 0.5)
            {
                if (desired_class_ids.count(class_id_point.x) > 0)
                {
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
    for (int index : indices)
    {
        cv::Rect box = boxes[index];
        int class_id = class_ids[index];
        std::string label = class_names[class_id] + ": " + std::to_string(confidences[index]);

        if (desired_class_ids.count(class_id) > 0)
        {
            FrameData event{frame.clone(), {box}, class_names[class_id], confidences[index], std::time(nullptr)};
            detected_events.push_back(event);
            object_detected = true;
        }
    }

    return object_detected;
}
