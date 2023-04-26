#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <nlohmann/json.hpp>

struct DetectionEvent {
    cv::Mat frame;
    std::string class_name;
    float confidence;
    std::time_t timestamp;
    std::string video_clip_path;
};

bool receive_detection_event(zmq::socket_t& socket, DetectionEvent& event);
void upload_to_s3(const DetectionEvent& event, const std::string& bucket_name);

int main(int argc, char* argv[]) {
    Aws::SDKOptions options;
    Aws::InitAPI(options);
    bool show_frames = true;
    if (argc > 1 && std::string(argv[1]) == "--debug") {
        show_frames = true;
    }

    // Set up ZeroMQ context and socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);
    socket.connect("tcp://localhost:5556");
    socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    if (show_frames) {
        cv::namedWindow("Received Frame", cv::WINDOW_NORMAL);
    }

    DetectionEvent event;
    while (true) {
        if (receive_detection_event(socket, event)) {
            std::cout << "Detection event:" << std::endl
                      << "  Class: " << event.class_name << std::endl
                      << "  Confidence: " << event.confidence << std::endl
                      << "  Timestamp: " << std::asctime(std::localtime(&event.timestamp))
                      << "  Video clip path: " << event.video_clip_path << std::endl;

            // Upload video clip to S3
            upload_to_s3(event, "wc1_training_data");

            if (show_frames) {
                if (!event.frame.empty()) {
                    cv::imshow("Received Frame", event.frame);
                    int key = cv::waitKey(1) & 0xFF;
                    if (key == 'q' || key == 'Q') {
                        break;
                    }
                } else {
                    std::cerr << "Warning: Received empty frame" << std::endl;
                }
            }        
        }
    }

    if (show_frames) {
        cv::destroyAllWindows();
    }

    Aws::ShutdownAPI(options);

    return 0;
}

std::vector<uchar> decode_base64(const std::string& base64_data) {
    using namespace boost::archive::iterators;
    typedef transform_width<binary_from_base64<const char*>, 8, 6> base64_decoder;
    std::vector<uchar> decoded_data(base64_decoder(base64_data.data()), base64_decoder(base64_data.data() + base64_data.size()));
    return decoded_data;
}

bool receive_detection_event(zmq::socket_t& socket, DetectionEvent& event) {
    zmq::message_t message;

    // Receive JSON message
    if (!socket.recv(message, zmq::recv_flags::dontwait)) {
        return false;
    }
    std::string json_data(message.data<char>(), message.size());
    nlohmann::json json_object = nlohmann::json::parse(json_data);

    // Decode frame
    std::string frame_base64_data = json_object["frame"];
    std::vector<uchar> decoded_frame_data = decode_base64(frame_base64_data);
    cv::Mat buf(1, decoded_frame_data.size(), CV_8UC1, decoded_frame_data.data());
    event.frame = cv::imdecode(buf, 1);

    // Parse other properties
    event.class_name = json_object["class_name"];
    event.confidence = json_object["confidence"];
    event.timestamp = json_object["timestamp"];
    event.video_clip_path = json_object["video_clip_path"];

    return true;
}

void upload_to_s3(const DetectionEvent& event, const std::string& bucket_name) {
    Aws::S3::S3Client s3_client;

    std::string key = "video_clips/" + event.class_name + "_" + std::to_string(event.timestamp) + ".mp4";

    Aws::S3::Model::PutObjectRequest put_request;
    put_request.SetBucket(bucket_name.c_str());
    put_request.SetKey(key.c_str());

    std::shared_ptr<Aws::IOStream> input_stream = Aws::MakeShared<Aws::FStream>("ALLOC_TAG", event.video_clip_path.c_str(), std::ios_base::in | std::ios_base::binary);
    put_request.SetBody(input_stream);

    auto put_object_outcome = s3_client.PutObject(put_request);

    if (put_object_outcome.IsSuccess()) {
        std::cout << "Successfully uploaded video clip to S3 bucket: " << bucket_name << std::endl;
    } else {
        std::cout << "Error uploading video clip to S3 bucket: " << put_object_outcome.GetError().GetMessage() << std::endl;
    }
}
