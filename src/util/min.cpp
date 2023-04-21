#include <opencv2/opencv.hpp>

int main() {
    cv::namedWindow("Test Window", cv::WINDOW_NORMAL);
    cv::Mat test_image = cv::imread("image.jpg");
    cv::imshow("Test Window", test_image);
    cv::waitKey(0);
    return 0;
}
