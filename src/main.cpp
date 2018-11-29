#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void canny(cv::Mat *frame, float threshold);

void usage();

int main(int argc, char **argv) {
    if (argc != 2) {
        usage();
        return -1;
    }

    cv::Mat frame;
    frame = imread(argv[1], cv::IMREAD_COLOR);
    if (!frame.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    canny(&frame, 200.0);

    namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    imshow("Display window", frame);
    std::cout << "Press any key to close the window.\n";
    cv::waitKey(0);

    return 0;
}

void canny(cv::Mat *frame, float threshold) {
    float highThreshold = threshold;
    if (highThreshold < 0) highThreshold = 0;
    else if (highThreshold > 255) highThreshold = 255;

    cv::Mat edges;
    cv::Canny(*frame, edges, highThreshold, highThreshold / 3, 3, true);
    *frame = edges;
}

void usage() {
    std::cout << "Usage: <executable> <image>\n";
}
