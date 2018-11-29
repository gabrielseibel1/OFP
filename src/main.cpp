#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "grid_potts_example.h"

void usage();

void flowArrows(cv::Mat *orig, cv::Mat *flow);

void flowColors(cv::Mat *orig, cv::Mat *flow);

int main(int argc, char **argv) {
    if (argc != 3) { //TODO change to 5 images (argc = 6) and use all pairs
        usage();
        return -1;
    }

    cv::Mat framesOrig[argc - 1];
    cv::Mat framesManip[argc - 1];

    for (int i = 0; i < argc - 1; i++) {
        framesOrig[i] = imread(argv[i + 1], cv::IMREAD_COLOR);
        if (!framesOrig[i].data) {
            std::cout << "Could not open or find the image " << i << std::endl;
            return -1;
        }

        cv::Canny(framesOrig[i], framesManip[i], 200, 200 / 3, 3, true);
        threshold(framesManip[i], framesManip[i], 0, 255, cv::THRESH_TOZERO /*dont use binary*/);

        namedWindow(std::to_string(i), cv::WINDOW_AUTOSIZE);
        imshow(std::to_string(i), framesManip[i]);
        std::cout << "Press any key to close the window.\n";
        cv::waitKey(0);
    }

    cv::Mat flow;
    //cv::calcOpticalFlowFarneback(framesManip[0], framesManip[1], flow, 0.4, 1, 12, 2, 8, 1.2, 0);
    cv::calcOpticalFlowFarneback(framesManip[0], framesManip[1], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    flowArrows(&framesOrig[0], &flow);
    flowColors(&framesOrig[0], &flow);

    return 0;
}

void usage() { //TODO change to 5 frames
    std::cout << "Usage: <executable> <image> <image2>\n";
}

void flowArrows(cv::Mat *orig, cv::Mat *flow) {
    cv::Mat flowArrows;
    orig->copyTo(flowArrows);
    auto size = flowArrows.size();
    for (int y = 0; y < size.height; y += 5) {
        for (int x = 0; x < size.width; x += 5) {
            // get the flow from y, x position * 10 for better visibility
            const cv::Point2f flowAtXY = flow->at<cv::Point2f>(y, x);
            // draw line at flow direction
            cv::line(flowArrows, cv::Point(x, y), cv::Point(cvRound(x + flowAtXY.x), cvRound(y + flowAtXY.y)), cv::Scalar(255,0,0));
            // draw initial point
            cv::circle(flowArrows, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
        }
    }
    // draw the results
    cv::namedWindow("Flow - Arrows", cv::WINDOW_AUTOSIZE);
    cv::imshow("Flow - Arrows", flowArrows);
    cv::waitKey(0);
}

void flowColors(cv::Mat *orig, cv::Mat *flow) {
    std::vector<float> ptsX, ptsY;
    auto size = orig->size();
    //get polar coordinates of flow vector
    for (int y = 0; y < size.height; y += 2) {
        for (int x = 0; x < size.width; x += 2) {
            // get the flow from y, x position * 10 for better visibility
            const cv::Point2f flowAtXY = flow->at<cv::Point2f>(y, x);

            ptsX.push_back(flowAtXY.x);
            ptsY.push_back(flowAtXY.y);
        }
    }

    std::vector<float> magnitudes, angles;
    cv::cartToPolar(ptsX, ptsY, magnitudes, angles);
    cv::normalize(magnitudes, magnitudes, 0, 255, cv::NORM_MINMAX);

    //create colorized flow from angles
    cv::Mat flowColors = cv::Mat::zeros(size, orig->type());
    cv::cvtColor(flowColors, flowColors, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels(3);
    split(flowColors, channels);
    int pointCount = 0;
    for (int y = 0; y < size.height; y += 2) {
        for (int x = 0; x < size.width; x += 2) {
            channels[0].at<uchar>(y, x) = static_cast<uchar>(angles[pointCount] * 180 / (M_PI / 2));
            channels[1].at<uchar>(y, x) = 255;
            channels[2].at<uchar>(y, x) = static_cast<uchar>(magnitudes[pointCount]);

            pointCount++;
        }
    }
    merge(channels, flowColors);
    cv::cvtColor(flowColors, flowColors, cv::COLOR_HSV2BGR);

    // draw the results
    cv::namedWindow("Flow - Colors", cv::WINDOW_AUTOSIZE);
    cv::imshow("Flow - Colors", flowColors);
    cv::waitKey(0);
}
