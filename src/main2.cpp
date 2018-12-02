#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "grid_potts_example.h"
#include <cstring>

#define BATCHES 7
#define FRAMES 5
#define FLOWS 4
#define SEPARATIONS FRAMES - 2
#define SEPARATIONS_CENTRAL_INDEX SEPARATIONS / 2
#define FRAMES_CENTRAL_INDEX FRAMES / 2

float getBackgroundMean(cv::Mat *flow);

float getForegroundMean(cv::Mat *flow);

float getMeanMagnitudeWithThreshold(cv::Mat *flow, float threshold, int layer);

void flowArrowsAndSave(cv::Mat *orig, cv::Mat *flow, std::string *name);

void flowColorsAndSave(cv::Mat *orig, cv::Mat *flow, std::string *name);

void separateBasedOnTwoFieldsAndSave(cv::Mat *frame, cv::Mat *field1, cv::Mat *field2, std::string *name,
                                     cv::Mat *separated);

void removeObstructions(std::string *name, cv::Mat *framesOrig, cv::Mat *separations);

int main(int argc, char *argv[]) {

    std::string baseNames[BATCHES] = {"bars", "beach", "building", "car", "run", "sunset", "window_bars"};

    for (int batch = 0; batch < BATCHES; batch++) {
        cv::Mat framesOrig[FRAMES];
        cv::Mat framesManip[FRAMES];

        for (int i = 0; i < FRAMES; i++) {
            std::string name = "../img/" + baseNames[batch] + "_" + std::to_string(i) + ".jpg";

            framesOrig[i] = imread(name, cv::IMREAD_COLOR);
            if (!framesOrig[i].data) {
                std::cout << "Could not open or find the image " << name << std::endl;
                return -1;
            }

            cv::cvtColor(framesOrig[i], framesManip[i], cv::COLOR_BGR2GRAY);
        }

        //calculate flows
        cv::Mat flows[FLOWS];
        for (int i = 0; i < 4; i++) {
            std::string name_arrows = baseNames[batch] + "_arrows_" + std::to_string(i) + ".jpg";
            std::string name_colors = baseNames[batch] + "_colors_" + std::to_string(i) + ".jpg";

            cv::calcOpticalFlowFarneback(framesManip[i], framesManip[i + 1], flows[i], 0.5, 3, 15, 3, 5, 1.2, 0);
            flowArrowsAndSave(&framesOrig[i], &flows[i], &name_arrows);
            flowColorsAndSave(&framesOrig[i], &flows[i], &name_colors);
        }

        //separate foreground and background
        cv::Mat separations[SEPARATIONS];
        for (int i = 0; i < SEPARATIONS; i++) {
            std::string name = baseNames[batch] + "_detected_" + std::to_string(i) + ".jpg";
            separateBasedOnTwoFieldsAndSave(&framesOrig[i + 1], &flows[i], &flows[i + 1], &name, &separations[i]);
        }

        //remove obstructions
        std::string name = baseNames[batch] + "_obstruction_free.jpg";
        removeObstructions(&name, framesOrig, separations);

    }

    return 0;
}

void removeObstructions(std::string *name, cv::Mat *framesOrig, cv::Mat *separations) {
    cv::Mat noObstructions;
    framesOrig[2].copyTo(noObstructions);

    auto previousSeparation = separations[0];
    auto centralSeparation = separations[1];
    auto nextSeparation = separations[2];

    for (int y = 0; y < noObstructions.size().height; ++y) {
        for (int x = 0; x < noObstructions.size().width; ++x) {


            if (centralSeparation.at<cv::Vec3b>(y, x)[2] >= 130) { //purple or red must be changed (is foreground)
                //find a pixel that is background and use it

                cv::Vec3b refPixel = noObstructions.at<cv::Vec3b>(y, x);

                if (previousSeparation.at<cv::Vec3b>(y, x)[2] < 130) {
                    //choose prev pixel
                    cv::Vec3b prevPixel = framesOrig[1].at<cv::Vec3b>(y, x);

                    refPixel[0] = prevPixel[0];
                    refPixel[1] = prevPixel[1];
                    refPixel[2] = prevPixel[2];


                } else if (nextSeparation.at<cv::Vec3b>(y, x)[2] < 130) {
                    //choose next pixel
                    cv::Vec3b nextPixel = framesOrig[3].at<cv::Vec3b>(y, x);

                    refPixel[0] = nextPixel[0];
                    refPixel[1] = nextPixel[1];
                    refPixel[2] = nextPixel[2];
                }

                noObstructions.at<cv::Vec3b>(y, x) = refPixel;
            }
        }
    }

    //save result
    cv::imwrite(*name, noObstructions);
}

void flowArrowsAndSave(cv::Mat *orig, cv::Mat *flow, std::string *name) {
    cv::Mat flowArrows;
    orig->copyTo(flowArrows);
    auto size = flowArrows.size();
    for (int y = 0; y < size.height; y += 5) {
        for (int x = 0; x < size.width; x += 5) {
            // get the flow from y, x position * 10 for better visibility
            const cv::Point2f flowAtXY = flow->at<cv::Point2f>(y, x);
            // draw line at flow direction
            cv::line(flowArrows, cv::Point(x, y), cv::Point(cvRound(x + flowAtXY.x), cvRound(y + flowAtXY.y)),
                     cv::Scalar(255, 0, 0));
            // draw initial point
            cv::circle(flowArrows, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
        }
    }

    // save the results
    cv::imwrite(*name, flowArrows);
}

void flowColorsAndSave(cv::Mat *orig, cv::Mat *flow, std::string *name) {
    std::vector<float> ptsX, ptsY;
    auto size = orig->size();
    //get polar coordinates of flow vector
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            // get the flow from y, x position * 10 for better visibility
            const cv::Point2f flowAtXY = flow->at<cv::Point2f>(y, x);
            ptsX.push_back(flowAtXY.x);
            ptsY.push_back(flowAtXY.y);
        }
    }

    std::vector<float> magnitudes, angles;
    cv::cartToPolar(ptsX, ptsY, magnitudes, angles);
    cv::normalize(magnitudes, magnitudes, 255, 0, cv::NORM_MINMAX);

    //create colorized flow from angles
    cv::Mat flowColors = cv::Mat::zeros(size, orig->type());
    cv::cvtColor(flowColors, flowColors, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels(3);
    split(flowColors, channels);
    int pointCount = 0;
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            channels[0].at<uchar>(y, x) = static_cast<uchar>(angles[pointCount] * 180 / (M_PI / 2));
            channels[1].at<uchar>(y, x) = 255;
            channels[2].at<uchar>(y, x) = static_cast<uchar>(magnitudes[pointCount]);

            pointCount++;
        }
    }
    merge(channels, flowColors);
    cv::cvtColor(flowColors, flowColors, cv::COLOR_HSV2BGR);

    // increase contrast
    cv::Mat contrast;
    flowColors.convertTo(contrast, -1, 1.5, 0);

    // save the results
    cv::imwrite(*name, contrast);
}

float getBackgroundMean(cv::Mat *flow) {
    auto size = flow->size();
    float mean = 0;
    float count = 0;
    //cv::Scalar intensity;
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            if (y == 0 || y == size.height - 1) {
                mean += flow->at<float>(y, x);
                count++;
            } else {
                if (x > 0 && x < size.width - 1) {
                    x = size.width - 2;
                } else {
                    mean += flow->at<float>(y, x);
                    count++;
                }
            }
        }
    }
    return mean / count;
}

float getForegroundMean(cv::Mat *flow) {
    auto size = flow->size();
    float mean = 0;
    float count = 0;
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            if (y != 0 && y != size.height - 1) {
                if (x != 0 && x != size.width - 1) {
                    mean += flow->at<float>(y, x);
                    count++;
                }
            }
        }
    }
    return mean / count;
}

float getMeanMagnitudeWithThreshold(cv::Mat *flow, float threshold, int layer) {
    auto size = flow->size();
    float media = 0;
    float count = 0;
    float value;
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            value = flow->at<float>(y, x);
            if (layer == 0) {
                if (value < threshold) {
                    media += value;
                    count++;
                }
            } else {
                if (value >= threshold) {
                    media += value;
                    count++;
                }
            }
        }
    }
    return media / count;
}

void separateBasedOnTwoFieldsAndSave(cv::Mat *frame, cv::Mat *field1, cv::Mat *field2, std::string *name,
                                     cv::Mat *separated) {
    cv::Mat meanField;
    //cv::addWeighted(*field1, 1, *field2, 1, 0, meanField);
    cv::multiply(*field1, *field2, meanField);

    /*for (int y = 0; y < meanField.size().height; y++) {
        for (int x = 0; x < meanField.size().width; x++) {
            field1->at<float>(y,x) = abs(field1->at<float>(y,x));
            field2->at<float>(y,x) = abs(field2->at<float>(y,x));
        }
    }*/

    float meanBg = getBackgroundMean(&meanField);
    float meanFg = getForegroundMean(&meanField);
    float oldThreshold = 0.0;
    float threshold = (meanBg + meanFg) / 2;
    while (abs(threshold - oldThreshold) > 0.1) {
        meanBg = getMeanMagnitudeWithThreshold(&meanField, threshold, 0);
        meanFg = getMeanMagnitudeWithThreshold(&meanField, threshold, 1);
        oldThreshold = threshold;
        threshold = (meanBg + meanFg) / 2;
    }

    float maxBlue = (meanBg + threshold) / 2;
    float maxPurple = (threshold + meanFg) / 2;

    //cv::threshold(meanField, meanField, maxBlue, 0, cv::THRESH_TOZERO_INV);

    cv::Mat separatedImage;
    frame->copyTo(separatedImage);
    cv::Vec3b color;
    auto size = frame->size();
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            color = separatedImage.at<cv::Vec3b>(y, x);

            if (abs(meanField.at<float>(y, x * 2 /*TODO dont *2*/)) < abs(maxBlue)) { //paint it blue
                color[0] = 255;
                color[1] = 255;
                color[2] = 0;
                separatedImage.at<cv::Vec3b>(y, x) = color;
            } else if (abs(meanField.at<float>(y, x * 2 /*TODO dont *2*/)) < abs(maxPurple)) { //paint it purple
                color[0] = 130;
                color[1] = 0;
                color[2] = 130;
                separatedImage.at<cv::Vec3b>(y, x) = color;
            } else { //paint it red
                color[0] = 0;
                color[1] = 0;
                color[2] = 255;
                separatedImage.at<cv::Vec3b>(y, x) = color;
            }
        }
    }

    cv::Mat blend;
    cv::addWeighted(*frame, 0.3, separatedImage, 0.7, 0, blend);

    // save the results
    cv::imwrite(*name, blend);

    separatedImage.copyTo(*separated);
}