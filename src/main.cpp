#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "grid_potts_example.h"
#include <cstring>
#define M_PI 3.1415

void usage();

float get_background_media(cv::Mat *flow);

float get_foreground_media(cv::Mat *flow);

float get_media_pixels_limiar(cv::Mat *flow, float limiar, int layer);

void flowArrows(cv::Mat *orig, cv::Mat *flow, int i);

void flowColors(cv::Mat *orig, cv::Mat *flow, int i);

void background_image(cv::Mat *orig, cv::Mat *flow, float limiar);

int main(int argc, char **argv) {
    //if (argc != 6) { //TODO change to 5 images (argc = 6) and use all pairs
        //usage();
        //return -1;
    //}
	std::string arquivosChar[5];
	arquivosChar[0] = "C:/Users/I866859/IdeaProjects/OFP/img/teste1.jpg";
	arquivosChar[1] = "C:/Users/I866859/IdeaProjects/OFP/img/teste2.jpg";
	arquivosChar[2] = "C:/Users/I866859/IdeaProjects/OFP/img/teste3.jpg";
	arquivosChar[3] = "C:/Users/I866859/IdeaProjects/OFP/img/teste4.jpg";
	arquivosChar[4] = "C:/Users/I866859/IdeaProjects/OFP/img/teste5.jpg";
	static const int size = 6;
    cv::Mat framesOrig[size - 1];
    cv::Mat framesManip[size - 1];

    for (int i = 0; i < size - 1; i++) {
        framesOrig[i] = imread(arquivosChar[i], cv::IMREAD_COLOR);
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

    cv::Mat flow[size - 2];
    //cv::calcOpticalFlowFarneback(framesManip[0], framesManip[1], flow, 0.4, 1, 12, 2, 8, 1.2, 0);
    for(int i = 0; i < 4; i++) {
        cv::calcOpticalFlowFarneback(framesManip[i], framesManip[i+1], flow[i], 0.5, 3, 15, 3, 5, 1.2, 0);    
        flowArrows(&framesOrig[i], &flow[i], i);
        flowColors(&framesOrig[i], &flow[i], i);
    }

	// NAO FUNCIONA MT BEM, CORTA A IMAGEM FOREGROUND
	/*auto size_f = flow[0].size();
	cv::Mat mean_flow = cv::Mat::zeros(size_f, flow[0].type());
	for (int i = 0; i < 4; i++) {
		for (int y = 0; y < size_f.height; y ++) {
			for (int x = 0; x < size_f.width; x ++) {
				mean_flow.at<float>(y, x) += flow[i].at<float>(y, x)/4.0;
			}
		}
	}*/

	float mean_bg = get_background_media(&flow[0]);
	float mean_fg = get_foreground_media(&flow[0]);
	float old_limiar = 0.0;
	float limiar = (mean_bg + mean_fg) / 2;
	while (abs(limiar - old_limiar) > 0.1) {
		mean_bg = get_media_pixels_limiar(&flow[0], limiar, 0);
		mean_fg = get_media_pixels_limiar(&flow[0], limiar, 1);
		old_limiar = limiar;
		limiar = (mean_bg + mean_fg) / 2;
	}

	background_image(&framesOrig[0], &flow[0], limiar);

    return 0;
}

void usage() { //TODO change to 5 frames
    std::cout << "Usage: <executable> <image1> <image2> <image3> <image4> <image5>\n";
}

void flowArrows(cv::Mat *orig, cv::Mat *flow, int i) {
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
    cv::namedWindow(std::to_string(i+5), cv::WINDOW_AUTOSIZE);
    cv::imshow(std::to_string(i+5), flowArrows);
    cv::waitKey(0);
}

void flowColors(cv::Mat *orig, cv::Mat *flow, int i) {
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
    cv::namedWindow(std::to_string(i+15), cv::WINDOW_AUTOSIZE);
    cv::imshow(std::to_string(i+15), flowColors);
    cv::waitKey(0);
}


float get_background_media(cv::Mat *flow) {
	auto size = flow->size();
	float media = 0;
	float count = 0;
	//cv::Scalar intensity;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			if (y == 0 || y == size.height - 1) {
				media += flow->at<float>(y, x);
				count++;
			}
			else {
				if (x > 0 && x < size.width - 1) {
					x = size.width - 2;
				}
				else {
					media += flow->at<float>(y, x);
					count++;
				}
			}
		}
	}
	return media/count;
}


float get_foreground_media(cv::Mat *flow) {
	auto size = flow->size();
	float media = 0;
	float count = 0;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			if (y != 0 && y != size.height - 1) {
				if (x != 0 && x != size.width - 1) {
					media += flow->at<float>(y, x);
					count++;
				}
			}
		}
	}
	return media/count;
}

float get_media_pixels_limiar(cv::Mat *flow, float limiar, int layer) {
	auto size = flow->size();
	float media = 0;
	float count = 0;
	float value;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			value = flow->at<float>(y, x);
			if (layer == 0) {
				if (value < limiar) {
					media += value;
					count++;
				}
			}
			else {
				if (value >= limiar) {
					media += value;
					count++;
				}
			}
		}
	}
	return media / count;
}


void background_image(cv::Mat *orig, cv::Mat *flow, float limiar) {
	cv::Mat background;
	cv::Vec3b color;
	orig->copyTo(background);
	auto size = orig->size();
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			color = background.at<cv::Vec3b>(y, x);
			if (flow->at<float>(y, x*2) < limiar) {
				color[0] = 255;
				color[1] = 255;
				color[2] = 0;
				background.at<cv::Vec3b>(y, x) = color;
			}
			else {
				color[0] = 0;
				color[1] = 0;
				color[2] = 255;
				background.at<cv::Vec3b>(y, x) = color;
			}
		}
	}
	namedWindow("Background", cv::WINDOW_AUTOSIZE);
	imshow("Background", background);
	std::cout << "Press any key to close the window.\n";
	cv::waitKey(0);
}