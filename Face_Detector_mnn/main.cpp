//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include <iostream>
#include <opencv2/opencv.hpp>

#include "retinaface.hpp"


using namespace std;


int main(int argc, char **argv) {

	if (argc <= 2) {
		fprintf(stderr, "Usage: %s <mnn .mnn> [image files...]\n", argv[0]);
		return 1;
	}

	std::string model_path = argv[1];

	Retinaface retinaface(model_path, 256, 0.5f, 0.4f, 1, false); // config model input
	
	auto start = chrono::steady_clock::now();
	for (int i = 2; i < argc; i++) {
		string image_file = argv[i];
		cout << "Processing " << image_file << endl;
		cv::Mat img = cv::imread(image_file);
		std::vector<landmark> boxes;
		retinaface.face_detect(img, boxes);
		auto end = chrono::steady_clock::now();
		chrono::duration<double> elapsed = end - start;
		cout << "time: " << elapsed.count() << " ms" << endl;
		for (int j = 0; j < boxes.size(); ++j) {
			int x, y, w, h;
			x = boxes[j].x1;
			y = boxes[j].y1;
			w = boxes[j].x2 - boxes[j].x1;
			h = boxes[j].y2 - boxes[j].y1;
			cv::Mat face_aligner;
			std::vector<cv::Point2f> keypoints;
			for (int index = 0; index < 5; index++) {
				keypoints.push_back(cv::Point2f(boxes[j].point[index]._x, boxes[j].point[index]._y));
			}

			cv::Rect rect(x, y, w, h);
			cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
			char test[80];
			sprintf(test, "%f", boxes[j].s);

			cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
			cv::circle(img, cv::Point(boxes[j].point[0]._x, boxes[j].point[0]._y), 1, cv::Scalar(0, 0, 225), 4);
			cv::circle(img, cv::Point(boxes[j].point[1]._x, boxes[j].point[1]._y), 1, cv::Scalar(0, 255, 225), 4);
			cv::circle(img, cv::Point(boxes[j].point[2]._x, boxes[j].point[2]._y), 1, cv::Scalar(255, 0, 225), 4);
			cv::circle(img, cv::Point(boxes[j].point[3]._x, boxes[j].point[3]._y), 1, cv::Scalar(0, 255, 0), 4);
			cv::circle(img, cv::Point(boxes[j].point[4]._x, boxes[j].point[4]._y), 1, cv::Scalar(255, 0, 0), 4);
		}
		cv::namedWindow("img", 0);
        cv::imshow("img", img);
        cv::waitKey(0);

	}

	
	return 0;
}
