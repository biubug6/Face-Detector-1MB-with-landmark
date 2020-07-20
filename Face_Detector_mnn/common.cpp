#include "common.h"



bool cmp(landmark a, landmark b) {
	if (a.s > b.s)
		return true;
	return false;
}

void nms(std::vector<landmark> &input_boxes, float nms_thresh)
{
	std::vector<float>vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}
	for (int i = 0; i < int(input_boxes.size()); ++i) {
		for (int j = i + 1; j < int(input_boxes.size());) {
			float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
			float w = std::max(float(0), xx2 - xx1 + 1);
			float h = std::max(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);
			if (ovr >= nms_thresh) {
				input_boxes.erase(input_boxes.begin() + j);
				vArea.erase(vArea.begin() + j);
			}
			else {
				j++;
			}
		}
	}
}

cv::Mat preprocess_img(cv::Mat& img, int input_h, int input_w) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0f);
	float r_h = input_h / (img.rows*1.0f);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * static_cast<float>(img.rows);
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	return out;
}
