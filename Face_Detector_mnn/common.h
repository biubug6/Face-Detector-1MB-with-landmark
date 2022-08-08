#ifndef _FACE_COMMON_HEADER_
#define _FACE_COMMON_HEADER_

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct location {
	float _x;
	float _y;
}location;

typedef struct landmark {
	float x1;
	float y1;
	float x2;
	float y2;
	float s;
	location point[5];
}landmark;

typedef struct rect_box {
	float cx;
	float cy;
	float sx;
	float sy;
}rect_box;

bool cmp(landmark a, landmark b);
void nms(std::vector<landmark> &input_boxes, float nms_thresh);
cv::Mat preprocess_img(cv::Mat& img, int input_h, int input_w);


#endif // !_FACE_COMMON_HEADER_

