#include <algorithm>

#include "retinaface.hpp"

Retinaface::Retinaface(
	std::string &mnn_path, int img_size, float threshold, float nms, int num_thread, bool retinaface) :
	_threshold(threshold),
	_nms(nms),
	_num_thread(num_thread),
	img_size(img_size),
	_mean_val{ 104.f, 117.f, 123.f },
	_retinaface(retinaface) {
	ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
	MNN::ScheduleConfig config;
	config.numThread = _num_thread;
	MNN::BackendConfig backendConfig;
	backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
	config.backendConfig = &backendConfig;

	ultraface_session = ultraface_interpreter->createSession(config);

	input_tensor = ultraface_interpreter->getSessionInput(ultraface_session, nullptr);
	if (_retinaface)
		create_anchor_retinaface(anchors, img_size, img_size);
	else
		create_anchor(anchors, img_size, img_size);

	ultraface_interpreter->resizeTensor(input_tensor, { 1, 3, img_size, img_size });
	ultraface_interpreter->resizeSession(ultraface_session);
	pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
		MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, _mean_val, 3, _norm_vals, 3));
	
}

void Retinaface::compose_results(
	cv::Mat img, MNN::Tensor *scores, MNN::Tensor *boxes, MNN::Tensor *landmarks, std::vector<landmark> &results) {
	// #pragma omp parallel for num_threads(2)
	//std::vector<landmark> total_box;
	float *sptr = scores->host<float>();
	float *bptr = boxes->host<float>();
	float *ldptr = landmarks->host<float>();
	for (int i = 0; i < anchors.size(); ++i)
	{
		float score = *(sptr + 1);
		if (score > _threshold)
		{
			rect_box anchor = anchors[i];
			//box tmp1;
			landmark result;

			float x_center = anchor.cx + bptr[0] * 0.1 * anchor.sx;
			float y_center = anchor.cy + bptr[1] * 0.1 * anchor.sy;
			float w = anchor.sx * exp(bptr[2] * 0.2);
			float h = anchor.sy * exp(bptr[3] * 0.2);

			result.x1 = (x_center - w / 2) * img_size;
			result.x1 = result.x1 > 0 ? result.x1 : 0;

			result.y1 = (y_center - h / 2) * img_size;
			result.y1 = result.y1 > 0 ? result.y1 : 0;

			result.x2 = (x_center + w / 2) * img_size;
			result.x2 = result.x2 <= img_size - 1 ? result.x2 : img_size - 1;

			result.y2 = (y_center + h / 2) * img_size;
			result.y2 = result.y2 <= img_size - 1 ? result.y2 : img_size - 1;

			result.s = score;

			// landmark

			for (int j = 0; j < 5; ++j) {
				result.point[j]._x = (anchor.cx + *(ldptr + (j << 1)) * 0.1 * anchor.sx) * img_size;
				result.point[j]._y = (anchor.cy + *(ldptr + (j << 1) + 1) * 0.1 * anchor.sy) * img_size;
			}

			{
				int x1, x2, y1, y2;
				float r_w = img_size / (img.cols * 1.0);
				float r_h = img_size / (img.rows * 1.0);
				if (r_h > r_w) {
					float pad = (img_size - r_w * img.rows) / 2;
					x1 = result.x1;
					x2 = result.x2;
					y1 = result.y1 - pad;
					y2 = result.y2 - pad;
					x1 = x1 / r_w;
					x2 = x2 / r_w;
					y1 = y1 / r_w;
					y2 = y2 / r_w;
					for (int index = 0; index < 5; index++) {
						result.point[index]._y -= pad;
						result.point[index]._y /= r_w;
						result.point[index]._x /= r_w;
					}
				}
				else {
					float pad = (img_size - r_h * img.cols) / 2;
					x1 = result.x1 - pad;
					x2 = result.x2 - pad;
					y1 = result.y1;
					y2 = result.y2;
					x1 = x1 / r_h;
					x2 = x2 / r_h;
					y1 = y1 / r_h;
					y2 = y2 / r_h;
					for (int index = 0; index < 5; index++) {
						result.point[index]._x -= pad;
						result.point[index]._x /= r_h;
						result.point[index]._y /= r_h;
					}
				}
				result.x1 = x1;
				result.x2 = x2;
				result.y1 = y1;
				result.y2 = y2;
			}

			results.push_back(result);
		}
		sptr += 2;
		bptr += 4;
		ldptr += 10;
	}
	
// 	for (auto face:total_box) {
// 		cv::Rect rect(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
// 		cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
// 	}

	std::sort(results.begin(), results.end(), cmp);
	
	nms(results, _nms);
	
}


void Retinaface::face_detect(cv::Mat& img, std::vector<landmark>& boxes)
{
	if (img.empty()) {
		std::cout << "image is empty ,please check!" << std::endl;
		return;
	}

	// 	int image_h = img.rows;
	// 	int image_w = img.cols;
	cv::Mat image = preprocess_img(img, img_size, img_size);

// 	ultraface_interpreter->resizeTensor(input_tensor, { 1, 3, img_size, img_size });
// 	ultraface_interpreter->resizeSession(ultraface_session);
// 	std::shared_ptr<MNN::CV::ImageProcess> pretreat(
// 		MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, _mean_val, 3, _norm_vals, 3));
 	pretreat->convert(image.data, img_size, img_size, image.step[0], input_tensor);

	// auto start = chrono::steady_clock::now();

	// run network
	ultraface_interpreter->runSession(ultraface_session);

	// get output data

	std::string output[3] = { "scores", "boxes","landmarks" };
	MNN::Tensor *tensor_scores = ultraface_interpreter->getSessionOutput(ultraface_session, output[0].c_str());
	MNN::Tensor *tensor_boxes = ultraface_interpreter->getSessionOutput(ultraface_session, output[1].c_str());
	MNN::Tensor *tensor_ldmarks = ultraface_interpreter->getSessionOutput(ultraface_session, output[2].c_str());

	MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
	MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
	MNN::Tensor tensor_ldmarks_host(tensor_ldmarks, tensor_ldmarks->getDimensionType());

	tensor_scores->copyToHostTensor(&tensor_scores_host);
	tensor_boxes->copyToHostTensor(&tensor_boxes_host);
	tensor_ldmarks->copyToHostTensor(&tensor_ldmarks_host);

	compose_results(img, tensor_scores, tensor_boxes, tensor_ldmarks, boxes);

}

Retinaface::~Retinaface() {
	ultraface_interpreter->releaseModel();
	ultraface_interpreter->releaseSession(ultraface_session);
}

void Retinaface::create_anchor(std::vector<rect_box> &anchor, int w, int h)
{
	//    anchor.reserve(num_boxes);
	anchor.clear();
	std::vector<std::vector<int> > feature_map(4), min_sizes(4);
	float steps[] = { 8, 16, 32, 64 };
	for (int i = 0; i < feature_map.size(); ++i) {
		feature_map[i].push_back(ceil(h / steps[i]));
		feature_map[i].push_back(ceil(w / steps[i]));
	}
	std::vector<int> minsize1 = { 10, 16, 24 };
	min_sizes[0] = minsize1;
	std::vector<int> minsize2 = { 32, 48 };
	min_sizes[1] = minsize2;
	std::vector<int> minsize3 = { 64, 96 };
	min_sizes[2] = minsize3;
	std::vector<int> minsize4 = { 128, 192, 256 };
	min_sizes[3] = minsize4;


	for (int k = 0; k < feature_map.size(); ++k)
	{
		std::vector<int> min_size = min_sizes[k];
		for (int i = 0; i < feature_map[k][0]; ++i)
		{
			for (int j = 0; j < feature_map[k][1]; ++j)
			{
				for (int l = 0; l < min_size.size(); ++l)
				{
					float s_kx = min_size[l] * 1.0 / w;
					float s_ky = min_size[l] * 1.0 / h;
					float cx = (j + 0.5) * steps[k] / w;
					float cy = (i + 0.5) * steps[k] / h;
					rect_box axil = { cx, cy, s_kx, s_ky };
					anchor.push_back(axil);
				}
			}
		}

	}

}

void Retinaface::create_anchor_retinaface(std::vector<rect_box> &anchor, int w, int h)
{
	//    anchor.reserve(num_boxes);
	anchor.clear();
	std::vector<std::vector<int> > feature_map(3), min_sizes(3);
	float steps[] = { 8, 16, 32 };
	for (int i = 0; i < feature_map.size(); ++i) {
		feature_map[i].push_back(ceil(h / steps[i]));
		feature_map[i].push_back(ceil(w / steps[i]));
	}
	std::vector<int> minsize1 = { 10, 20 };
	min_sizes[0] = minsize1;
	std::vector<int> minsize2 = { 32, 64 };
	min_sizes[1] = minsize2;
	std::vector<int> minsize3 = { 128, 256 };
	min_sizes[2] = minsize3;

	for (int k = 0; k < feature_map.size(); ++k)
	{
		std::vector<int> min_size = min_sizes[k];
		for (int i = 0; i < feature_map[k][0]; ++i)
		{
			for (int j = 0; j < feature_map[k][1]; ++j)
			{
				for (int l = 0; l < min_size.size(); ++l)
				{
					float s_kx = min_size[l] * 1.0 / w;
					float s_ky = min_size[l] * 1.0 / h;
					float cx = (j + 0.5) * steps[k] / w;
					float cy = (i + 0.5) * steps[k] / h;
					rect_box axil = { cx, cy, s_kx, s_ky };
					anchor.push_back(axil);
				}
			}
		}

	}

}

