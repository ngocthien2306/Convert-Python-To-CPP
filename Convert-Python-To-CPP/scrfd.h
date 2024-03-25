#ifndef LITE_AI_TOOLKIT_ORT_CV_SCRFD_H
#define LITE_AI_TOOLKIT_ORT_CV_SCRFD_H

#include "ort_core.h"
#include <cmath>
#include <algorithm>
#include <cassert>
using namespace std;
namespace ortcv
{
	class LITE_EXPORTS SCRFD : public BasicOrtHandler
	{
		public:
			explicit SCRFD(const string& _onnx_path, unsigned int _num_threads = 1);

			~SCRFD() override = default;
			

        private:
            // nested classes
            typedef struct
            {
                float cx;
                float cy;
                float stride;
            } SCRFDPoint;
            typedef struct
            {
                float ratio;
                int dw;
                int dh;
                bool flag;
            } SCRFDScaleParams;

        private:


            const float mean_vals[3] = { 127.5f, 127.5f, 127.5f }; // RGB
            const float scale_vals[3] = { 1.f / 128.f, 1.f / 128.f, 1.f / 128.f };

            unsigned int fmc = 3; // feature map count
            bool use_kps = false;
            vector<int> feat_stride_fpn = { 8, 16, 32 };
            unsigned int num_anchors = 2;
            bool center_points_is_update = false;
            unordered_map<int, vector<SCRFDPoint>> center_points;

            static constexpr const unsigned int nms_pre = 1000;
            static constexpr const unsigned int max_nms = 30000;

        private:
            Ort::Value preprocessing(const cv::Mat& mat_rs) override;

            void _init_vars();

            void generate_bboxes_kps(const SCRFDScaleParams& scale_params,
                vector<types::BoxfWithLandmarks>& bbox_kps_collection,
                vector<Ort::Value>& output_tensors,
                float score_threshold, float img_height,
                float img_width); // rescale & exclude

            void nms(std::vector<types::BoxfWithLandmarks>& input,
                std::vector<types::BoxfWithLandmarks>& output,
                float iou_threshold, unsigned int topk);

            void generate_points(const int target_height, const int target_width);

            void generate_bboxes_single_stride(const SCRFDScaleParams& scale_params,
                Ort::Value& score_pred,
                Ort::Value& bbox_pred,
                unsigned int stride,
                float score_threshold,
                float img_height,
                float img_width,
                std::vector<types::BoxfWithLandmarks>& bbox_kps_collection);

            void generate_bboxes_kps_single_stride(const SCRFDScaleParams& scale_params,
                Ort::Value& score_pred,
                Ort::Value& bbox_pred,
                Ort::Value& kps_pred,
                unsigned int stride,
                float score_threshold,
                float img_height,
                float img_width,
                std::vector<types::BoxfWithLandmarks>& bbox_kps_collection,
                string metric);

            void forward(cv::Mat img, const SCRFDScaleParams& scale_params,
                vector<types::BoxfWithLandmarks>& bbox_kps_collection,
                double nms_thresh, float img_height, float img_width);

        public:

            void detect(cv::Mat img, vector<types::BoxfWithLandmarks>& detected_boxes_kps, double nms_thresh, const int max_nms = 10000, const string metric="default", const double iou_threshold=0.5, const int topk=400);
            void detect2(cv::Mat img, vector<types::BoxfWithLandmarks>& detected_boxes_kps, 
                double nms_thresh, const int max_nms = 10000, 
                const string metric = "default", const double iou_threshold = 0.5, 
                const int topk = 400);


	};
}


class Distance {
public:
    static vector<vector<double>> distance2bbox(const vector<vector<double>>& points, const vector<vector<double>>& distance, const vector<int>& max_shape);
    static vector<vector<double>> distance2kps(const vector<vector<double>>& points, const vector<vector<double>>& distance, const vector<int>& max_shape);
};


#endif //LITE_AI_TOOLKIT_ORT_CV_SCRFD_H