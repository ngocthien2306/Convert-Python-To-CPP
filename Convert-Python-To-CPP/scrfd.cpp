#include "models.h"
#include "ort_utils.h"

using ortcv::SCRFD;

SCRFD::SCRFD(const string& _onnx_path, unsigned int _num_threads) :
    BasicOrtHandler(_onnx_path, _num_threads)
{
    _init_vars();
}

void SCRFD::_init_vars()
{
    if (num_outputs == 6)
    {
        fmc = 3;
        feat_stride_fpn = { 8, 16, 32 };
        num_anchors = 2;
        use_kps = false;
    } // kps
    else if (num_outputs == 9)
    {
        fmc = 3;
        feat_stride_fpn = { 8, 16, 32 };
        num_anchors = 2;
        use_kps = true;
    }
    else if (num_outputs == 15) {
        fmc = 5;
        feat_stride_fpn = { 8, 16, 32, 64, 128 };
        num_anchors = 1;
        use_kps = false;
    }
    else if (num_outputs == 10) {
        fmc = 5;
        feat_stride_fpn = { 8, 16, 32, 64, 128 };
        num_anchors = 1;
        use_kps = false;
    }

}


Ort::Value SCRFD::preprocessing(const cv::Mat& mat_rs)
{
    std::vector<int> input_size = { mat_rs.rows, mat_rs.cols };

    cv::Mat mat_cp = mat_rs.clone();

    cv::cvtColor(mat_cp, mat_cp, cv::COLOR_BGR2RGB);

    // normalize image
    if (mat_cp.type() != CV_32FC3) mat_cp.convertTo(mat_cp, CV_32FC3);
    for (unsigned int i = 0; i < mat_cp.rows; ++i) {
        cv::Vec3f* p = mat_cp.ptr<cv::Vec3f>(i);
        for (unsigned int j = 0; j < mat_cp.cols; ++j) {
            p[j][0] = (p[j][0] - mean_vals[0]) * scale_vals[0];
            p[j][1] = (p[j][1] - mean_vals[1]) * scale_vals[1];
            p[j][2] = (p[j][2] - mean_vals[2]) * scale_vals[2];
        }
    }
    return ortcv::utils::transform::create_tensor(
        mat_cp, input_node_dims, memory_info_handler,
        input_values_handler, ortcv::utils::transform::CHW);
}

void SCRFD::forward(cv::Mat mat_rs, const SCRFDScaleParams& scale_params, vector<types::BoxfWithLandmarks> &bbox_kps_collection, double score_threshold, float img_height, float img_width) {

    cv::Mat mat_cp = mat_rs.clone();
    cv::cvtColor(mat_cp, mat_cp, cv::COLOR_BGR2RGB);

    // normalize image
    if (mat_cp.type() != CV_32FC3) mat_cp.convertTo(mat_cp, CV_32FC3);
    for (unsigned int i = 0; i < mat_cp.rows; ++i) {
        cv::Vec3f* p = mat_cp.ptr<cv::Vec3f>(i);
        for (unsigned int j = 0; j < mat_cp.cols; ++j) {
            p[j][0] = (p[j][0] - mean_vals[0]) * scale_vals[0];
            p[j][1] = (p[j][1] - mean_vals[1]) * scale_vals[1];
            p[j][2] = (p[j][2] - mean_vals[2]) * scale_vals[2];
        }
    }

    // convert mat to tensor
    Ort::Value input_tensor = ortcv::utils::transform::create_tensor(
        mat_cp, input_node_dims, memory_info_handler,
        input_values_handler, ortcv::utils::transform::CHW);

    // predict
    auto output_tensors = ort_session->Run(
        Ort::RunOptions{ nullptr }, input_node_names.data(),
        &input_tensor, 1, output_node_names.data(), num_outputs
    );

    // score_8, score_16, score_32, bbox_8, bbox_16, bbox_32
    Ort::Value& score_8 = output_tensors.at(0);  // e.g [1,12800,1]
    Ort::Value& score_16 = output_tensors.at(1); // e.g [1,3200,1]
    Ort::Value& score_32 = output_tensors.at(2); // e.g [1,800,1]
    Ort::Value& bbox_8 = output_tensors.at(3);   // e.g [1,12800,4]
    Ort::Value& bbox_16 = output_tensors.at(4);  // e.g [1,3200,4]
    Ort::Value& bbox_32 = output_tensors.at(5);  // e.g [1,800,4]

    // generate center points.
    const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
    const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640
    this->generate_points(input_height, input_width);

    bbox_kps_collection.clear();
    

    if (use_kps)
    {
        Ort::Value& kps_8 = output_tensors.at(6);   // e.g [1,12800,10]
        Ort::Value& kps_16 = output_tensors.at(7);  // e.g [1,3200,10]
        Ort::Value& kps_32 = output_tensors.at(8);  // e.g [1,800,10]

        // level 8 & 16 & 32 with kps
        this->generate_bboxes_kps_single_stride(scale_params, score_8, bbox_8, kps_8, 8, score_threshold,
            img_height, img_width, bbox_kps_collection, "default");
        this->generate_bboxes_kps_single_stride(scale_params, score_16, bbox_16, kps_16, 16, score_threshold,
            img_height, img_width, bbox_kps_collection, "default");
        this->generate_bboxes_kps_single_stride(scale_params, score_32, bbox_32, kps_32, 32, score_threshold,
            img_height, img_width, bbox_kps_collection, "default");
    } // no kps
    else
    {
        // level 8 & 16 & 32
        this->generate_bboxes_single_stride(scale_params, score_8, bbox_8, 8, score_threshold,
            img_height, img_width, bbox_kps_collection);
        this->generate_bboxes_single_stride(scale_params, score_16, bbox_16, 16, score_threshold,
            img_height, img_width, bbox_kps_collection);
        this->generate_bboxes_single_stride(scale_params, score_32, bbox_32, 32, score_threshold,
            img_height, img_width, bbox_kps_collection);
    }

    #if LITEORT_DEBUG
        std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
    #endif

}


void SCRFD::detect2(cv::Mat mat, vector<types::BoxfWithLandmarks>& detected_boxes_kps, double score_threshold, const int max_nms, const string metric, const double iou_threshold, const int topk) {
    cv::Mat mat_rs;
    SCRFDScaleParams scale_params;
    const int target_height = (int)input_node_dims.at(2);
    const int target_width = (int)input_node_dims.at(3);

    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
        cv::Scalar(0, 0, 0));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float)target_width / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float)img_width * r); // floor
    int new_unpad_h = static_cast<int>((float)img_height * r); // floor
    int pad_w = target_width - new_unpad_w; // >=0
    int pad_h = target_height - new_unpad_h; // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.ratio = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.flag = true;

    vector<types::BoxfWithLandmarks> bbox_kps_collection;

    this->forward(mat_rs, scale_params, bbox_kps_collection, score_threshold, img_height, img_width);
    this->nms(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);



}

void SCRFD::detect(cv::Mat mat, vector<types::BoxfWithLandmarks>& detected_boxes_kps, double nms_thresh, const int max_nms, const string metric, const double iou_threshold, const int topk) {
    cv::Mat mat_rs;
    SCRFDScaleParams scale_params;
    const int target_height = (int)input_node_dims.at(2);
    const int target_width = (int)input_node_dims.at(3);

    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
        cv::Scalar(0, 0, 0));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float)target_width / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float)img_width * r); // floor
    int new_unpad_h = static_cast<int>((float)img_height * r); // floor
    int pad_w = target_width - new_unpad_w; // >=0
    int pad_h = target_height - new_unpad_h; // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.ratio = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.flag = true;


    Ort::Value input_tensor = this->preprocessing(mat_rs);

    auto output_tensors = ort_session->Run(
        Ort::RunOptions{ nullptr }, input_node_names.data(),
        &input_tensor, 1, output_node_names.data(), num_outputs
    );

    std::vector<types::BoxfWithLandmarks> bbox_kps_collection;
    this->generate_bboxes_kps(scale_params, bbox_kps_collection, output_tensors,
        nms_thresh, img_height, img_width);

    this->nms(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);

}


void SCRFD::generate_points(const int target_height, const int target_width)
{
    if (center_points_is_update) return;
    // 8, 16, 32
    for (auto stride : feat_stride_fpn)
    {
        unsigned int num_grid_w = target_width / stride;
        unsigned int num_grid_h = target_height / stride;
        // y
        for (unsigned int i = 0; i < num_grid_h; ++i)
        {
            // x
            for (unsigned int j = 0; j < num_grid_w; ++j)
            {
                // num_anchors, col major
                for (unsigned int k = 0; k < num_anchors; ++k)
                {
                    SCRFDPoint point;
                    point.cx = (float)j;
                    point.cy = (float)i;
                    point.stride = (float)stride;
                    center_points[stride].push_back(point);
                }

            }
        }
    }

    center_points_is_update = true;
}



void SCRFD::generate_bboxes_kps(const SCRFDScaleParams& scale_params,
    std::vector<types::BoxfWithLandmarks>& bbox_kps_collection,
    std::vector<Ort::Value>& output_tensors,
    float score_threshold,
    float img_height,
    float img_width)
{
    // score_8, score_16, score_32, bbox_8, bbox_16, bbox_32
    Ort::Value& score_8 = output_tensors.at(0);  // e.g [1,12800,1]
    Ort::Value& score_16 = output_tensors.at(1); // e.g [1,3200,1]
    Ort::Value& score_32 = output_tensors.at(2); // e.g [1,800,1]
    Ort::Value& bbox_8 = output_tensors.at(3);   // e.g [1,12800,4]
    Ort::Value& bbox_16 = output_tensors.at(4);  // e.g [1,3200,4]
    Ort::Value& bbox_32 = output_tensors.at(5);  // e.g [1,800,4]

    // generate center points.
    const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
    const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640
    this->generate_points(input_height, input_width);

    bbox_kps_collection.clear();

    if (use_kps)
    {
        Ort::Value& kps_8 = output_tensors.at(6);   // e.g [1,12800,10]
        Ort::Value& kps_16 = output_tensors.at(7);  // e.g [1,3200,10]
        Ort::Value& kps_32 = output_tensors.at(8);  // e.g [1,800,10]

        // level 8 & 16 & 32 with kps
        this->generate_bboxes_kps_single_stride(scale_params, score_8, bbox_8, kps_8, 8, score_threshold,
            img_height, img_width, bbox_kps_collection, "");
        this->generate_bboxes_kps_single_stride(scale_params, score_16, bbox_16, kps_16, 16, score_threshold,
            img_height, img_width, bbox_kps_collection, "");
        this->generate_bboxes_kps_single_stride(scale_params, score_32, bbox_32, kps_32, 32, score_threshold,
            img_height, img_width, bbox_kps_collection, "");
    } // no kps
    else
    {
        // level 8 & 16 & 32
        this->generate_bboxes_single_stride(scale_params, score_8, bbox_8, 8, score_threshold,
            img_height, img_width, bbox_kps_collection);
        this->generate_bboxes_single_stride(scale_params, score_16, bbox_16, 16, score_threshold,
            img_height, img_width, bbox_kps_collection);
        this->generate_bboxes_single_stride(scale_params, score_32, bbox_32, 32, score_threshold,
            img_height, img_width, bbox_kps_collection);
    }

#if LITEORT_DEBUG
    std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
#endif
}

void SCRFD::generate_bboxes_single_stride(
    const SCRFDScaleParams& scale_params, Ort::Value& score_pred, Ort::Value& bbox_pred,
    unsigned int stride, float score_threshold, float img_height, float img_width,
    std::vector<types::BoxfWithLandmarks>& bbox_kps_collection)
{
    unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
    nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

    auto stride_dims = score_pred.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int num_points = stride_dims.at(0);  // 12800
    const float* score_ptr = score_pred.GetTensorMutableData<float>(); // [1,12800,1]
    const float* bbox_ptr = bbox_pred.GetTensorMutableData<float>();   // [1,12800,4]

    float ratio = scale_params.ratio;
    int dw = scale_params.dw;
    int dh = scale_params.dh;

    unsigned int count = 0;
    auto& stride_points = center_points[stride];

    for (unsigned int i = 0; i < num_points; ++i)
    {
        const float cls_conf = score_ptr[i];
        if (cls_conf < score_threshold) continue; // filter
        auto& point = stride_points.at(i);
        const float cx = point.cx; // cx
        const float cy = point.cy; // cy
        const float s = point.stride; // stride

        // bbox
        const float* offsets = bbox_ptr + i * 4;
        float l = offsets[0]; // left
        float t = offsets[1]; // top
        float r = offsets[2]; // right
        float b = offsets[3]; // bottom

        types::BoxfWithLandmarks box_kps;
        float x1 = ((cx - l) * s - (float)dw) / ratio;  // cx - l x1
        float y1 = ((cy - t) * s - (float)dh) / ratio;  // cy - t y1
        float x2 = ((cx + r) * s - (float)dw) / ratio;  // cx + r x2
        float y2 = ((cy + b) * s - (float)dh) / ratio;  // cy + b y2
        box_kps.box.x1 = std::max(0.f, x1);
        box_kps.box.y1 = std::max(0.f, y1);
        box_kps.box.x2 = std::min(img_width - 1.f, x2);
        box_kps.box.y2 = std::min(img_height - 1.f, y2);
        box_kps.box.score = cls_conf;
        box_kps.box.label = 1;
        box_kps.box.label_text = "face";
        box_kps.box.flag = true;
        box_kps.flag = true;

        bbox_kps_collection.push_back(box_kps);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

    if (bbox_kps_collection.size() > nms_pre_)
    {
        std::sort(
            bbox_kps_collection.begin(), bbox_kps_collection.end(),
            [](const types::BoxfWithLandmarks& a, const types::BoxfWithLandmarks& b)
            { return a.box.score > b.box.score; }
        ); // sort inplace
        // trunc
        bbox_kps_collection.resize(nms_pre_);
    }
}

void SCRFD::generate_bboxes_kps_single_stride(
    const SCRFDScaleParams& scale_params, Ort::Value& score_pred, Ort::Value& bbox_pred,
    Ort::Value& kps_pred, unsigned int stride, float score_threshold, float img_height,
    float img_width, std::vector<types::BoxfWithLandmarks>& bbox_kps_collection, string metric)
{
    unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
    nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

    auto stride_dims = score_pred.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int num_points = stride_dims.at(0);  // 12800
    const float* score_ptr = score_pred.GetTensorMutableData<float>(); // [1,12800,1]
    const float* bbox_ptr = bbox_pred.GetTensorMutableData<float>();   // [1,12800,4]
    const float* kps_ptr = kps_pred.GetTensorMutableData<float>();     // [1,12800,10]

    float ratio = scale_params.ratio;
    int dw = scale_params.dw;
    int dh = scale_params.dh;

    unsigned int count = 0;
    auto& stride_points = center_points[stride];

    for (unsigned int i = 0; i < num_points; ++i)
    {
        const float cls_conf = score_ptr[i];
        if (cls_conf < score_threshold) continue; // filter
        auto& point = stride_points.at(i);
        const float cx = point.cx; // cx
        const float cy = point.cy; // cy
        const float s = point.stride; // stride

        // bbox
        const float* offsets = bbox_ptr + i * 4;
        float l = offsets[0]; // left
        float t = offsets[1]; // top
        float r = offsets[2]; // right
        float b = offsets[3]; // bottom

        if (metric == "max") {

        }
        else {

        }


        types::BoxfWithLandmarks box_kps;
        float x1 = ((cx - l) * s - (float)dw) / ratio;  // cx - l x1
        float y1 = ((cy - t) * s - (float)dh) / ratio;  // cy - t y1
        float x2 = ((cx + r) * s - (float)dw) / ratio;  // cx + r x2
        float y2 = ((cy + b) * s - (float)dh) / ratio;  // cy + b y2
        box_kps.box.x1 = std::max(0.f, x1);
        box_kps.box.y1 = std::max(0.f, y1);
        box_kps.box.x2 = std::min(img_width - 1.f, x2);
        box_kps.box.y2 = std::min(img_height - 1.f, y2);
        box_kps.box.score = cls_conf;
        box_kps.box.label = 1;
        box_kps.box.label_text = "face";
        box_kps.box.flag = true;

        // landmarks
        const float* kps_offsets = kps_ptr + i * 10;
        for (unsigned int j = 0; j < 10; j += 2)
        {
            cv::Point2f kps;
            float kps_l = kps_offsets[j];
            float kps_t = kps_offsets[j + 1];
            float kps_x = ((cx + kps_l) * s - (float)dw) / ratio;  // cx + l x
            float kps_y = ((cy + kps_t) * s - (float)dh) / ratio;  // cy + t y
            kps.x = min(max(0.f, kps_x), img_width - 1.f);
            kps.y = min(max(0.f, kps_y), img_height - 1.f);
            box_kps.landmarks.points.push_back(kps);
        }
        box_kps.landmarks.flag = true;
        box_kps.flag = true;

        bbox_kps_collection.push_back(box_kps);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

    if (bbox_kps_collection.size() > nms_pre_)
    {
        sort(
            bbox_kps_collection.begin(), bbox_kps_collection.end(),
            [](const types::BoxfWithLandmarks& a, const types::BoxfWithLandmarks& b)
            { return a.box.score > b.box.score; }
        ); // sort inplace
        // trunc
        bbox_kps_collection.resize(nms_pre_);
    }

}

void SCRFD::nms(vector<types::BoxfWithLandmarks>& input,
    vector<types::BoxfWithLandmarks>& output,
    float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;
    std::sort(
        input.begin(), input.end(),
        [](const types::BoxfWithLandmarks& a, const types::BoxfWithLandmarks& b)
        { return a.box.score > b.box.score; }
    );
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<types::BoxfWithLandmarks> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float iou = static_cast<float>(input[i].box.iou_of(input[j].box));

            if (iou > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }

        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}




vector<vector<double>> Distance::distance2bbox(const vector<vector<double>>& points, const vector<vector<double>>& distance, const vector<int>& max_shape) {
    vector<vector<double>> result(points.size(), vector<double>(4));

    for (size_t i = 0; i < points.size(); ++i) {
        double x1 = points[i][0] - distance[i][0];
        double y1 = points[i][1] - distance[i][1];
        double x2 = points[i][0] + distance[i][2];
        double y2 = points[i][1] + distance[i][3];

        if (!max_shape.empty()) {
            x1 = min(max(x1, 0.0), static_cast<double>(max_shape[1]));
            y1 = min(max(y1, 0.0), static_cast<double>(max_shape[0]));
            x2 = min(max(x2, 0.0), static_cast<double>(max_shape[1]));
            y2 = min(max(y2, 0.0), static_cast<double>(max_shape[0]));
        }

        result[i][0] = x1;
        result[i][1] = y1;
        result[i][2] = x2;
        result[i][3] = y2;
    }

    return result;
}

vector<vector<double>> Distance::distance2kps(const vector<vector<double>>& points, const vector<vector<double>>& distance, const vector<int>& max_shape) {
    vector<vector<double>> preds;

    for (size_t i = 0; i < distance[0].size(); i += 2) {
        vector<double> px(points.size());
        vector<double> py(points.size());

        for (size_t j = 0; j < points.size(); ++j) {
            px[j] = points[j][i % 2] + distance[j][i];
            py[j] = points[j][(i + 1) % 2] + distance[j][i + 1];

            if (!max_shape.empty()) {
                px[j] = min(max(px[j], 0.0), static_cast<double>(max_shape[1]));
                py[j] = min(max(py[j], 0.0), static_cast<double>(max_shape[0]));
            }
        }

        preds.push_back(px);
        preds.push_back(py);
    }

    vector<vector<double>> result(points.size(), vector<double>(distance[0].size()));
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < distance[0].size(); ++j) {
            if (j % 2 == 0)
                result[i][j] = preds[j][i];
            else
                result[i][j] = preds[j][i];
        }
    }

    return result;
}

