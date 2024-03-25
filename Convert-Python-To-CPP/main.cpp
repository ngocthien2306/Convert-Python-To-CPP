#include <stdio.h>
#include "lib.model.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void live_opencv() {
    string rtspUrl = "rtsp://admin:vuletech123@192.168.1.126:554/cam/realmonitor?channel=1&subtype=0";
    cv::Mat myImage;
    cv::namedWindow("Video Player");
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "No video stream detected" << endl;
        system("pause");
    }
    while (true) {
        cap >> myImage;
        if (myImage.empty()) {
            break;
        }
        Mat resizedImage;
        resize(myImage, resizedImage, Size(1280, 720));
        imshow("Video Player", resizedImage);
        char c = (char)cv::waitKey(1);
        if (c == 27) {
            break;
        }
    }
    cap.release();
}
int main(int argc, char** argv)
{
    string onnx_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/models/scrfd_500m_bnkps_shape640x640.onnx";
    string test_img_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/images/image.jpg";
    string save_img_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/images/output/output_cpp.jpg";

    const double nms_threshold = 0.0;
    const double iou_threshold = 0.5;
    const int max_nms = 10000;


    face::detect::SCRFD* scrfd = new face::detect::SCRFD(onnx_path);

    vector<lite::types::BoxfWithLandmarks> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    cv::Mat img_plot = img_bgr.clone();
    scrfd->detect2(img_bgr, detected_boxes, 0.0, 10000, "default", 0.7, 400);

    lite::utils::draw_boxes_with_landmarks_inplace(img_plot, detected_boxes);

    cv::imwrite(save_img_path, img_plot);

    return 0;

}