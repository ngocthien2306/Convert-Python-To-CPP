#include <stdio.h>
#include "lib.model.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void inference_video(face::detect::SCRFD* scrfd, std::string test_video_path, std::string save_video_path) {
    const double score_threshold = 0.25;
    const double iou_threshold = 0.5;
    const int max_nms = 10000;
    const int topk = 100;

    cv::VideoCapture cap(test_video_path);
    if (!cap.isOpened()) {
        std::cout << "Error: Couldn't open the video file" << std::endl;
        return;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int current_frame = 0;

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter video_out(save_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame, frame_plot;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    double frame_count = 0;

    while (cap.read(frame)) {
        frame_plot = frame.clone();
        std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
        scrfd->detect(frame, detected_boxes, score_threshold, max_nms, "", iou_threshold, topk);
        lite::utils::draw_boxes_with_landmarks_inplace(frame_plot, detected_boxes);

        // Calculate FPS
        current_frame++;
        double progress = static_cast<double>(current_frame) / total_frames * 100.0;
        auto cur_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = cur_time - last_time;
        last_time = cur_time;
        frame_count++;
        double current_fps = 1.0 / elapsed.count();

        // Draw FPS on the frame
        cv::putText(frame_plot, "FPS: " + std::to_string(current_fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Display the processed frame
        cv::imshow("Detection", frame_plot);
        video_out.write(frame_plot);

        // Wait for a key press
        int key = cv::waitKey(1);
        if (key == 27) // esc key
            break;

        // Print progress and FPS to console
        std::cout << "\rProcessing frame " << current_frame << "/" << total_frames << " [" << std::fixed << std::setprecision(2) << progress << "%] FPS: " << std::fixed << std::setprecision(2) << current_fps << " " << (cur_time - start_time).count() / 1e9 << " s";
        std::cout.flush();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\nProcessing completed in " << elapsed.count() << " seconds." << std::endl;

    cap.release();
    video_out.release();
    cv::destroyAllWindows(); // Close all OpenCV windows
}

void inference_image(face::detect::SCRFD* scrfd, std::string test_img_path, std::string save_img_path) {
    const double score_threshold = 0.2;
    const double iou_threshold = 0.5;
    const int max_nms = 10000;
    const int topk = 100;

    cv::Mat img_bgr = cv::imread(test_img_path);
    if (img_bgr.empty()) {
        std::cerr << "Error: Couldn't open the image file" << std::endl;
        return;
    }

    cv::Mat img_plot = img_bgr.clone();
    std::vector<lite::types::BoxfWithLandmarks> detected_boxes;

    auto start_time = std::chrono::high_resolution_clock::now();

    scrfd->detect(img_bgr, detected_boxes, score_threshold, max_nms, "", iou_threshold, topk);

    auto end_time = std::chrono::high_resolution_clock::now();

    start_time = std::chrono::high_resolution_clock::now();

    scrfd->detect(img_bgr, detected_boxes, score_threshold, max_nms, "", iou_threshold, topk);

    end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;

    lite::utils::draw_boxes_with_landmarks_inplace(img_plot, detected_boxes);

    cv::imshow("Detection", img_plot);

    cv::imwrite(save_img_path, img_plot);


    std::cout << "Detection completed in " << elapsed.count() << " seconds." << std::endl;
}
int main(int argc, char** argv)
{
    string onnx_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/models/scrfd_500m_bnkps_shape640x640.onnx";
    string test_img_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/images/image.jpg";
    string save_img_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/images/output/output_cpp.jpg";
    string test_video_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/videos/videov.mp4";
    string save_video_path = "C:/Users/Admin/source/AI/AI-Engineer-Tests/public/videos/output/output_cpp.mp4";

    face::detect::SCRFD* scrfd = new face::detect::SCRFD(onnx_path);
   
    // inference_image(scrfd, test_img_path, save_img_path);
    inference_video(scrfd, test_video_path, save_video_path);
    return 0;

}