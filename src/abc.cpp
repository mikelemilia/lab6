#include "../include/ObjectDetection.h"
#include "../include/utils.h"

#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    vector<String> paths;
    vector<Mat> objects;

    ObjectDetection object;
    ObjectDetection scene;

    glob("../data/objects/obj2.png", paths);       // get the object paths

    for (auto &p : paths) objects.emplace_back(imread(p));      // insert the objects into

    VideoCapture cap("../data/video.mov"); // open the video

    if (!cap.isOpened()) {
        cout << "Could not initialize capturing...\n";
        return 0;
    } else {

        Mat image_matches;
        Mat frame, frame_prev;
        vector<Mat> frame_pyramid, frame_prev_pyramid;
        vector<Point2f> tracked_keypoint, scene_point2f;
        vector<u_int8_t> status;  // u_int8_t == unsigned char
        vector<u_int8_t> err;  // u_int8_t == unsigned char

        namedWindow("Results", CV_WINDOW_NORMAL);

        for (auto &obj : objects) {

            // Gestione primo frame

            cap >> frame;

            // Compute the keypoints of the object
            vector<KeyPoint> object_keypoints = object.siftKeypoints(obj);

            // Compute the descriptors of the object
            Mat object_descriptors = object.siftDescriptors(obj, object_keypoints);

            // Compute the keypoints of the scene
            vector<KeyPoint> scene_keypoints = scene.siftKeypoints(frame);

            frame.copyTo(frame_prev);

            // Gestione di tutti gli altri frame

            for (;;) {

                cap >> frame; // get a new frame from camera
                if (frame.empty()) break; // exit the loop if the frame is empty or the video reach it's end

                buildOpticalFlowPyramid(frame_prev, frame_prev_pyramid, Size(10,10), 3);
                buildOpticalFlowPyramid(frame, frame_pyramid, Size(10,10), 3);

                scene_point2f = to_Point2f(scene_keypoints);

                calcOpticalFlowPyrLK(frame_prev_pyramid, frame_pyramid, scene_point2f, tracked_keypoint, status, err, Size(10,10));

                // Compute the descriptors of the scene
                Mat scene_descriptors = scene.siftDescriptors(frame, scene_keypoints);

                // Compute the matches between descriptors of the object and of the scene
                vector<DMatch> match = scene.compute_matches(object_descriptors, scene_descriptors);

                // Find the object in the scene (if possible) and draw a bounding box on it
                image_matches = scene.draw_contours(object_keypoints, scene_keypoints, match, obj, frame);

//                cvtColor(frame, image_matches, CV_BGR2GRAY);
//                GaussianBlur(image_matches, image_matches, Size(7,7), 1.5, 1.5);
//                Canny(image_matches, image_matches, 0, 30, 3);

                // Show keypoints detected
                resize(image_matches, image_matches, Size(), 0.7, 0.7);
                imshow("Results", image_matches);
                if (waitKey(1) >= 0) break;

            }

        }
    }

    return 0;
}


vector<Point2f> to_Point2f(vector<KeyPoint> &vec){

    vector<Point2f> points;

    KeyPoint::convert(vec, points);

    return points;
}