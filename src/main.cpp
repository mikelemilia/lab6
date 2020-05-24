#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <random>

#include "../include/objectDetection.h"

using namespace cv;
using namespace std;

int main() {

    //----------------------
    //loading data
    //----------------------
    VideoCapture cap("../data/video.mov");

    vector<String> names;
    vector<Mat> objects;

    vector<KeyPoint> frame_key;
    Mat frame_desc;

    Mat vis;

    float ratio = 3;

    objectDetection detector;

    glob("../data/objects/obj*.png", names, false);

    for (auto &name : names) {
        objects.push_back(imread(name));

    }

    //----------------------
    //objects feature detection
    //----------------------

    vector<Mat> obj_desc;
    vector<vector<KeyPoint>> obj_key;

    for (auto obj : objects) {
        obj_key.push_back(detector.SIFTKeypoints(obj));
        obj_desc.push_back(detector.SIFTFeatures(obj));
    }

    if (cap.isOpened()) { // check if we succeeded

        //----------------------
        //elaboration of first frame
        //----------------------

        int frames, frame_rate;
        Mat frame, frameGray;
        frames = cap.get(CAP_PROP_FRAME_COUNT);
        frame_rate = cap.get(CAP_PROP_FPS);
        cap >> frame;

        //----------------------
        //frame feature detection
        //----------------------

        frame_key = detector.SIFTKeypoints(frame);
        frame_desc = detector.SIFTFeatures(frame);

        vector<vector<DMatch>> match;
        vector<vector<Point2f>> vertex(objects.size());
        vector<Scalar> color;
        random_device randomDevice;
        mt19937 mt(randomDevice());
        uniform_int_distribution<int> rnd(0, 255);
        for (auto obj : objects) {
            color.emplace_back(Scalar(rnd(mt),rnd(mt),rnd(mt)));
        }

        //----------------------
        //feature matching
        //----------------------

        for (int i = 0; i < objects.size(); i++) {
            match.push_back(detector.matchImages(ratio, NORM_L2, obj_desc[i], frame_desc, obj_key[i], frame_key));

            drawMatches(objects[i], obj_key[i], frame, frame_key, match[i], vis); //visualize each match
            namedWindow("KEYPOINTS", WINDOW_NORMAL);
            resizeWindow("KEYPOINTS", Size(1000, 500));
            imshow("KEYPOINTS", vis);
            waitKey(0);
        }

        //------------------
        //get keypoints from frame and object to be tracked
        //-------------------------

        vector<Point2f> track_keypoints;
        vector<vector<Point2f>> obj_track_points(objects.size());
        vector<int> index; //where the ith object keypoints start

        int i = 0, j = 0;
        for (auto &v :match) {

            index.push_back(i);

            for (auto &p : v) {
                obj_track_points[j].push_back(obj_key[j][p.queryIdx].pt);
                track_keypoints.push_back(frame_key[p.trainIdx].pt);
            }
            i = track_keypoints.size();
            j++;
        }
        index.push_back(track_keypoints.size());

        //----------------------
        //draw box preview
        //----------------------

        int k = 0;
        int start, end;
        for (auto &obj : objects) {

            start = index[k];
            end = index[k + 1];
            vertex[k] = detector.findProjection(objects[k], obj_track_points[k],
                                                vector<Point2f>(track_keypoints.begin() + start,
                                                                track_keypoints.begin() + end));
            frame = detector.drawBox(frame, vertex[k], color[k]);
            k++;
        }

        drawKeypoints(frame, frame_key, vis);
        namedWindow("KEYPOINTS", WINDOW_NORMAL);
        imshow("KEYPOINTS", vis);
        waitKey(0);
        destroyAllWindows();


        //----------------------
        //pre-processing for tracking
        //----------------------

        Mat prev_frame;
        vector<Mat> pyramid_prev, pyramid_next;
        Mat status;
        vector<Point2f> shift_points, shift_vertex;
        Size wind_size = Size(7, 7);
        int maxlevel = 3;

        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        buildOpticalFlowPyramid(frameGray, pyramid_prev, wind_size, maxlevel);
        j = 1;
        string framerate, timebar;
        double fps, avg_fps = 0;
        int pause;
        long long duration;


        for (;;) {

            auto t1 = chrono::high_resolution_clock::now();
            auto tr = chrono::high_resolution_clock::now();
            cap >> frame;

            if (frame.empty())
                break; // reach to the end of the video file

            //for efficiency we discard one frame out of two for the pyramidal optical flow estimation
            if ((j % 2 == 0) || (j == 1)) {
            cvtColor(frame, frameGray, COLOR_BGR2GRAY);
            buildOpticalFlowPyramid(frameGray, pyramid_next, wind_size, maxlevel); //<---- BOTTLENECK

            //These lines are used to estimate the bottleneck of the elaboration
            //auto t2 = chrono::high_resolution_clock::now();
            //duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            //auto tr = t2;
            //cout << "PYRAMID: " << duration * 1.0e-3 << " ms" << endl;

            calcOpticalFlowPyrLK(pyramid_prev, pyramid_next, track_keypoints, shift_points, status, noArray(),
                                 wind_size, maxlevel,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 15, 0.05));
            //auto t2 = chrono::high_resolution_clock::now();
            //duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            //auto tr = t2;
            //cout << "LUKAS-KANADE: " << duration * 1.0e-3 << " ms" << endl;
            }

            //----------------------
            //estimate the vertex shift and rotation for each object using optical flow
            //----------------------
            k = 0;
            for (auto &obj : objects) {

                start = index[k];
                end = index[k + 1];
                vertex[k] = detector.findProjection(obj, obj_track_points[k],
                                                    vector<Point2f>(shift_points.begin() + start,
                                                                    shift_points.begin() + end));
                frame = detector.drawBox(frame, vertex[k], color[k]);

                k++;

            }

            //auto t2 = chrono::high_resolution_clock::now();
            //duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            //auto tr = t2;
            //cout << "DRAW BOX: " << duration * 1.0e-3 << " ms" << endl;


            //----------------------
            //drawing keypoints witth differetn colors
            //----------------------
            Scalar hue;
            for (int h = 0; h < shift_points.size(); h++) {

                for (k = 1; k < index.size(); k++) {
                    if (h >= index[k - 1] && h < index[k])
                        hue = color[k - 1];
                }
                circle(frame, shift_points[h], 3, hue, -1);
            }

            //auto t2 = chrono::high_resolution_clock::now();
            //duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            //auto tr = t2;
            //cout << "DRAW POINTS: " << duration * 1.0e-3 << " ms" << endl;
            //cout << "------------------" << endl;


            //----------------------
            //update pyramid
            //----------------------
            track_keypoints = shift_points;
            i = 0;
            for (auto &p : pyramid_next) {
                p.copyTo(pyramid_prev[i++]);
            }
            j++;

            //----------------------
            //compute statistics about frame flow
            //----------------------
            auto t2 = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
            pause = max(1, (int)(frame_rate - duration * 1.0e-3));
            duration = duration + (long long)pause*1.0e3; //total frame duration is execution time + pause, we neglet the least intructions
            //std::cout << "FRAME DURATION [ms]" << duration*1.0e-3 << std::endl;

            if (j % 10 != 0)
                avg_fps = avg_fps + 1 / (duration * 1.0e-6);
            else {
                avg_fps = avg_fps / 10;
                fps = avg_fps;
                framerate = "FPS: " + to_string(fps);
                avg_fps = 0;
            }

            timebar = "VIDEO PROCEEDING: " + to_string((int) (j * 100.0 / frames)) + "%";

            putText(frame, framerate, Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255));
            putText(frame, timebar, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
            namedWindow("TRACKING", WINDOW_NORMAL);
            imshow("TRACKING", frame);

            //dynamical adjustment of frame rate (if possible)
            //if (30 - (duration * 1.0e-3) > 1)
            //	pause = 30 - (duration * 1.0e-3);
            //else
            //	pause = 1;
            
            //std::cout << "Duration: " << duration * 1.0e3 << " - Pause: " << pause << std::endl;

            if (waitKey(pause) >= 0) break;

        }
    }


    cout << "END, press any key to exit" << endl;
    waitKey(0);
}


