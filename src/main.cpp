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

void initLab6(size_t argc, char *argv[], vector<String> &paths, int &ratio, int &wSize, int &levels);

int main(int argc, char **argv) {

    //----------------------
    // variables
    //----------------------

    vector<String> paths;               // paths to objects and video
    vector<String> objects_names;       // name of the objects
    vector<Mat> objects;                // vector that contain all the objects


    vector<KeyPoint> frame_key;
    Mat frame_desc;

    Mat vis;

    int ratio; //<- EXPERIMENTS
    int wSize;

    int maxLevel;  //<- EXPERIMENTS

    objectDetection detector;

    vector<Mat> obj_desc;
    vector<vector<KeyPoint>> obj_key;

    double frames, frame_rate;
    Mat frame, frameGray;

    //----------------------
    //loading data
    //----------------------

    initLab6(argc, argv, paths, ratio, wSize, maxLevel);

    VideoCapture cap(paths[0]);

    try {
        glob(paths[1] + "/*.png", objects_names, false);        // collecting all the objects
    } catch (Exception &e) {
        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);
    }

    for (auto &name : objects_names) {
        objects.push_back(imread(name));
    }

    if (!cap.isOpened()) {
        cerr << "Error! No video was found!" << endl;
    } else { // check if we succeeded

        //----------------------
        //objects feature detection
        //----------------------

        for (auto &obj : objects) {
            obj_key.push_back(detector.SIFTKeypoints(obj));
            obj_desc.push_back(detector.SIFTFeatures(obj));
        }

        //----------------------
        //elaboration of first frame
        //----------------------

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
            color.emplace_back(Scalar(rnd(mt), rnd(mt), rnd(mt))); //random color for esch object
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
            vertex[k] = detector.findProjection(objects[k], obj_track_points[k], vector<Point2f>(track_keypoints.begin() + start, track_keypoints.begin() + end));
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
        Size wind_size = Size(wSize, wSize); //<- EXPERIMENTS


        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        buildOpticalFlowPyramid(frameGray, pyramid_prev, wind_size, maxLevel);
        j = 1;
        string framerate, timebar;
        double fps, avg_fps = 0;
        int pause;
        long long duration;


        for (;;) {

            auto t1 = chrono::high_resolution_clock::now();
            auto tr = chrono::high_resolution_clock::now();
            cap >> frame;

            if (frame.empty()) {
                cout << "Video ended" << endl;
                break; // reach to the end of the video file
            }

            //for efficiency we discard one frame out of two for the pyramidal optical flow estimation
            //if ((j % 2 == 0) || (j == 1)) {
            cvtColor(frame, frameGray, COLOR_BGR2GRAY);
            buildOpticalFlowPyramid(frameGray, pyramid_next, wind_size,
                                    maxLevel); //<---- BOTTLENECK in WINDOWS, to avoid to compute more times the pyramid, we collect all the keypoints in the same vector and we use a list of indexes

            //These lines are used to estimate the bottleneck of the elaboration
            auto t2 = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            tr = t2;
            cout << "PYRAMID: " << duration * 1.0e-3 << " ms" << endl;

            calcOpticalFlowPyrLK(pyramid_prev, pyramid_next, track_keypoints, shift_points, status, noArray(),
                                 wind_size, maxLevel,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 15, 0.05));  //<- EXPERIMENTS
            t2 = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            tr = t2;
            cout << "LUKAS-KANADE: " << duration * 1.0e-3 << " ms" << endl;
            //}

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

            t2 = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            tr = t2;
            cout << "DRAW BOX: " << duration * 1.0e-3 << " ms" << endl;


            //----------------------
            //drawing keypoints with different colors
            //----------------------
            Scalar hue;
            for (int h = 0; h < shift_points.size(); h++) {

                for (k = 1; k < index.size(); k++) {
                    if (h >= index[k - 1] && h < index[k])
                        hue = color[k - 1];
                }
                circle(frame, shift_points[h], 3, hue, -1);
            }

            t2 = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(t2 - tr).count();
            tr = t2;
            cout << "DRAW POINTS: " << duration * 1.0e-3 << " ms" << endl;
            cout << "------------------" << endl;


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
            t2 = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

            //dynamical adjustment of frame rate (when possible)
            pause = max(1, (int) (frame_rate - duration * 1.0e-3));
            duration = duration + (long long) pause *
                                  1.0e3; //total frame duration is execution time + pause, we neglet the least intructions
            //cout << "FRAME DURATION [ms]" << duration*1.0e-3 << endl;

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

            if (waitKey(pause) >= 0) {
                int key = waitKey();
                cout << "Press any key to resume the video or <ESC> to stop the video" << endl;
                if (key == 27)
                    break;
            }

        }
    }

    destroyAllWindows();

    cout << "Termination: press <ENTER> to exit..." << endl;
    fflush(stdin);
    getc(stdin);
}

void initLab6(size_t argc, char *argv[], vector<String> &paths, int &ratio, int &wSize, int &levels) {

    const String keys =
            "{help h usage ? |<none>| Print help message }"
            "{@video         |      | Input video path}"
            "{@objects       |      | Input objects path }"
            "{@ratio         |3     | Ratio used to select good matches}"
            "{@windowSize    |7     | Size of the window passed to LK}"
            "{@levels        |3     | Number of pyramid levels}";

    CommandLineParser parser(argc, argv, keys);
    parser.about("\nCOMPUTER VISION - LAB6\n");
    if (parser.has("help")) {
        parser.printMessage();
        exit(1);
    }

    paths.emplace_back(parser.get<String>("@video"));
    paths.emplace_back(parser.get<String>("@objects"));

    ratio = parser.get<int>("@ratio");
    wSize = parser.get<int>("@windowSize");
    levels = parser.get<int>("@levels");

    if (!parser.check()) {
        parser.printErrors();
        exit(-1);
    }

}



