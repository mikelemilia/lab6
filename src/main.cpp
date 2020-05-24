#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream> 
#include <stdlib.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <numeric> 
#include <opencv2/calib3d/calib3d.hpp>
#include "../include/objectDetection.h"
#include <opencv2/video/tracking.hpp>
#include <tgmath.h>
#include <chrono>

using namespace cv;

//std::vector<Point2f> findProjection(Mat obj, Mat frame, std::vector<KeyPoint> obj_key, std::vector<KeyPoint> frame_key, std::vector<DMatch> matches);
//std::vector<Point2f> findProjection(Mat obj, Mat frame, std::vector<Point2f> obj_key, std::vector<Point2f> frame_key);
//std::vector<DMatch> matchImages(float ratio, bool visual, int dist, Mat obj_desc, Mat frame_desc, std::vector<KeyPoint> obj_key, std::vector<KeyPoint> frame_key);
//Mat drawBox(Mat img, Mat img_object, std::vector<Point2f> scene_corners, Scalar color);
//std::vector<Point2f> shiftObj(std::vector<Point2f> vertex, std::vector<Point2f> track_keypoints, std::vector<Point2f> shift_points, int start, int end);


int main() {
	
	VideoCapture cap("../data/video.mov");

	std::vector<String> names;
	std::vector<Mat> objects;

	std::vector<KeyPoint> frame_key;
	Mat frame_desc;

	Mat vis;

	int ratio = 3;

	objectDetection detector;

	glob("../data/objects/obj*.png", names, false);

	for (auto name : names)
	{
		objects.push_back(imread(name));

	}

	std::vector<Mat> obj_desc;
	std::vector<std::vector<KeyPoint>> obj_key;

	for (auto obj : objects) {
		obj_key.push_back(detector.SIFTKeypoints(obj));
		obj_desc.push_back(detector.SIFTFeatures(obj));
	}
		
	if (cap.isOpened()) // check if we succeeded 
	{

		int frames;
		Mat frame, frameGray;
		frames = cap.get(CAP_PROP_FRAME_COUNT);
		cap >> frame;

		frame_key = detector.SIFTKeypoints(frame);
		frame_desc = detector.SIFTFeatures(frame);

		std::vector<std::vector<DMatch>> match;
		std::vector<std::vector<Point2f>> vertex(objects.size());
		std::vector<Scalar> color;
		

		for (auto obj : objects) {
			color.push_back(Scalar(rand()%256, rand()%256, rand()%256));
		}

		for (int i = 0; i < objects.size(); i++) {
			match.push_back(detector.matchImages(ratio, false, NORM_L2, obj_desc[i], frame_desc, obj_key[i], frame_key));

			drawMatches(objects[i], obj_key[i], frame, frame_key, match[i], vis);
			namedWindow("KEYPOINTS", WINDOW_NORMAL);
			imshow("KEYPOINTS", vis);
			waitKey(0);
		}

		//get keypoints from frame to be tracked
		std::vector<Point2f> track_keypoints;
		std::vector<std::vector<Point2f>> obj_track_points(objects.size());
		std::vector<int> index; //where the ith objects keypoints start

		int i=0 ,j = 0;
		for (auto &v :match ){

			index.push_back(i);

			for (auto &p : v) {
				obj_track_points[j].push_back(obj_key[j][p.queryIdx].pt);
				track_keypoints.push_back(frame_key[p.trainIdx].pt);
			}
			i = track_keypoints.size();
			j++;
		}
		index.push_back(track_keypoints.size());

		int k = 0;
		int start, end;
		for (auto& obj : objects) {
			
			start = index[k];
			end = index[k + 1];
			vertex[k] = detector.findProjection(objects[k], frame, obj_track_points[k], std::vector<Point2f>(track_keypoints.begin() + start, track_keypoints.begin() + end));
			frame = detector.drawBox(frame, objects[k], vertex[k], color[k]);
			k++;
		}

		drawKeypoints(frame, frame_key, vis);
		namedWindow("KEYPOINTS", WINDOW_NORMAL);
		imshow("KEYPOINTS", vis);
		waitKey(0);
		destroyAllWindows();

		Mat prev_frame;
		std::vector<Mat> pyramid_prev, pyramid_next;
		Mat status;
		std::vector<Point2f> shift_points, shift_vertex;
		Size wind_size = Size(7, 7);
		int maxlevel = 3;

		cvtColor(frame, frameGray, COLOR_BGR2GRAY);
		buildOpticalFlowPyramid(frameGray, pyramid_prev, wind_size, maxlevel);
		j = 1;
		std::string framerate, timebar;
		double fps, avg_fps = 0;
		int pause=1;
		std::chrono::steady_clock::time_point t1, t2, tr;
		long long duration;


		for(;;) { 
		 
			t1 = std::chrono::high_resolution_clock::now();
			tr = std::chrono::high_resolution_clock::now();
			cap >> frame;

			if (frame.empty())
				break; // reach to the end of the video file
		
			if ( (j % 2 == 0) || (j==1)) {
				cvtColor(frame, frameGray, COLOR_BGR2GRAY);
				buildOpticalFlowPyramid(frameGray, pyramid_next, wind_size, maxlevel); //<---- BOTTLENECK
				//t2 = std::chrono::high_resolution_clock::now();
				//duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - tr).count();
				//tr = t2;
				//std::cout << "PYRAMID: " << duration * 1.0e-3 << " ms" << std::endl;

				calcOpticalFlowPyrLK(pyramid_prev, pyramid_next, track_keypoints, shift_points, status, noArray(), wind_size, maxlevel, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 15, 0.05));
				//t2 = std::chrono::high_resolution_clock::now();
				//duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - tr).count();
				//tr = t2;
				//std::cout << "LUKAS-KANADE: " << duration * 1.0e-3 << " ms" << std::endl;
			}

			//mean shift
			int start, end;
			int k = 0;
			for (auto& obj : objects) {

				start = index[k];
				end = index[k + 1];
				vertex[k] = detector.findProjection(obj, frame, obj_track_points[k], std::vector<Point2f>(shift_points.begin() + start, shift_points.begin() + end));

				frame = detector.drawBox(frame, objects[k], vertex[k], color[k]);

				k++;

			}

			//t2 = std::chrono::high_resolution_clock::now();
			//duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - tr).count();
			//tr = t2;
			//std::cout << "DRAW BOX: " << duration * 1.0e-3 << " ms" << std::endl;

			Scalar hue;
			for (int i = 0; i < shift_points.size(); i++) {
				
				for (k = 1; k < index.size(); k++) {
					if (i >= index[k - 1] && i < index[k])
						hue = color[k - 1];
				}
				circle(frame,shift_points[i], 3, hue, -1);
			}

			//t2 = std::chrono::high_resolution_clock::now();
			//duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - tr).count();
			//tr = t2;
			//std::cout << "DRAW POINTS: " << duration * 1.0e-3 << " ms" << std::endl;
			//std::cout << "------------------" << std::endl;
			
			track_keypoints = shift_points; 
			int i = 0;
			for (auto& p : pyramid_next) {
				p.copyTo(pyramid_prev[i++]);
			}
			
			j++;

			t2 = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

			if(j%10 != 0)
			avg_fps = avg_fps + 1 / (duration * 1.0e-6 );
			else {
				avg_fps = avg_fps / 10;
				fps = avg_fps;
				framerate = "FPS: " + std::to_string(fps);
				avg_fps = 0;
			}
			
			timebar = "VIDEO PROCEEDING: " + std::to_string( (int)( j * 100.0 / frames) ) +"%";
	

			putText(frame, framerate, Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255));
			putText(frame, timebar, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
			namedWindow("TRACKING", WINDOW_NORMAL);
			imshow("TRACKING", frame);

			//dynamical adjustment of frame rate (if possible)
			//if (30 - (duration * 1.0e-3) > 1)
			//	pause = 30 - (duration * 1.0e-3);
			//else
			//	pause = 1;
			pause = min(1, (int)(30 - (duration * 10e-6)) );

			if (waitKey(pause) >= 0) break;
			
		} 
	}


	std::cout << "END, press any key to exit" << std::endl;
	system("pause"); 

}





//std::vector<Point2f> shiftObj(std::vector<Point2f> vertex, std::vector<Point2f> track_keypoints, std::vector<Point2f> shift_points, int start, int end) {
//
//	
//	Point2f shift = Point2f(0, 0);
//	for (int i = start; i < end; i++) {
//		shift = shift + (shift_points[i] - track_keypoints[i]);
//	}
//	shift = Point2f(shift.x / (end-start), shift.y / (end-start));
//	std::cout << "SHIFT: " << shift << std::endl;
//
//	
//	for (int i = 0; i < vertex.size(); i++) {
// 			vertex[i] = vertex[i] + shift;
//	}
//
//	return vertex;
//
//}


//std::vector<Point2f> findProjection(Mat obj, Mat frame, std::vector<Point2f> obj_key, std::vector<Point2f> frame_key) {
//	
//	std::vector<Point2f> obj_vertex(4);
//	Mat temp, H;
//	std::vector<Point2f> vertex;
//
//	//compute homography between frame and object
//	H = findHomography(obj_key, frame_key, RANSAC, 3, noArray(), 10, 0.99);
//
//	obj_vertex[0] = Point2f(0, 0);
//	obj_vertex[1] = Point2f(obj.cols, 0);
//	obj_vertex[2] = Point2f(0, obj.rows);
//	obj_vertex[3] = Point2f(obj.cols, obj.rows);
//
//	perspectiveTransform(obj_vertex, vertex, H);
//
//	return vertex;
//
//}

//std::vector<DMatch> matchImages(float ratio, bool visual, int dist, Mat obj_desc, Mat frame_desc, std::vector<KeyPoint> obj_key, std::vector<KeyPoint> frame_key) {
//
//	Mat vis, mask;
//	float min_dist=-1;
//
//	std::vector<DMatch> matches, refine_matches, inlier_matches;
//
//	Ptr<BFMatcher> matcher = BFMatcher::create(dist);
//
//	matcher->match(obj_desc, frame_desc, matches);
//
//	//REFINE MATCHES
//
//	//seek for the minimun distance among the matches of the actual couple
//	for (auto& match : matches) {
//		if ((match.distance < min_dist) || (min_dist == -1))
//			min_dist = match.distance;
//	}
//
//	//discard all matches with distance > ratio * min_dist
//	for (auto& match : matches) {
//
//		if (match.distance <= ratio * min_dist) {
//			refine_matches.push_back(match);
//		}
//	}
//
//	//INLIER FILTERING
//
//	std::vector<Point2f> points1, points2;
//
//	//retrieve matched keypoints and convert them in Point2f
//	for (auto& match : refine_matches) {
//		points1.push_back(obj_key[match.queryIdx].pt);
//		points2.push_back(frame_key[match.trainIdx].pt);
//	}
//
//	//compute homography
//	findHomography(points1, points2, RANSAC, 3, mask);
//
//	//discard all outlies points
//	for (int j = 0; j < mask.rows; j++) {
//
//		if (mask.at<uchar>(j, 0) == 1)
//			inlier_matches.push_back(refine_matches[j]);
//	}
//
//
////for (int i = 0; i < inlier_matches.size(); i++)
////	if (inlier_matches[i].size() < 1) {
////		std::cout << "ERROR: no inlier matches in couple " << i << ", try with larger ratio" << std::endl;
////		refine_matches.clear();
////		inlier_matches.clear();
////		refine_matches.resize(names.size() - 1);
////		inlier_matches.resize(names.size() - 1);
////		return 0;
////	}
//
//	return inlier_matches;
//}
//
//
//Mat drawBox(Mat img, Mat img_object, std::vector<Point2f> scene_corners, Scalar color) {
//
//	line(img, scene_corners[0], scene_corners[1], color, 4);
//	line(img, scene_corners[0], scene_corners[2], color, 4);
//	line(img, scene_corners[1], scene_corners[3], color, 4);
//	line(img, scene_corners[2], scene_corners[3], color, 4);
//
//	return img;
//}



