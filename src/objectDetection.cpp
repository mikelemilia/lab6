#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../include/objectDetection.h"

using namespace cv;
using namespace std;

objectDetection::objectDetection() = default;

vector<KeyPoint> objectDetection::SIFTKeypoints(Mat &image) {

	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

	sift->detect(image, keypoints);
	return keypoints;

}

Mat objectDetection::SIFTFeatures(Mat &image) {

	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

	sift->compute(image, keypoints, descriptor);

	return descriptor;

}

vector<Point2f> objectDetection::findProjection(Mat &obj, vector<Point2f> &obj_key, const vector<Point2f>& frame_key) {

	vector<Point2f> obj_vertex(4);
	Mat temp, H;
	vector<Point2f> vertex;

	//compute homography between frame and object
	H = findHomography(obj_key, frame_key, RANSAC, 3, noArray(), 100);

    obj_vertex[0] = Point2f(0, 0);
    obj_vertex[1] = Point2f(obj.cols, 0);
    obj_vertex[2] = Point2f(0, obj.rows);
    obj_vertex[3] = Point2f(obj.cols, obj.rows);

	if(!H.empty()){ // check if findHomography() return an empty Mat, if it did not have enough corresponding Points
        perspectiveTransform(obj_vertex, vertex, H);
	}

	return vertex;

}

vector<DMatch> objectDetection::matchImages(int ratio, int dist, Mat &obj_desc, Mat &frame_desc, vector<KeyPoint> &obj_key, const vector<KeyPoint> &frame_key) {

	Mat vis, mask;
	float min_dist = -1;

	vector<DMatch> matches, refine_matches, inlier_matches;

	Ptr<BFMatcher> matcher = BFMatcher::create(dist);

	matcher->match(obj_desc, frame_desc, matches);

	//REFINE MATCHES

	//seek for the minimun distance among the matches of the actual couple
	for (auto &match : matches) {
		if ((match.distance < min_dist) || (min_dist == -1))
			min_dist = match.distance;
	}

	//discard all matches with distance > ratio * min_dist
	for (auto &match : matches) {

		if (match.distance <= ratio * min_dist) {
			refine_matches.push_back(match);
		}
	}

	//INLIERS FILTERING

	vector<Point2f> points1, points2;

	//retrieve matched keypoints and convert them in Point2f
	for (auto& match : refine_matches) {
		points1.push_back(obj_key[match.queryIdx].pt);
		points2.push_back(frame_key[match.trainIdx].pt);
	}

	//compute homography
	findHomography(points1, points2, RANSAC, 3, mask);

	//discard all outliers points
	for (int j = 0; j < mask.rows; j++) {
		if (mask.at<uchar>(j, 0) == 1)
			inlier_matches.push_back(refine_matches[j]);
	}


	//for (int i = 0; i < inlier_matches.size(); i++)
	//	if (inlier_matches[i].size() < 1) {
	//		cout << "ERROR: no inlier matches in couple " << i << ", try with larger ratio" << endl;
	//		refine_matches.clear();
	//		inlier_matches.clear();
	//		refine_matches.resize(names.size() - 1);
	//		inlier_matches.resize(names.size() - 1);
	//		return 0;
	//	}

	return inlier_matches;
}


Mat  objectDetection::drawBox(Mat img, vector<Point2f> scene_corners, Scalar &color) {

	line(img, scene_corners[0], scene_corners[1], color, 4);
	line(img, scene_corners[0], scene_corners[2], color, 4);
	line(img, scene_corners[1], scene_corners[3], color, 4);
	line(img, scene_corners[2], scene_corners[3], color, 4);

	return img;
}


