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

using namespace cv;


objectDetection::objectDetection() {}

std::vector<KeyPoint> objectDetection::SIFTKeypoints(Mat image) {

	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

	sift->detect(image, keypoints);

	return keypoints;

}

Mat objectDetection::SIFTFeatures(Mat image) {

	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

	sift->compute(image, keypoints, descriptor);

	return descriptor;

}

//objectDetection::objectDetection() {}
//
//std::vector<KeyPoint> objectDetection::ORBKeypoints(Mat image) {
//
//	Ptr<ORB> orb = xfeatures2d::SIFT::create();
//
//	orb->detect(image, keypoints);
//
//	return keypoints;
//
//}
//
//Mat objectDetection::ORBFeatures(Mat image) {
//
//	Ptr<ORB> orb = xfeatures2d::SIFT::create();
//
//	orb->compute(image, keypoints, descriptor);
//
//	return descriptor;
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
//		//REFINE MATCHES
//
//		//seek for the minimun distance among the matches of the actual couple
//		for (auto &match : matches) {
//			if ((match.distance < min_dist) || (min_dist == -1))
//				min_dist = match.distance;
//		}
//
//		//discard all matches with distance > ratio * min_dist
//		for (auto &match : matches) {
//
//			if (match.distance <= ratio * min_dist) {
//				refine_matches.push_back(match);
//			}
//		}
//
//		//INLIER FILTERING
//
//		std::vector<Point2f> points1, points2;
//
//		//retrieve matched keypoints and convert them in Point2f
//		for (auto &match : refine_matches ) {
//			points1.push_back(obj_key[match.queryIdx].pt);
//			points2.push_back(frame_key[match.trainIdx].pt);
//		}
//
//		//compute homography
//		findHomography(points1, points2, RANSAC, 3, mask);
//
//		//discard all outlies points
//		for (int j = 0; j < mask.rows; j++) {
//
//			if (mask.at<uchar>(j, 0) == 1)
//				inlier_matches.push_back(refine_matches[j]);
//		}
//
//		////visualization of mathces when required
//		//if (visual) {
//		//	drawMatches(proj_images[i], keypoints[i], proj_images[i + 1], keypoints[i + 1], inlier_matches[i], vis);
//		//	namedWindow("INLIER MATCHES", WINDOW_NORMAL);
//		//	imshow("INLIER MATCHES", vis);
//
//		//	waitKey();
//
//		//}
//
//
//
//	//std::cout << "Press any key to go on " << std::endl;
//	//waitKey();
//	//destroyAllWindows();
//
//	//for (int i = 0; i < inlier_matches.size(); i++)
//	//	if (inlier_matches[i].size() < 1) {
//	//		std::cout << "ERROR: no inlier matches in couple " << i << ", try with larger ratio" << std::endl;
//	//		refine_matches.clear();
//	//		inlier_matches.clear();
//	//		refine_matches.resize(names.size() - 1);
//	//		inlier_matches.resize(names.size() - 1);
//	//		return 0;
//	//	}
//
//	return inlier_matches;
//}
//
//
//std::vector<Point2f> findProjection(Mat obj, Mat frame, std::vector<KeyPoint> obj_key, std::vector<KeyPoint> frame_key, std::vector<DMatch> matches) {
//
//	std::vector<Point2f> points1, points2;
//	Mat temp, H;
//	std::vector<Point2f> vertex(4);
//
//	for (auto& match : matches) {
//		points1.push_back(obj_key[match.queryIdx].pt);
//		points2.push_back(frame_key[match.trainIdx].pt);
//	}
//
//	//compute homography
//	H = findHomography(points1, points2, RANSAC, 3, Mat());
//
//
//	temp = H*(Mat)obj.at<Point2f>(0, 0);
//	vertex.push_back((Point2f)temp);
//	temp = H * (Mat)obj.at<Point2f>(0, obj.cols);
//	vertex.push_back((Point2f)temp);
//	temp = H * (Mat)obj.at<Point2f>(obj.rows, obj.cols);
//	vertex.push_back((Point2f)temp);
//	temp = H * (Mat)obj.at<Point2f>(obj.rows, obj.cols);
//	vertex.push_back((Point2f)temp);
//
//	return vertex;
//
//}
//

