
using namespace cv;

class objectDetection {

public:

	objectDetection(); //constructor 

	//Load images from the pattern and returns a vector of Mat with input images
	//std::vector<Mat> loadImages();


	//Perform feature detection with SIFT algorithm
	//A vector of keypoints is returned
	Mat SIFTFeatures(Mat image);
	std::vector<KeyPoint> SIFTKeypoints(Mat image);

	//Perform feature detection with ORB algorithm
	//A vector of keypoints is returned
	//std::vector<KeyPoint>  ORBFeatures(Mat image);

	//Perfom the matching of keypoints over all neighbor couples, in 3 phases: 
	//1. Matching between the different features extracted,
	//2. Refine the matches found by selecting ones with distance less than ratio * min_distance, where ratio is a parameter 
	//3. Filter out outlier using findHomography()
	// visual is a parameter used to decide if yuo want to visualize the matches
	//return value for wrong matches
	
	static std::vector<DMatch> matchImages(float ratio, bool visual, int dist, Mat obj_desc, Mat frame_desc, std::vector<KeyPoint> obj_key, std::vector<KeyPoint> frame_key);

	static std::vector<Point2f> findProjection(Mat obj, Mat frame, std::vector<KeyPoint> obj_key, std::vector<KeyPoint> frame_key, std::vector<DMatch> matches);


protected:

	//std::vector<std::string> names; //names of immages to be loaded
	//std::vector<Mat> input_images, proj_images; //vectors of images
	std::vector<KeyPoint> keypoints, refine_keypoints; //a vector of keypoints/refined keypoints for each image
	Mat descriptor; //a descriptor Mat for each image
	//std::vector<DMatch> matches, refine_matches, inlier_matches; //a vector of matches/refined matches/inlier matches for each image

	//std::string path, pattern;

};
