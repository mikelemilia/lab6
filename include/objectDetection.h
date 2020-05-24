
using namespace cv;
using namespace std;

class objectDetection {

public:

	objectDetection(); //constructor 

	//Load images from the pattern and returns a vector of Mat with input images
	//vector<Mat> loadImages();


	//Perform feature detection with SIFT algorithm
	//A vector of keypoints is returned
	Mat SIFTFeatures(Mat &image);
	vector<KeyPoint> SIFTKeypoints(Mat &image);

	//Perform feature detection with ORB algorithm
	//A vector of keypoints is returned
	//vector<KeyPoint>  ORBFeatures(Mat image);

	//Perfom the matching of keypoints over all neighbor couples, in 3 phases: 
	//1. Matching between the different features extracted,
	//2. Refine the matches found by selecting ones with distance less than ratio * min_distance, where ratio is a parameter 
	//3. Filter out outlier using findHomography()
	// visual is a parameter used to decide if yuo want to visualize the matches
	//return value for wrong matches
	
	vector<DMatch> matchImages(float ratio, int dist, Mat &obj_desc, Mat &frame_desc, vector<KeyPoint> &obj_key, vector<KeyPoint> frame_key);

    vector<Point2f> findProjection(Mat &obj, vector<Point2f> &obj_key, const vector<Point2f>& frame_key);

    Mat drawBox(Mat img, vector<Point2f> scene_corners, Scalar &color);

protected:

	//vector<string> names; //names of immages to be loaded
	//vector<Mat> input_images, proj_images; //vectors of images
	vector<KeyPoint> keypoints, refine_keypoints; //a vector of keypoints/refined keypoints for each image
	Mat descriptor; //a descriptor Mat for each image
	//vector<DMatch> matches, refine_matches, inlier_matches; //a vector of matches/refined matches/inlier matches for each image

	//string path, pattern;

};
