// Contains contour operations for Hand Detection. Such as finding the image contours in given image, 
// finding the nth biggest contours in a given list of contours, and determining which contour is bigger.
// Author: Quintin Nguyen, Akhil Lal, Matthew Cho

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <opencv2/core/types.hpp>
#include <vector>
#include <stdlib.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

struct Hand {
	Point location = Point(-1, -1);
	int type = -1;
};

double const min_contour_area_percent = 0.04;


// FindImageContours
// Preconditions: image is of the correct type and correctly allocated
// Postconditions: vector of contours within the image is returned
vector<vector<Point>> FindImageContours(const Mat& object) {
	Mat thresh;
	threshold(object, thresh, 90, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	return contours;
}

// FindNthBiggestContour
// Preconditions: contours list and box is of the correct type and are correctly
//                allocated, n is an constant integer
// Postconditions: Returns the index of the nth biggest contour
int FindNthBiggestContour(const vector<vector<Point>>& contours, 
	                      Rect& box, const int n, const int area) {
	int index = (int)contours.size() - n;
	if (contourArea(contours[index]) >= (area * min_contour_area_percent)) {
		box = boundingRect(contours[index]);
		return index;
	}
	return -1;
}

// CompareContourAreas
// Preconditions: contour1 and contour2 are of the correct type and are correctly allocated
// Postconditions: Returns true if contour2 is bigger than contour1, false if not
// Credit: StackOverFlow User dom
bool CompareContourAreas(const vector<Point> contour1, const vector<Point> contour2) {
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return (i < j);
}
