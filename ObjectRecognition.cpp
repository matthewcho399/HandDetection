// Contains functions to determine if image contains a hand for Hand Detection. Such as finding the hand and
// determining its position by seeing how many fingers are held up.
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

double const ratio_thresh = 0.7;
int const local_skip_points = 5;

int FindNthBiggestContour(const vector<vector<Point>>& contours, Rect& box,
	                      const int n, const int area);


// FindTopEdge
// Preconditions: object is of the correct type and is correctly allocated
// Postconditions: Returns a vector of points of the found top edges
vector<Point> FindTopEdge(const Mat& object) {
	vector<Point> points;
	for (int i = 0; i < object.cols; i += local_skip_points) {
		for (int j = 0; j < object.rows; j++) {
			if (object.at<uchar>(j, i) == 255) { // White here
				points.push_back(Point(i, j));
				break;
			}
		}
	}
	return points;
}

// FindLocalMaximaMinima
// Preconditions: points is a list of found top edges that is computed from a picture.
//                middle represents the middle row of the entire contour area. Both
//                of these parameters must be computed first and passed in correctly.
// Postconditions: Returns the number of local minim points found.
// Credit: Original local minima and maxima algorithm by GeeksforGeeks, but has
//         since been heavily modified and contributed to to better fit the needs
//         of this program
int FindLocalMaximaMinima(const vector<Point>& points, const int middle) {
	vector<int> max, min;
	for (int i = 1; i < points.size() - 1; i++) {
		bool skip = false;
		int next = i + 1;
		int prev = i - 1;

		//If equal in the middle
		if (points[next].y == points[i].y) {
			if ((next + 1) < int(points.size())) {
				next++;
				skip = true;
			}
			else break;
		}
		if (points[prev].y == points[i].y && (prev - 1) >= 0) {
			prev--;
			skip = true;
		}

		// Condition for local minima 
		if ((points[prev].y > points[i].y) and
			(points[i].y < points[next].y))
			min.push_back(i);
		// Condition for local maxima 
		else if ((points[prev].y < points[i].y) and
			(points[i].y > points[next].y))
			max.push_back(i);

		if (skip) {
			i++;
			skip = false;
		}
	}

	//If equal in the start/end
	if (points[points.size() - 1].y == points[points.size() - 2].y) {
		if (points[points.size() - 2].y < points[points.size() - 3].y)
			min.push_back((int)points.size() - 2);
		else max.push_back((int)points.size() - 2);
	}
	if (points[0].y == points[1].y) {
		if (points[1].y < points[2].y)
			min.push_back(1);
		else max.push_back(1);
	}

	// Local min and max must be smaller than middle
	vector<int> true_minima;
	vector<int> true_maxima;
	for (int i = 0; i < min.size(); i++) {
		if (middle > points[min[i]].y)
			true_minima.push_back(min[i]);
	}
	for (int i = 0; i < max.size(); i++) {
		if (middle > points[max[i]].y)
			true_maxima.push_back(max[i]);
	}

	// Final Check
	if (true_minima.size() == 1) return 1;
	if (true_minima.size() > 0 && true_minima.size() < 6 &&
		true_maxima.size() > 0 && true_maxima.size() < 6) {
		if (true_minima.size() == true_maxima.size())
			return ((int)true_minima.size() + 1);
		else if ((int)true_minima.size() - true_maxima.size() == 1)
			return (int)true_minima.size();
	}
	return -1;
}





// SearchForHand
// Preconditions: The functions FindNthBiggestContour and FindLocalMaximaMinima exist and are fully 
//                implemented. front is a binary image. List of contours must already be computed
//                for front.
// Postconditions: A hand object is returned with the following values: the type and the x and y
//                 location coordinates. If a hand is not detected all hand values are -1.
Hand SearchForHand(const Mat& front, const vector<vector<Point>>& contours, Rect& box) {
	Hand hand;
	Mat only_object;
	for (int i = 1; i <= contours.size(); i++) {
		int contour_index = FindNthBiggestContour(contours, box, i, (front.rows * front.cols));
		if (contour_index == -1) {
			break;
		}
		Mat pic(front.rows, front.cols, CV_8U, Scalar::all(0));
		drawContours(pic, contours, contour_index, Scalar(255, 255, 255), FILLED);
		only_object = pic(box);

		int type = FindLocalMaximaMinima(FindTopEdge(only_object), (only_object.rows / 2));

		if (type != -1) {
			hand.type = type;
			hand.location.x = box.x;
			hand.location.y = box.y;
			return hand;
		}
	}
	return hand;
}