// Contains function relating to the position of the hand, location of the hand,
// and movement direction for Hand Detection. Used to mostly print various 
// information to the screen.
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

#define STAYING_STILL 0;
#define MOVE_DOWN 1;
#define MOVE_UP 2;
#define MOVE_LEFT 3;
#define MOVE_RIGHT 4;

struct Hand {
	Point location = Point(-1, -1);
	int type = -1;
};

Scalar const text_color = { 0, 255, 0 };
int const movement_threshold = 11;


// PrintHandLocation
// Precondition: Parameters are properly formatted and passed in correctly
// Postcondition: Will write the hand location on the passed in frame
void PrintHandLocation(Mat& frame, const Point hand_pos) {
	string hand_location = "Hand Location: (" + to_string(hand_pos.x) + ", " +
		                    to_string(hand_pos.y) + ")";
	putText(frame, hand_location, Point{ 3, frame.rows - 6 }, 1, 1.5, text_color, 2);
}

// MovementDirectionShape
// Precondition: Parameter is properly formatted and passed in correctly
// Postcondition: Will return an image based on the direction that was passed in
Mat MovementDirectionShape(const int direction) {
	Mat shape;
	if (direction == -1) {
		shape = imread("none.jpg");
	}
	else if (direction == 0) {
		shape = imread("stay.jpg");
	}
	else {
		shape = imread("arrow.jpg");
		if (direction == 1) {	//Down
			rotate(shape, shape, ROTATE_90_CLOCKWISE);
		}
		else if (direction == 2) {	//Up
			rotate(shape, shape, ROTATE_90_COUNTERCLOCKWISE);
		}
		else if (direction == 3) {	//Left
			rotate(shape, shape, ROTATE_180);
		}
		else {}	//Right
	}
	return shape;
}

// PrintHandType
// Preconditions: frame is of the correct type and correctly allocated, h_type
//                is a constant integer
// Postconditions: Text representing the hand position matched is put on the screen
void PrintHandType(Mat& frame, const int h_type) {
	string type;
	if (h_type == 1)
		type = "1 Finger Up";
	else if (h_type == 2)
		type = "2 Fingers Up";
	else if (h_type == 3)
		type = "3 Fingers Up";
	else if (h_type == 4)
		type = "4 Fingers Up";
	else if (h_type == 5)
		type = "5 Fingers Up";
	else
		type = "No Hand Detected";
	string hand_type = "Hand Type: " + type;
	putText(frame, hand_type, Point{ 3, frame.rows - 30 }, 1, 1.5, text_color, 2);
}

// HandMovementDirection
// Precondition: Parameters are properly formatted and passed in correctly
// Postcondition: Will return an integer that tells which way the hand moved.
//                Either no moving/no hand detected, or up, down, left, or right
int HandMovementDirection(const Hand& current, const Hand& previous) {
	int change_in_x = current.location.x - previous.location.x;
	int change_in_y = current.location.y - previous.location.y;
	if (current.type == -1) {
		return -1;
	}
	if (abs(change_in_x) >= abs(change_in_y)) {
		if (abs(change_in_x) > movement_threshold && previous.type != -1) {
			if (change_in_x > 0) {
				return MOVE_RIGHT;
			}
			else return MOVE_LEFT;
		}
	}
	else {
		if (abs(change_in_y) > movement_threshold && previous.type != -1) {
			if (change_in_y > 0) {
				return MOVE_DOWN;
			}
			else return MOVE_UP;
		}
	}
	return STAYING_STILL;
}
