// Contains main function for Hand-Detection. Goes through each frame in the input video and detects 
// the hand, and displays its coordinates, position, and movement in a returned video. Uses various
// functions from other files that help detect the hand.
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

string const video_name_path = "hand.mp4";
int const skip_frames = 3;
Scalar const box_color = Scalar{ 0, 0, 255 };

Mat ExtractBackground(VideoCapture& video);
void PrepareImage(Mat& image);
Mat BackgroundRemover(const Mat& front, const Mat& back);
vector<vector<Point>> FindImageContours(const Mat& object);
bool CompareContourAreas(const vector<Point> contour1, const vector<Point> contour2);
Hand SearchForHand(const Mat& front, const vector<vector<Point>>& contours, Rect& box);
void PrintHandType(Mat& frame, const int h_type);
void PrintHandLocation(Mat& frame, const Point hand_pos);
int HandMovementDirection(const Hand& current, const Hand& previous);
Mat MovementDirectionShape(const int direction);


// Main Method
// Precondition: hand.mp4 exists in the code directory and is a valid mp4 video file.
// Postcondition: Video output.avi gets outputted that identifies a hand with a box surrounding
//                the hand, hand type and location is displayed on screen. And the movement
//                direction of the hand is also displayed
int main(int argc, char argv[]) {
	VideoCapture cap(video_name_path);
	if (!cap.isOpened()) return -1;

	int const frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int const frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

	Mat background = ExtractBackground(cap);
	PrepareImage(background);

	Mat frame;
	Hand current_hand;
	Hand previous_hand;
	Mat original_frame(frame_height, frame_width, CV_8UC3);
	Mat front(frame_height, frame_width, CV_8UC3);

	VideoWriter output_vid("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
		30, Size(frame_width, frame_height));

	int frame_num = 1;
	int previous_shape_type = -1;
	Rect prev_box;

	while (true) {
		cap >> frame;				// Reads in image frame
		if (!frame.data) break;	// if there's no more frames then break
		if (frame_num % skip_frames == 0) {	//decreases the number of frames being analyzed
			original_frame = frame.clone();

			PrepareImage(frame);
			front = BackgroundRemover(frame, background);

			vector<vector<Point>> contours = FindImageContours(front);
			sort(contours.begin(), contours.end(), CompareContourAreas);
			Rect box;
			current_hand = SearchForHand(front, contours, box);

			//Print info to screen
			PrintHandType(original_frame, current_hand.type);
			PrintHandLocation(original_frame, current_hand.location);
			int shape_type = HandMovementDirection(current_hand, previous_hand);
			Mat shape = MovementDirectionShape(shape_type);
			shape.copyTo(original_frame(Rect(0, 0, shape.cols, shape.rows)));
			if (current_hand.type != -1) {
				rectangle(original_frame, box, box_color, 2);
				prev_box = box;
			}
			previous_shape_type = shape_type;
			previous_hand.location = current_hand.location;
			previous_hand.type = current_hand.type;

			output_vid.write(original_frame);
			frame_num++;
		}
		else {
			PrintHandType(frame, previous_hand.type);
			PrintHandLocation(frame, previous_hand.location);
			Mat shape = MovementDirectionShape(previous_shape_type);
			shape.copyTo(frame(Rect(0, 0, shape.cols, shape.rows)));
			if (previous_hand.type != -1) {
				rectangle(frame, prev_box, box_color, 2);
			}
			output_vid.write(frame);
			frame_num++;
		}
	}
	output_vid.release();
	cap.release();
	return 0;
}