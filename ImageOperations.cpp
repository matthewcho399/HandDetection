// Contains functions that preform operations on images for Hand Detection. Function such as modifying its
// color qualities and pixel values, operations on the whole image, such as smoothing, and background
// extraction. These functions are used to mostly help prepare the image and help make the hand stand
// out more in the image.
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

double const contrast_num = 1.1;
int const sat_val = 28;
int const gaus_blur_size = 11;
int const gaus_blur_amount = 3;
int const median_blur = 7;
int const brightness_level = 40;
int const number_random_frames = 30;
int const background_remover_thresh = 20;
int const red_color_thresh = 190;


// FixComputedColor
// Precondition: Parameter is passed in correctly
// Postcondition: Will return num as an int and makes sure that
//                num will not be greater than 255 or less than 0.
int FixComputedColor(double num) {
	num = trunc(num);
	if (num > 255) {
		num = 255;
	}
	else if (num < 0) {
		num = 0;
	}
	return int(num);
}

// ModifyContrast
// Precondition: Parameters are passed in correctly. pic is a colored image.
// Postcondition: pic will be modified depending on the amount of contrast passed in
void ModifyContrast(Mat& pic, double const contrast) {
	double ave_blue = 0;
	double ave_green = 0;
	double ave_red = 0;

	for (int row = 0; row < pic.rows; row++) {
		for (int col = 0; col < pic.cols; col++) {
			ave_blue += pic.at<Vec3b>(row, col)[0];
			ave_green += pic.at<Vec3b>(row, col)[1];
			ave_red += pic.at<Vec3b>(row, col)[2];
		}
	}

	ave_blue = ave_blue / pic.total();
	ave_green = ave_green / pic.total();
	ave_red = ave_red / pic.total();

	for (int row = 0; row < pic.rows; row++) {
		for (int col = 0; col < pic.cols; col++) {
			double new_blue = ave_blue - double(pic.at<Vec3b>(row, col)[0]);
			new_blue = ave_blue - (new_blue * contrast);

			double new_green = ave_green - double(pic.at<Vec3b>(row, col)[1]);
			new_green = ave_green - (new_green * contrast);

			double new_red = ave_red - double(pic.at<Vec3b>(row, col)[2]);
			new_red = ave_red - (new_red * contrast);

			pic.at<Vec3b>(row, col)[0] = FixComputedColor(new_blue);
			pic.at<Vec3b>(row, col)[1] = FixComputedColor(new_green);
			pic.at<Vec3b>(row, col)[2] = FixComputedColor(new_red);
		}
	}
}

// ModifySaturation
// Preconditions: image is colored in BGR and of the correct type and correctly allocated
// Postconditions: given image's saturation is changed by the global constant sat_val amount
void ModifySaturation(Mat& image, int const saturate) {
	Mat saturated;
	cvtColor(image, saturated, COLOR_BGR2HSV);
	for (int row = 0; row < saturated.rows; row++) {
		for (int col = 0; col < saturated.cols; col++) {
			double og = (double)saturated.at<Vec3b>(row, col)[1];
			int newVal = FixComputedColor(og + saturate);
			saturated.at<Vec3b>(row, col)[1] = newVal;
		}
	}
	cvtColor(saturated, image, COLOR_HSV2BGR);
}

// PrepareImage
// Precondition: Parameters and image is properly formatted, passed in correctly and colored
// Postcondition: Will modify image by putting various blurrs and filters on top. image will
//                be modified slightly differently depending if it is a background or not.
void PrepareImage(Mat& image) {
	medianBlur(image, image, median_blur);
	ModifyContrast(image, contrast_num);
	GaussianBlur(image, image, Size(gaus_blur_size, gaus_blur_size), gaus_blur_amount);
	image.convertTo(image, -1, 1, brightness_level);
	ModifySaturation(image, sat_val);
}

// BackgroundRemover
// Precondition: Parameters are properly formatted, passed in correctly and colored
// Postcondition: Will return a binary Matt where the white spots are the differences
//                between the 2 passed in Mats.
Mat BackgroundRemover(const Mat& front, const Mat& back) {
	Mat output(back.rows, back.cols, CV_8U);
	for (int row = 0; row < back.rows; row++) {
		for (int col = 0; col < back.cols; col++) {
			int front_color_b = front.at<Vec3b>(row, col)[0];
			int front_color_g = front.at<Vec3b>(row, col)[1];
			int front_color_r = front.at<Vec3b>(row, col)[2];
			int back_color_b = back.at<Vec3b>(row, col)[0];
			int back_color_g = back.at<Vec3b>(row, col)[1];
			int back_color_r = back.at<Vec3b>(row, col)[2];
			if (abs(front_color_b - back_color_b) < background_remover_thresh &&
				abs(front_color_g - back_color_g) < background_remover_thresh &&
				abs(front_color_r - back_color_r) < background_remover_thresh) { // Very similar
				output.at<uchar>(row, col) = 0;
			}
			else {	// Not similar. Object here
				if ((front_color_r >= red_color_thresh) ||
					(front_color_r < red_color_thresh &&
						front_color_r > front_color_b &&
						front_color_r > front_color_g)) {
					output.at<uchar>(row, col) = 255;
				}
				else output.at<uchar>(row, col) = 0;;
			}
		}
	}
	return output;
}

// ExtractBackground
// Preconditions: video is correctly formatted and allocated
// Postconditions: the calculated background from the video is returned as a Mat
Mat ExtractBackground(VideoCapture& video) {
	const int frame_width = (int)video.get(CAP_PROP_FRAME_WIDTH);
	const int frame_height = (int)video.get(CAP_PROP_FRAME_HEIGHT);
	const int number_of_frames = (int)video.get(CAP_PROP_FRAME_COUNT);
	vector<int> random_frames;

	// Determine which random frames to use for background calculation
	for (int i = 0; i < number_random_frames; i++) {
		int random_frame = rand() % number_of_frames;
		while (find(random_frames.begin(), random_frames.end(), random_frame) !=
			   random_frames.end()) {
			random_frame = rand() % number_of_frames;
		}
		random_frames.push_back(random_frame);
	}

	Mat extracted_background(frame_height, frame_width, CV_8UC3, Scalar::all(0));
	Mat frame;
	int curFrame = 0;
	bool firstRandomFrame = true;
	vector<vector<vector<int>>> backgroundPixels(frame_height);

	// Go through each random frame and add its values to each pixel in extracted background
	for (;;) {
		video >> frame;
		if (frame.empty()) {
			break;
		}
		if (find(random_frames.begin(), random_frames.end(), curFrame) != random_frames.end()) {
			for (int row = 0; row < frame_height; row++) {
				if (firstRandomFrame) {
					vector<vector<int>> temp(frame_width);
					backgroundPixels.at(row) = temp;
				}
				for (int col = 0; col < frame_width; col++) {
					if (firstRandomFrame) {
						vector<int> colorValues(3);
						backgroundPixels.at(row).at(col) = colorValues;
						backgroundPixels.at(row).at(col).at(2) = frame.at<Vec3b>(row, col)[2];
						backgroundPixels.at(row).at(col).at(1) = frame.at<Vec3b>(row, col)[1];
						backgroundPixels.at(row).at(col).at(0) = frame.at<Vec3b>(row, col)[0];
					}
					else {
						backgroundPixels.at(row).at(col).at(2) += frame.at<Vec3b>(row, col)[2];
						backgroundPixels.at(row).at(col).at(1) += frame.at<Vec3b>(row, col)[1];
						backgroundPixels.at(row).at(col).at(0) += frame.at<Vec3b>(row, col)[0];
					}
				}
			}
			firstRandomFrame = false;
		}
		curFrame++;
	}

	// Average every pixel in background to get final background from video
	for (int row = 0; row < frame_height; row++) {
		for (int col = 0; col < frame_width; col++) {
			extracted_background.at<Vec3b>(row, col)[2] =
				FixComputedColor(backgroundPixels.at(row).at(col).at(2) / number_random_frames);
			extracted_background.at<Vec3b>(row, col)[1] =
				FixComputedColor(backgroundPixels.at(row).at(col).at(1) / number_random_frames);
			extracted_background.at<Vec3b>(row, col)[0] =
				FixComputedColor(backgroundPixels.at(row).at(col).at(0) / number_random_frames);
		}
	}
	video.set(CAP_PROP_POS_MSEC, 0);
	return extracted_background;
}