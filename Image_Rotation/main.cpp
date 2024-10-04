#include <iostream>
#include <opencv2\opencv.hpp>

cv::Mat problem_a_rotate_forward(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	int width, height;
	width = img.cols; height = img.rows;

	cv::Point centerPoint = cv::Point(width / 2, height / 2);

	cv::Mat rotateMatrix = cv::getRotationMatrix2D(centerPoint, angle, 1.0);
	cv::warpAffine(img, output, rotateMatrix, img.size());

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////
	cv::imshow("a_output", output); cv::waitKey(0);
	return output;
}

cv::Mat problem_b_rotate_backward(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	int width, height;
	width = img.cols; height = img.rows;

	cv::Point centerPoint = cv::Point(width / 2, height / 2);

	cv::Mat rotateMatrix = cv::getRotationMatrix2D(centerPoint, -angle, 1.0);
	cv::warpAffine(img, output, rotateMatrix, img.size());

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("b_output", output); cv::waitKey(0);

	return output;
}

cv::Mat problem_c_rotate_backward_interarea(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("c_output", output); cv::waitKey(0);

	return output;
}

cv::Mat Example_change_brightness(cv::Mat img, int num, int x, int y) {
	/*
	img : input image
	num : number for brightness (increase or decrease)
	x : x coordinate of image (for square part)
	y : y coordinate of image (for square part)

	*/
	cv::Mat output = img.clone();
	int size = 100;
	int height = (y + 100 > img.cols) ? img.cols : y + 100;
	int width = (x + 100 > img.rows) ? img.rows : x + 100;

	for (int i = x; i < width; i++)
	{
		for (int j = y; j < height; j++)
		{
			for (int c = 0; c < img.channels(); c++)
			{
				int t = img.at<cv::Vec3b>(i, j)[c] + num;
				output.at<cv::Vec3b>(i, j)[c] = t > 255 ? 255 : t < 0 ? 0 : t;
			}
		}

	}
	cv::imshow("output1", img);
	cv::imshow("output2", output);
	cv::waitKey(0);
	return output;
}

int main(void){

	double angle = -15.0f;

	cv::Mat input = cv::imread("lena.jpg");
	cv::imshow("a_output", input);
	//Fill problem_a_rotate_forward and show output
	problem_a_rotate_forward(input, angle);
	//Fill problem_b_rotate_backward and show output
	problem_b_rotate_backward(input, angle);
	//Fill problem_c_rotate_backward_interarea and show output
	//problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want
	Example_change_brightness(input, 100, 50, 125);
}