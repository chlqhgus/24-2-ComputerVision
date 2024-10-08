#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <opencv2\opencv.hpp>

cv::Mat problem_a_rotate_forward(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

    double rad = angle * M_PI /180; // 각도를 radian으로 변환하기 위한 공식
    double cos_rad = cos(rad);
    double sin_rad = sin(rad);

    // cordinate rotation 시 center point를 계산하기 위해 변환하기 전 image size를 계산해야 한다.
    // check image size to calculate center point
    int height = img.rows; 
    int width = img.cols;

    int height_center = height / 2;
    int width_center = width / 2;

    //rotation 이후 image의 size가 변화할 수 있기 때문에 size를 계산해주어야 한다.
    //Since image size can be changed after rotation, we need to calculate rotationed image size
    int output_width = static_cast<int>(abs(width * cos_rad) + abs(height * sin_rad)); 
    int output_height = static_cast<int>(abs(width * sin_rad) + abs(height * cos_rad)); //pixel location have to be (int,int), so use abs

    int output_width_center = output_width / 2;
    int output_height_center = output_height / 2;

    //declare output Image size
    output = cv::Mat::zeros(cv::Size(output_width, output_height), img.type());

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            
            //중심점을 기준으로 rotate 되므로, 현재 위치에서 중심점의 값을 빼주어야 한다.
            int x_temp = x - width_center; 
            int y_temp = y - height_center;

            double rotate_x = cos_rad * x_temp - sin_rad * y_temp + output_width_center;
            double rotate_y = sin_rad*x_temp + cos_rad*y_temp+ output_height_center;

            int output_x = round(rotate_x);
            int output_y = round(rotate_y);

            if (output_x >= 0 && output_x < output_width && output_y >= 0 && output_y < output_height) {
                output.at<cv::Vec3b>(output_y, output_x) = img.at<cv::Vec3b>(y, x);
            }

        }
    }

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

    double rad = angle * M_PI / 180; 
    double cos_rad_backward = cos(-rad);
    double sin_rad_backward = sin(-rad);

    int height = img.rows;
    int width = img.cols;

    int height_center = height / 2;
    int width_center = width / 2;

    int output_width = static_cast<int>(abs(width * cos_rad_backward) + abs(height * sin_rad_backward));
    int output_height = static_cast<int>(abs(width * sin_rad_backward) + abs(height * cos_rad_backward));

    int output_width_center = output_width / 2;
    int output_height_center = output_height / 2;

    //declare output Image size
    output = cv::Mat::zeros(cv::Size(output_width, output_height), img.type());

    for (int x = 0; x < output_width; x++) {
        for (int y = 0; y < output_height; y++) {
            
            //backward는 forward와 반대로, 변환된 이미지로부터 original image의 픽셀을 가져와야 함.
            int output_x_temp = x - output_width_center;
            int output_y_temp = y - output_height_center;

            double before_x = cos_rad_backward * output_x_temp - sin_rad_backward * output_y_temp + width_center;
            double before_y = sin_rad_backward * output_x_temp + cos_rad_backward * output_y_temp + height_center;

            int img_x = round(before_x);
            int img_y = round(before_y);

            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                output.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(img_y,img_x);
            }
        }
    }

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

    double rad = angle * M_PI / 180;
    double cos_rad_backward = cos(-rad);
    double sin_rad_backward = sin(-rad);

    int height = img.rows;
    int width = img.cols;

    int height_center = height / 2;
    int width_center = width / 2;

    int output_width = static_cast<int>(abs(width * cos_rad_backward) + abs(height * sin_rad_backward));
    int output_height = static_cast<int>(abs(width * sin_rad_backward) + abs(height * cos_rad_backward));

    int output_width_center = output_width / 2;
    int output_height_center = output_height / 2;

    //declare output Image size
    output = cv::Mat::zeros(cv::Size(output_width, output_height), img.type());

    for (int x = 0; x < output_width; x++) {
        for (int y = 0; y < output_height; y++) {

            //backward는 forward와 반대로, 변환된 이미지로부터 original image의 픽셀을 가져와야 함.
            int output_x_temp = x - output_width_center;
            int output_y_temp = y - output_height_center;

            double before_x = cos_rad_backward * output_x_temp - sin_rad_backward * output_y_temp + width_center;
            double before_y = sin_rad_backward * output_x_temp + cos_rad_backward * output_y_temp + height_center;

            int x1 = floor(before_x);
            int x2 = ceil(before_x);
            int y1 = floor(before_y);
            int y2 = ceil(before_y);

            if (x1 < 0 || x2 >= img.cols || y1 < 0 || y2 >= img.rows) continue;
            cv::Vec3b pixel_x1y1 = img.at<cv::Vec3b>(y1,x1);
            cv::Vec3b pixel_x1y2 = img.at<cv::Vec3b>(y2,x1);
            cv::Vec3b pixel_x2y1 = img.at<cv::Vec3b>(y1,x2);
            cv::Vec3b pixel_x2y2 = img.at<cv::Vec3b>(y2, x2);           
            double d1 = before_x - x1;
            double d2 = x2 - before_x; //d2 = 1-d1
            double d3 = y2 - before_y; //d3 = 1-d4
            double d4 = before_y - y1;
           
            /*
            
            Bilinear Interpolation의 수식적 의미를 사용해서 아래 코드로 구현해보았으나 특정 지점에서 hole이 발생함.
            이는 아마 d1,d2,d3,d4를 각각 구하는 방법에서 오차가 발생했기 때문이라고 판단하였음.
            정수에서 실수를 빼는 것이기 때문에 오차가 발생했던 것.

            이를 해결하기 위해서는 d2,d3를 모두 구하는 것이 아닌, (1-d1)과 같이 표현해야 오차가 최소화될 것임

            */
            //cv::Vec3b pixel_final = (d1 * d3 * pixel_x2y1 + d2 * d3 * pixel_x1y1 + d1 * d4 * pixel_x2y2 + d2 * d4 * pixel_x1y2)/((d1+d2)*(d3+d4));

            cv::Vec3b pixel_final = (d1 * (1-d4) * pixel_x2y1 + (1-d1) * (1-d4) * pixel_x1y1 + d1 * d4 * pixel_x2y2 + (1-d1) * d4 * pixel_x1y2); //d2=1-d1, d3=1-d4 대입
            output.at<cv::Vec3b>(y, x) = pixel_final;
 
        }
    }


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
	problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want
	Example_change_brightness(input, 100, 50, 125);
}
