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
	//problem_a_rotate_forward(input, angle);
	//Fill problem_b_rotate_backward and show output
	problem_b_rotate_backward(input, angle);
	//Fill problem_c_rotate_backward_interarea and show output
	//problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want
	//Example_change_brightness(input, 100, 50, 125);
}


/*
* 
* #include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// 회전 행렬을 생성하는 함수 (45도 CCW 회전)
Mat getRotationMatrix(double angle) {
    double radians = angle * CV_PI / 180.0;
    double cosA = cos(radians);
    double sinA = sin(radians);
    Mat rotationMatrix = (Mat_<double>(2, 2) << cosA, -sinA, sinA, cosA);
    return rotationMatrix;
}

// Forward Method로 이미지 회전 (좌표 변환 후 원본에서 값 복사)
Mat rotateImageForward(const Mat& inputImage, double angle) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // 회전 행렬
    Mat rotationMatrix = getRotationMatrix(angle);

    // 회전된 이미지 크기 설정 (단순히 원래 이미지 크기로 설정)
    Mat outputImage = Mat::zeros(rows, cols, inputImage.type());

    // 이미지의 중심을 기준으로 좌표 변환
    Point2f center(cols / 2.0, rows / 2.0);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            // 원본 좌표를 새로운 좌표로 변환
            Point2f originalPoint = Point2f(x, y) - center;
            Point2f rotatedPoint = Point2f(
                rotationMatrix.at<double>(0, 0) * originalPoint.x + rotationMatrix.at<double>(0, 1) * originalPoint.y,
                rotationMatrix.at<double>(1, 0) * originalPoint.x + rotationMatrix.at<double>(1, 1) * originalPoint.y
            );

            rotatedPoint += center;

            // 유효한 좌표 범위 체크
            if (rotatedPoint.x >= 0 && rotatedPoint.x < cols && rotatedPoint.y >= 0 && rotatedPoint.y < rows) {
                // 정수 좌표로 변환 후 값 복사
                outputImage.at<Vec3b>(y, x) = inputImage.at<Vec3b>((int)rotatedPoint.y, (int)rotatedPoint.x);
            }
        }
    }
    return outputImage;
}

// Backward Method로 이미지 회전 (보간법 적용)
Mat rotateImageBackward(const Mat& inputImage, double angle, bool useBilinear = false) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // 회전 행렬
    Mat rotationMatrix = getRotationMatrix(angle).inv(); // 역행렬 사용

    // 회전된 이미지 크기 설정 (단순히 원래 이미지 크기로 설정)
    Mat outputImage = Mat::zeros(rows, cols, inputImage.type());

    // 이미지의 중심을 기준으로 좌표 변환
    Point2f center(cols / 2.0, rows / 2.0);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            // 목적지 좌표를 원본 좌표로 변환
            Point2f rotatedPoint = Point2f(x, y) - center;
            Point2f originalPoint = Point2f(
                rotationMatrix.at<double>(0, 0) * rotatedPoint.x + rotationMatrix.at<double>(0, 1) * rotatedPoint.y,
                rotationMatrix.at<double>(1, 0) * rotatedPoint.x + rotationMatrix.at<double>(1, 1) * rotatedPoint.y
            );

            originalPoint += center;

            // 보간법 선택 (Nearest Neighbor 또는 Bilinear)
            if (originalPoint.x >= 0 && originalPoint.x < cols && originalPoint.y >= 0 && originalPoint.y < rows) {
                if (useBilinear) {
                    // Bilinear Interpolation (양선형 보간법)
                    int x1 = floor(originalPoint.x);
                    int y1 = floor(originalPoint.y);
                    int x2 = ceil(originalPoint.x);
                    int y2 = ceil(originalPoint.y);

                    Vec3b p1 = inputImage.at<Vec3b>(y1, x1);
                    Vec3b p2 = inputImage.at<Vec3b>(y1, x2);
                    Vec3b p3 = inputImage.at<Vec3b>(y2, x1);
                    Vec3b p4 = inputImage.at<Vec3b>(y2, x2);

                    double a = originalPoint.x - x1;
                    double b = originalPoint.y - y1;

                    Vec3b interpolatedPixel = p1 * (1 - a) * (1 - b) + p2 * a * (1 - b) + p3 * (1 - a) * b + p4 * a * b;
                    outputImage.at<Vec3b>(y, x) = interpolatedPixel;

                } else {
                    // Nearest Neighbor Interpolation (최근접 이웃법)
                    outputImage.at<Vec3b>(y, x) = inputImage.at<Vec3b>((int)round(originalPoint.y), (int)round(originalPoint.x));
                }
            }
        }
    }
    return outputImage;
}

int main() {
    // 이미지 읽기
    Mat inputImage = imread("input.jpg");

    if (inputImage.empty()) {
        cout << "이미지를 불러올 수 없습니다!" << endl;
        return -1;
    }

    // 45도 회전
    Mat rotatedForward = rotateImageForward(inputImage, 45.0);
    Mat rotatedBackwardNearest = rotateImageBackward(inputImage, 45.0, false);  // Nearest Neighbor
    Mat rotatedBackwardBilinear = rotateImageBackward(inputImage, 45.0, true);  // Bilinear Interpolation

    // 결과 출력
    imwrite("rotated_forward.jpg", rotatedForward);
    imwrite("rotated_backward_nearest.jpg", rotatedBackwardNearest);
    imwrite("rotated_backward_bilinear.jpg", rotatedBackwardBilinear);

    // 출력된 이미지를 창에 표시
    imshow("Original Image", inputImage);
    imshow("Rotated Forward", rotatedForward);
    imshow("Rotated Backward Nearest", rotatedBackwardNearest);
    imshow("Rotated Backward Bilinear", rotatedBackwardBilinear);

    waitKey(0);
    return 0;
}

*/

/*
설명:
getRotationMatrix(): 45도 회전 행렬을 계산합니다.
rotateImageForward(): Forward method를 사용하여 이미지를 회전합니다. 각 픽셀을 새로운 좌표로 이동시키고, 변환 후의 좌표에 값을 복사합니다.
rotateImageBackward(): Backward method를 사용합니다. 목적지 좌표에서 원본 이미지의 대응 좌표를 찾아 보간법을 통해 값을 가져옵니다. useBilinear 플래그에 따라 최근접 이웃법(Nearest Neighbor) 또는 양선형 보간법(Bilinear Interpolation)을 선택할 수 있습니다.
main(): 이미지를 읽어들이고, Forward 및 Backward 회전 방법을 각각 수행한 후 결과를 파일로 저장하고 화면에 출력합니다.
*/