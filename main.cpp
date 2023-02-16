#include <iostream>
#include <chrono>
#include <string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;
using namespace std;


void onMouse(int event, int x, int y, int flags, void *param) {
    Mat *im = reinterpret_cast<Mat *>(param);
    switch (event) {
        case 1:     //鼠标左键按下响应：返回坐标和灰度
            std::cout << "at(" << x << "," << y << ")value is:"
                      << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
            break;
        case 2:    //鼠标右键按下响应：输入坐标并返回该坐标的灰度
            std::cout << "input(x,y)" << endl;
            std::cout << "x =" << endl;
            cin >> x;
            std::cout << "y =" << endl;
            cin >> y;
            std::cout << "at(" << x << "," << y << ")value is:"
                      << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
            break;
    }
}

/*
 * [599.1767578125, 0, 1051.742061275407;
 0, 989.3424682617188, 571.4513472305662;
 0, 0, 1]
 * */

// 这个函数是鱼眼相机要走的
void ProjectPointFromOriginToUndistorted(cv::Matx33d K, cv::Vec4d D, cv::Point2f input, cv::Point2f &output) {
    float x, y, theta, theta_d, r, x_origin, y_origin;
    float theta_cal[2], d_theta;

    x_origin = input.x - K(0, 2);
    y_origin = input.y - K(1, 2);
    theta_d = sqrt(x_origin * x_origin + y_origin * y_origin)
              / K(0, 0);
    theta_cal[0] = 0;
    theta_cal[1] = CV_PI / 2;
    d_theta = fabs(theta_cal[1] - theta_cal[0]);
    while (d_theta > 0.01) {
        float val[3];
        float middle_theta_cal = 0.5 * (theta_cal[1] + theta_cal[0]);

        val[0] = theta_cal[0] + D(0) * pow(theta_cal[0], 3)
                 + D(1) * pow(theta_cal[0], 5) + D(2) * pow(theta_cal[0], 7)
                 + D(3) * pow(theta_cal[0], 9) - theta_d;

        val[1] = theta_cal[1] + D(0) * pow(theta_cal[1], 3)
                 + D(1) * pow(theta_cal[1], 5) + D(2) * pow(theta_cal[1], 7)
                 + D(3) * pow(theta_cal[1], 9) - theta_d;

        val[2] = middle_theta_cal + D(0) * pow(middle_theta_cal, 3)
                 + D(1) * pow(middle_theta_cal, 5) + D(2) * pow(middle_theta_cal, 7)
                 + D(3) * pow(middle_theta_cal, 9) - theta_d;

        if (fabs(val[2]) < 1e-6) {
            break;
        }
        if (val[0] * val[2] > 0) {
            theta_cal[0] = middle_theta_cal;
        } else {
            theta_cal[1] = middle_theta_cal;
        }
        d_theta = theta_cal[1] - theta_cal[0];
    }

    theta = 0.5 * (theta_cal[1] + theta_cal[0]);
    x = tan(theta) / sqrt(x_origin * x_origin + y_origin * y_origin) * x_origin;
    y = tan(theta) / sqrt(x_origin * x_origin + y_origin * y_origin) * y_origin;
    output.x = x * K(0, 0) + K(0, 2);
    output.y = y * K(1, 1) + K(1, 2);
}

void CalcCameraExtrinsics() {
    std::vector<cv::Point2f> pixel_uv_vec;
    std::vector<cv::Point3f> point_world_vec;
    cv::Mat rvec, tvec, R, T;

    cv::Mat K(3, 3, CV_64FC1);
    K.at<double>(0, 0) = 290.62;
    K.at<double>(0, 1) = 0;
    K.at<double>(0, 2) = 643.44;
    K.at<double>(1, 0) = 0;
    K.at<double>(1, 1) = 290.62;
    K.at<double>(1, 2) = 365.61;
    K.at<double>(2, 0) = 0;
    K.at<double>(2, 1) = 0;
    K.at<double>(2, 2) = 1;

    cv::Mat D(4, 1, CV_64FC1);
    D.at<double>(0, 0) = 0.165618;
    D.at<double>(1, 0) = 0.020838;
    D.at<double>(2, 0) = -0.023782;
    D.at<double>(3, 0) = 0.002512;

    point_world_vec.push_back(cv::Point3f(-3100, 0, 0));
    point_world_vec.push_back(cv::Point3f(-2250, 850, 0));
    point_world_vec.push_back(cv::Point3f(-1400, 0, 0));
    point_world_vec.push_back(cv::Point3f(-2250, -850, 0));
    point_world_vec.push_back(cv::Point3f(1400, 0, 0));
    point_world_vec.push_back(cv::Point3f(2250, 850, 0));
    point_world_vec.push_back(cv::Point3f(3100, 0, 0));
    point_world_vec.push_back(cv::Point3f(2250, -850, 0));

    pixel_uv_vec.push_back(cv::Point2f(298, 373));
    pixel_uv_vec.push_back(cv::Point2f(419, 338));
    pixel_uv_vec.push_back(cv::Point2f(441, 380));
    pixel_uv_vec.push_back(cv::Point2f(266, 444));
    pixel_uv_vec.push_back(cv::Point2f(835, 383));
    pixel_uv_vec.push_back(cv::Point2f(862, 341));
    pixel_uv_vec.push_back(cv::Point2f(987, 380));
    pixel_uv_vec.push_back(cv::Point2f(1013, 453));

    // 计算得到原始图像中像素点坐标在畸变校正后的图像中的坐标
    for (int i = 0; i < pixel_uv_vec.size(); i++) {
        cv::Matx33d K;
        cv::Vec4d D;
        K(0, 0) = 290.62;
        K(0, 1) = 0;
        K(0, 2) = 643.44;
        K(1, 0) = 0;
        K(1, 1) = 290.62;
        K(1, 2) = 365.61;
        K(2, 0) = 0;
        K(2, 1) = 0;
        K(2, 2) = 1;
        D(0) = 0.165618;
        D(1) = 0.020838;
        D(2) = -0.023782;
        D(3) = 0.002512;
        ProjectPointFromOriginToUndistorted(K, D, pixel_uv_vec[i], pixel_uv_vec[i]);
    }

    // 计算得到旋转向量和平移向量，将旋转向量转换为旋转矩阵
    cv::solvePnP(point_world_vec, pixel_uv_vec, K, cv::Mat(), rvec, tvec);
    cv::Rodrigues(rvec, R);

    T.create(4, 4, CV_64FC1);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec.at<double>(0, 0);
    T.at<double>(1, 3) = tvec.at<double>(1, 0);
    T.at<double>(2, 3) = tvec.at<double>(2, 0);
    T.at<double>(3, 0) = T.at<double>(3, 1) = T.at<double>(3, 2) = 0;
    T.at<double>(3, 3) = 1;

    // 验证计算得到的外参的准确性，通过利用计算得到的外参将世界坐标投影至像素坐标系
    // 并将结果与输入的像素坐标对比来验证
    for (int i = 0; i < point_world_vec.size(); i++) {
        // 世界坐标系变换至相机坐标系
        float ux, uy;
        cv::Mat pt_cam, pt_world;
        pt_world.create(4, 1, CV_64FC1);
        pt_world.at<double>(0, 0) = point_world_vec[i].x;
        pt_world.at<double>(1, 0) = point_world_vec[i].y;
        pt_world.at<double>(2, 0) = point_world_vec[i].z;
        pt_world.at<double>(3, 0) = 1;

        pt_cam = T * pt_world;

        // 相机坐标系变换至像素坐标系
        float x = pt_cam.at<double>(0, 0);
        float y = pt_cam.at<double>(1, 0);
        float z = pt_cam.at<double>(2, 0);

        if (z < 1e-6)
            z = 1.f;

        float r = sqrtf(x * x + y * y);

        if (r < 1e-6) {
            // 位于光心
            ux = K.at<double>(0, 2);
            uy = K.at<double>(1, 2);
        } else {
            float theta = atan2f(r, z);

            float theta_d = 1.f;
            float res = 1.f;
            for (int i = 0; i < 4; ++i) {
                theta_d *= theta * theta;
                res += D.at<double>(i, 0) * theta_d;
            }

            res *= theta;

            ux = x * K.at<double>(0, 0) * res / r + K.at<double>(0, 2);
            uy = y * K.at<double>(1, 1) * res / r + K.at<double>(1, 2);
        }
        printf("ux, uy: %f %f\n", ux, uy);
    }

    // 将图像透视变换至地面
    int img_w = 3000;
    int img_h = 1000;
    float world_w = 7000;// 单位为mm
    float scale = img_w / world_w;
    cv::Mat dst_img(img_h, img_w, CV_8UC3);
    cv::Mat src_img = cv::imread("back.bmp");
    for (int v = 0; v < img_h; v++) {
        for (int u = 0; u < img_w; u++) {
            // 目标图像到世界坐标系的映射
            float world_x, world_y, world_z;
            float ux, uy;
            world_x = (u - 0.5 * img_w) / scale;
            world_y = -(v - 0.5 * img_h) / scale;
            world_z = 0;

            cv::Mat pt_cam, pt_world;
            pt_world.create(4, 1, CV_64FC1);
            pt_world.at<double>(0, 0) = world_x;
            pt_world.at<double>(1, 0) = world_y;
            pt_world.at<double>(2, 0) = world_z;
            pt_world.at<double>(3, 0) = 1;

            pt_cam = T * pt_world;

            float x = pt_cam.at<double>(0, 0);
            float y = pt_cam.at<double>(1, 0);
            float z = pt_cam.at<double>(2, 0);

            if (z < 1e-6)
                z = 1.f;

            float r = sqrtf(x * x + y * y);

            if (r < 1e-6) {
                ux = K.at<double>(0, 2);
                uy = K.at<double>(1, 2);
            } else {
                float theta = atan2f(r, z);

                float theta_d = 1.f;
                float res = 1.f;
                for (int i = 0; i < 4; ++i) {
                    theta_d *= theta * theta;
                    res += D.at<double>(i, 0) * theta_d;
                }

                res *= theta;

                ux = x * K.at<double>(0, 0) * res / r + K.at<double>(0, 2);
                uy = y * K.at<double>(1, 1) * res / r + K.at<double>(1, 2);
            }
            if (ux < 0 || ux > src_img.cols - 1
                || uy < 0 || uy > src_img.rows - 1) {
                dst_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
            } else// 只是为了验证畸变校正流程，为方便这里用了最近邻差值
            {
                dst_img.at<cv::Vec3b>(v, u) = src_img.at<cv::Vec3b>((int) uy, (int) ux);
            }
        }
    }
    cv::imwrite("dst.bmp", dst_img);

}

void OpenCVFisheyeImageUndistortion() {
    cv::Mat img, undistortImg;
    cv::Matx33d K, P;
    cv::Vec4d D;
    cv::Mat mapX, mapY;
    img = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/jibian/1053.jpg");
    // const cv::Mat K2 = (cv::Mat_<double>(3, 3)
    //            << 1010.1193051331736 / 1.4, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831 /
    //                                                                       1.0, 577.4709247205907, 0.0, 0.0, 1.0);
    // 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0
    K(0, 0) = 1010.1193051331736;
    K(0, 1) = 0;
    K(0, 2) = 1007.7481675024154;
    K(1, 0) = 0;
    K(1, 1) = 1009.4943272753831;
    K(1, 2) = 577.4709247205907;
    K(2, 0) = 0;
    K(2, 1) = 0;
    K(2, 2) = 1;
    // -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457
    D(0) = -0.06663067168381479;
    D(1) = 0.0009026610617662017;
    D(2) = -0.007498635027107796;
    D(3) = 0.0019139336144852457;

    P = K;
    P(0, 0) /= 1.5;
    P(1, 1) /= 1.0;

    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(),
                                         P, cv::Size(img.cols, img.rows), CV_16SC2, mapX, mapY);

    cv::remap(img, undistortImg, mapX, mapY, CV_INTER_LINEAR);
    cv::imshow("src", img);
    cv::imshow("corrected", undistortImg);
    cv::imwrite("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053-2.jpg", undistortImg);
    cv::waitKey();
}

// 验证了3D->2D是没什么问题的，用的是去畸变的像素点，solvepnp用的畸变系数为0，2D->3D也验证了
void test_solvepnp() {
    // 看博客好像是坐标系没什么关系，对这个3D点，比如我就选择车头正中间的地面上的点作为世界坐标系的原点
    float threeDim[4][3] = {{3000, 1580,  0},
                            {3000, -1720, 0},
                            {6000, -1720, 0},
                            {6000, 1580,  0}};  // 这个点是距离光心的位置？
    // float twoDim[4][2] = { {628,797}, {1395,802}, {1258,501}, {763,497}};  // 原图
    float twoDim[4][2] = {{807,  805},  // 去畸变的图
                          {1303, 810},
                          {1205, 493},
                          {904,  490}};
    float lane[18][2] = {
//            {731,  1075},
//            {752,  1022},
//            {763,  970},
//            {783,  917},
//            {794,  865},
//            {810,  812},
//            {827,  760},
//            {836,  707},
//            {852,  655},
//            {869,  602},
//            {883,  550},
//            {899,  497},
//            {918,  445},
//            {934,  392},
//            {953,  340},
//            {971,  287},
//            {989,  235},
//            {1007, 182},
            {299, 1075},
            {333, 1022},
            {368, 970},
            {411, 917},
            {446, 865},
            {484, 812},
            {523, 760},
            {564, 707},
            {601, 655},
            {638, 602},
            {672, 550},
            {709, 497},
            {752, 445},
            {786, 392},
            {821, 340},
            {860, 287},
            {896, 235},
            {930, 182},
    };
    float lane2[18][2] = {
//            {1353,  1075},
//            {1348,  1022},
//            {1332,  970},
//            {1318,  917},
//            {1312,  865},
//            {1298,  812},
//            {1281,  760},
//            {1265,  707},
//            {1247,  655},
//            {1232,  602},
//            {1213,  550},
//            {1202,  497},
//            {1181,  445},
//            {1163,  392},
//            {1142,  340},
//            {1128,  287},
//            {1108,  235},
//            {1095, 182},
            {1796, 1075},
            {1761, 1022},
            {1724, 970},
            {1689, 917},
            {1650, 865},
            {1612, 812},
            {1576, 760},
            {1540, 707},
            {1504, 655},
            {1467, 602},
            {1432, 550},
            {1394, 497},
            {1367, 445},
            {1320, 392},
            {1280, 340},
            {1240, 287},
            {1203, 235},
            {1166, 182},
    };


    vector<Point2f> lane_;
    vector<Point2f> lane2_;
    for (int i = 0; i < 18; i++) {
        lane_.push_back(Point2f(lane[i][0], lane[i][1]));
        lane2_.push_back(Point2f(lane2[i][0], lane2[i][1]));
    }
    vector<Point3f> outDim;
    vector<Point2f> inDim;
    vector<float> distCoeff(0);

    for (int i = 0; i < 4; i++) {
        outDim.push_back(Point3f(threeDim[i][0], threeDim[i][1], threeDim[i][2]));
        inDim.push_back(Point2f(twoDim[i][0], twoDim[i][1]));
    }

    const cv::Mat D = (cv::Mat_<double>(4, 1)
            << -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457);
    const int ImgWidth = 1920;
    const int ImgHeight = 1080;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 0;
    const cv::Mat K2 = (cv::Mat_<double>(3, 3)
            << 1010.1193051331736 / 1.4, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831 /
                                                                       1.0, 577.4709247205907, 0.0, 0.0, 1.0);
    cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K2, D, imageSize, alpha, imageSize, 0);

    Mat sourceImage = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
    namedWindow("Source", 1);
    for (int i = 0; i < inDim.size(); ++i) {
        circle(sourceImage, inDim[i], 3, Scalar(0, 255, 0), -1, 8);
    }
    // imshow("Source",sourceImage);


//    Mat cameraMatrix(3,3,CV_32F);
//    float tempMatrix[3][3] = { { 2697.6,0 ,597.4 }, { 0, 2682,515.6 }, { 0, 0 ,1} };
//    for (int i = 0; i < 3;i++)
//    {
//        for (int j = 0; j < 3;j++)
//        {
//            cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
//        }
//    }

    Mat rvec1, tvec1;
    solvePnP(outDim, inDim, NewCameraMatrix, Mat(), rvec1, tvec1);
    cout << rvec1 << endl;
    cout << tvec1 << endl;
    cout << "11111111----------------------------------11111111" << endl;
    cv::Mat rvecM1(3, 3, cv::DataType<double>::type);  //旋转矩阵
    Rodrigues(rvec1, rvecM1);
    cout << rvecM1 << endl;
    cout << tvec1 << endl;
    cout<<NewCameraMatrix<<endl;

    // 此处用于求相机位于坐标系内的旋转角度,2D-3D的转换并不用求,
    // 这边几个角度没什么用
    const double PI = 3.1415926;
    double thetaZ = atan2(rvecM1.at<double>(1, 0), rvecM1.at<double>(0, 0)) / PI * 180;
    double thetaY = atan2(-1 * rvecM1.at<double>(2, 0), sqrt(rvecM1.at<double>(2, 1) * rvecM1.at<double>(2, 1)
                                                             + rvecM1.at<double>(2, 2) * rvecM1.at<double>(2, 2))) /
                    PI * 180;
    double thetaX = atan2(rvecM1.at<double>(2, 1), rvecM1.at<double>(2, 2)) / PI * 180;
    cout << "theta x  " << thetaX << endl << "theta Y: " << thetaY << endl << "theta Z: " << thetaZ << endl;


    ///根据公式求Zc，即s
    cv::Mat imagePoint = cv::Mat::ones(3, 1, cv::DataType<double>::type);
    cv::Mat tempMat, tempMat2;
    //输入一个2D坐标点，便可以求出相应的s
    imagePoint.at<double>(0, 0) = 904;
    imagePoint.at<double>(1, 0) = 490;
    double zConst = 0;//实际坐标系的距离
    //计算参数s
    double s;
    tempMat = rvecM1.inv() * NewCameraMatrix.inv() * imagePoint;  // M1矩阵
    tempMat2 = rvecM1.inv() * tvec1;  // M2矩阵
    s = zConst + tempMat2.at<double>(2, 0);
    s /= tempMat.at<double>(2, 0);
    cout << "s : " << s << endl;

    // **********************************
    ///3D to 2D
    cv::Mat worldPoints = Mat::ones(4, 1, cv::DataType<double>::type);
    worldPoints.at<double>(0, 0) = 6000;
    worldPoints.at<double>(1, 0) = 1580;
    worldPoints.at<double>(2, 0) = 0;
    cout << "world Points :  " << worldPoints << endl;
    Mat image_points = Mat::ones(3, 1, cv::DataType<double>::type);
    //setIdentity(image_points);

    // 下面这个流程能不能替换到透视变换那边
    Mat RT_;
    hconcat(rvecM1, tvec1, RT_);
    cout << "RT_" << RT_ << endl;
    image_points = NewCameraMatrix * RT_ * worldPoints;
    Mat D_Points = Mat::ones(3, 1, cv::DataType<double>::type);
    D_Points.at<double>(0, 0) = image_points.at<double>(0, 0) / image_points.at<double>(2, 0);
    D_Points.at<double>(1, 0) = image_points.at<double>(1, 0) / image_points.at<double>(2, 0);
    //cv::projectPoints(worldPoints, rvec1, tvec1, cameraMatrix1, distCoeffs1, imagePoints);
    cout << "3D to 2D:   " << D_Points << endl;

    //camera_coordinates
    Mat camera_cordinates = -rvecM1.inv() * tvec1;

    for (int i = 0; i < 18; ++i) {
        imagePoint.at<double>(0, 0) = lane_[i].x;;
        imagePoint.at<double>(1, 0) = lane_[i].y;
        double zConst = 0;//实际坐标系的距离
        //计算参数s
        double s;
        tempMat = rvecM1.inv() * NewCameraMatrix.inv() * imagePoint;
        tempMat2 = rvecM1.inv() * tvec1;
        s = zConst + tempMat2.at<double>(2, 0);
        s /= tempMat.at<double>(2, 0);
        // cout << "s : " << s << endl;
        cv::Mat imagePoint_your_know = cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
        imagePoint_your_know.at<double>(0, 0) = lane_[i].x;
        imagePoint_your_know.at<double>(1, 0) = lane_[i].y;
        Mat wcPoint = rvecM1.inv() * (NewCameraMatrix.inv() * s * imagePoint_your_know - tvec1);
        Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
        // cout << "2D to 3D :" << worldPoint << endl;

        imagePoint.at<double>(0, 0) = lane2_[i].x;;
        imagePoint.at<double>(1, 0) = lane2_[i].y;
        double zConst2 = 0;//实际坐标系的距离
        //计算参数s
        double s2;
        tempMat = rvecM1.inv() * NewCameraMatrix.inv() * imagePoint;
        tempMat2 = rvecM1.inv() * tvec1;
        s2 = zConst2 + tempMat2.at<double>(2, 0);
        s2 /= tempMat.at<double>(2, 0);
        // cout << "s : " << s << endl;
        cv::Mat imagePoint_your_know2 = cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
        imagePoint_your_know2.at<double>(0, 0) = lane2_[i].x;
        imagePoint_your_know2.at<double>(1, 0) = lane2_[i].y;
        Mat wcPoint2 = rvecM1.inv() * (NewCameraMatrix.inv() * s * imagePoint_your_know2 - tvec1);
        Point3f worldPoint2(wcPoint2.at<double>(0, 0), wcPoint2.at<double>(1, 0), wcPoint2.at<double>(2, 0));
        // cout << "2D to 3D :" << worldPoint2 << endl;
        cout << "2D to 3D :" << worldPoint.y - worldPoint2.y << endl;
    }
//    int img_w = 1920;
//    int img_h = 1080;
//    cv::Mat dst_img(img_h,img_w,CV_8UC3);
//    cv::Mat src_img = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
//    cv::warpPerspective(src_img,dst_img,rvecM1*tvec1,cv::Size(img_w,img_h));
//
//
//
//    imshow("Source", sourceImage);
//    imshow("Source2", dst_img);
//    waitKey(0);

}

void test_solvepnp2() {
    cv::Mat rvec, tvec, R, T;
    // 点是按照1   2
    //       4   3
    float threeDim[4][3] = {{3000, 1580,  0}, // x y z的顺序
                            {3000, -1720, 0},
                            {6000, -1720, 0},
                            {6000, 1580,  0}};  // 这个点是距离光心的位置？
//    float twoDim[4][2] = {{628,  797},
//                          {1395, 802},
//                          {1258, 501},
//                          {763,  497}};
    // 原图经过变换算出来的点： 原始内参下的点    这个是鱼眼模型算出来的点
    // [597.415, 814.534]
    // [1430.94, 822.685]
    // [1265.53, 498.746]
    // [757.977, 495.399]
    float twoDim[4][2] = {{807,  805},
                          {1303, 810},
                          {1205, 493},
                          {904,  490}};  // 去畸变的点

    vector<Point3f> outDim;
    vector<Point2f> inDim;
    vector<float> distCoeff(0);

    for (int i = 0; i < 4; i++) {
        outDim.push_back(Point3f(threeDim[i][0], threeDim[i][1], threeDim[i][2]));
        inDim.push_back(Point2f(twoDim[i][0], twoDim[i][1]));
    }
    // const cv::Mat K = (cv::Mat_<double>(3, 3)
    //         << 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0);
    const cv::Mat D = (cv::Mat_<double>(4, 1)
            << -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457);
    const cv::Mat D2 = (cv::Mat_<double>(5, 1)
            << 0, 0, 0, 0, 0);
    const int ImgWidth = 1920;
    const int ImgHeight = 1080;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 0;
    const cv::Mat K2 = (cv::Mat_<double>(3, 3)
            << 1010.1193051331736 / 1.4, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831 /
                                                                       1.0, 577.4709247205907, 0.0, 0.0, 1.0);
    cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K2, D, imageSize, alpha, imageSize, 0);
    cout << NewCameraMatrix << endl;
//
//    Mat sourceImage = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
//    namedWindow("Source", 1);
//    for (int i = 0; i < inDim.size(); ++i) {
//        circle(sourceImage, inDim[i], 3, Scalar(0, 255, 0), -1, 8);
//    }
    // imshow("Source",sourceImage);
    // 计算得到原始图像中像素点坐标在畸变校正后的图像中的坐标

//    for (int i = 0; i < inDim.size(); i++) {
//        cv::Matx33d K;
//        cv::Vec4d D;
//        /*[599.1767578125, 0, 1051.742061275407;
//         0, 989.3424682617188, 571.4513472305662;
//         0, 0, 1]
//         * */
//        K(0, 0) = 599.1767578125;
//        K(0, 1) = 0;
//        K(0, 2) = 1051.742061275407;
//        K(1, 0) = 0;
//        K(1, 1) = 989.3424682617188;
//        K(1, 2) = 571.4513472305662;
//        K(2, 0) = 0;
//        K(2, 1) = 0;
//        K(2, 2) = 1;
//        // 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0
////        K(0, 0) = 1010.1193051331736;
////        K(0, 1) = 0;
////        K(0, 2) = 1007.7481675024154;
////        K(1, 0) = 0;
////        K(1, 1) = 1009.4943272753831;
////        K(1, 2) = 577.4709247205907;
////        K(2, 0) = 0;
////        K(2, 1) = 0;
////        K(2, 2) = 1;
//
//        // -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457
//        D(0) = 0;
//        D(1) = 0;
//        D(2) = 0;
//        D(3) = 0;
//        ProjectPointFromOriginToUndistorted(K, D, inDim[i], inDim[i]);
//    }
    for (int i = 0; i < inDim.size(); ++i) {
        cout << inDim[i] << endl;
    }

    // 计算得到旋转向量和平移向量，将旋转向量转换为旋转矩阵
    cv::solvePnP(outDim, inDim, NewCameraMatrix, D2, rvec, tvec);
    cv::Rodrigues(rvec, R);

    T.create(4, 4, CV_64FC1);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec.at<double>(0, 0);
    T.at<double>(1, 3) = tvec.at<double>(1, 0);
    T.at<double>(2, 3) = tvec.at<double>(2, 0);
    T.at<double>(3, 0) = T.at<double>(3, 1) = T.at<double>(3, 2) = 0;
    T.at<double>(3, 3) = 1;
//    cout<<R<<endl;
//    cout<<T<<endl;
    // 第一个方法计算出来的RT矩阵,和这个方法算出来的一样的
    // RT_[0.002919347595132238, -0.9999813796559083, 0.005358894576756645, -57.04754470101124;
    // -0.4945244400306179, -0.00610144434352311,inDim -0.8691423074441414, 2433.772627727664;
    // 0.8691588207123151, -0.0001127758346887164, -0.4945330440522541, 1376.410228531058]

    // 验证计算得到的外参的准确性，通过利用计算得到的外参将世界坐标投影至像素坐标系
    // 并将结果与输入的像素坐标对比来验证
    for (int i = 0; i < inDim.size(); i++) {
        // 世界坐标系变换至相机坐标系
        float ux, uy;
        cv::Mat pt_cam, pt_world;
        pt_world.create(4, 1, CV_64FC1);
        pt_world.at<double>(0, 0) = outDim[i].x;
        pt_world.at<double>(1, 0) = outDim[i].y;
        pt_world.at<double>(2, 0) = outDim[i].z;
        pt_world.at<double>(3, 0) = 1;

        pt_cam = T * pt_world;

        // 相机坐标系变换至像素坐标系
        float x = pt_cam.at<double>(0, 0);
        float y = pt_cam.at<double>(1, 0);
        float z = pt_cam.at<double>(2, 0);

        if (z < 1e-6)
            z = 1.f;

        float r = sqrtf(x * x + y * y);

        if (r < 1e-6) {
            // 位于光心
            ux = NewCameraMatrix.at<double>(0, 2);
            uy = NewCameraMatrix.at<double>(1, 2);
        } else {
            ux = x * NewCameraMatrix.at<double>(0, 0) / z + NewCameraMatrix.at<double>(0, 2);
            uy = y * NewCameraMatrix.at<double>(1, 1) / z + NewCameraMatrix.at<double>(1, 2);
        }
        printf("ux, uy: %f %f  -> x,y: %f %f\n", ux, uy, inDim[i].x, inDim[i].y);
    }

    // 将图像透视变换至地面
    int img_w = 1920 / 3;
    int img_h = 1080 / 3;
    float world_w = 70000;// 单位为mm
    float scale = img_w / world_w;
    cv::Mat dst_img(img_h, img_w, CV_8UC3);
    cv::Mat src_img = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/Ultra_Fast_Lane_Detection_TensorRT/UFLD_C++/results/res_0001.jpg");
    double count_matmul = 0;
    double count_copy = 0;
    double count_sqrt = 0;
    double gen_points = 0;
    auto start = std::chrono::system_clock::now();
#pragma omp for
    for (int v = 0; v < img_h; v++) {
        for (int u = 0; u < img_w; u++) {
            auto before_gen = std::chrono::system_clock::now();
            // 目标图像到世界坐标系的映射
            float world_x, world_y, world_z;
            float ux, uy;
            world_x = (u - 0.5 * img_w) / scale;
            world_y = -(v - 0.5 * img_h) / scale;
            world_z = 0;

            cv::Mat pt_cam, pt_world;
            pt_world.create(4, 1, CV_64FC1);
            pt_world.at<double>(0, 0) = world_x;
            pt_world.at<double>(1, 0) = world_y;
            pt_world.at<double>(2, 0) = world_z;
            pt_world.at<double>(3, 0) = 1;
            auto after_gen = std::chrono::system_clock::now();
            gen_points += std::chrono::duration_cast<std::chrono::nanoseconds>(after_gen - before_gen).count();

            auto before_matmul = std::chrono::system_clock::now();
            pt_cam = T * pt_world;
            auto after_matmul = std::chrono::system_clock::now();
            count_matmul += std::chrono::duration_cast<std::chrono::nanoseconds>(after_matmul - before_matmul).count();

            float x = pt_cam.at<double>(0, 0);
            float y = pt_cam.at<double>(1, 0);
            float z = pt_cam.at<double>(2, 0);

            if (z < 1e-6)
                z = 1.f;

            auto before_sqrt = std::chrono::system_clock::now();
            float r = sqrtf(x * x + y * y);
            auto after_sqrt = std::chrono::system_clock::now();
            count_sqrt += std::chrono::duration_cast<std::chrono::nanoseconds>(after_sqrt - before_sqrt).count();

            auto before_copy = std::chrono::system_clock::now();
            if (r < 1e-6) {
                ux = NewCameraMatrix.at<double>(0, 2);
                uy = NewCameraMatrix.at<double>(1, 2);
            } else {
                ux = x * NewCameraMatrix.at<double>(0, 0) / z + NewCameraMatrix.at<double>(0, 2);
                uy = y * NewCameraMatrix.at<double>(1, 1) / z + NewCameraMatrix.at<double>(1, 2);
            }
            if (ux < 0 || ux > src_img.cols - 1
                || uy < 0 || uy > src_img.rows - 1) {
                dst_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
            } else// 只是为了验证畸变校正流程，为方便这里用了最近邻差值
            {
                dst_img.at<cv::Vec3b>(v, u) = src_img.at<cv::Vec3b>((int) uy, (int) ux);
            }
            auto end_copy = std::chrono::system_clock::now();
            count_copy += std::chrono::duration_cast<std::chrono::nanoseconds>(end_copy - before_copy).count();
        }
    }
    auto end = std::chrono::system_clock::now();
    cout << "gen points costs:" << gen_points << "us" << endl;
    cout << "the pt_cam = T * pt_world; costs:" << count_matmul << "us" << endl;
    cout << "sqrt costs:" << count_sqrt << "us" << endl;
    cout << "the copy costs:" << count_copy << "us" << endl;
    std::cout << "transform time is " // 只统计模型预测时间, 不包含图像预处理后处理
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " us" << std::endl;

//    cv::Mat src_img2 = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/res_0108.jpg");
//    Point2f srcPoints[4];
//    Point2f dstPoints[4];
//    srcPoints[0] = Point2f(742, 1080);
//    srcPoints[1] = Point2f(1373, 1080);
//    srcPoints[2] = Point2f(1080, 108);
//    srcPoints[3] = Point2f(1020, 108);
//
//    dstPoints[0] = Point2f(742, 1080);
//    dstPoints[1] = Point2f(1373, 1080);
//    dstPoints[2] = Point2f(1373, 108);
//    dstPoints[3] = Point2f(742, 108);
//    Mat r, warp;
//    r = getPerspectiveTransform(srcPoints, dstPoints);
//    warpPerspective(src_img2, warp, r, cv::Size(1920, 1080));
//    // cv::imshow("src",src_img);
//    cv::imwrite("warp.jpg", warp);
    cv::imwrite("dst_qujibian.jpg", dst_img);
}

void test_solvepnp4() {
    cv::Mat rvec, tvec, R, T;
    float threeDim[4][3] = {{3000, 1580,  0},
                            {3000, -1720, 0},
                            {6000, -1720, 0},
                            {6000, 1580,  0}};  // 这个点是距离光心的位置？不知道带来的误差到底有多大？
    float twoDim[4][2] = {{628,  797},
                          {1395, 802},
                          {1258, 501},
                          {763,  497}};
    // 原图经过变换算出来的点： 原始内参下的点
    // [597.415, 814.534]
    // [1430.94, 822.685]
    // [1265.53, 498.746]
    // [757.977, 495.399]
    // float twoDim[4][2] = { {807,805}, {1303,810}, {1205,493}, {904,490}};  // 去畸变的点

    vector<Point3f> outDim;
    vector<Point2f> inDim;
    vector<float> distCoeff(0);

    for (int i = 0; i < 4; i++) {
        outDim.push_back(Point3f(threeDim[i][0], threeDim[i][1], threeDim[i][2]));
        inDim.push_back(Point2f(twoDim[i][0], twoDim[i][1]));
    }
    const cv::Mat K = (cv::Mat_<double>(3, 3)
            << 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0);
    const cv::Mat D = (cv::Mat_<double>(4, 1)
            << -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457);
    const cv::Mat D2 = (cv::Mat_<double>(5, 1)
            << 0, 0, 0, 0, 0);
    const int ImgWidth = 1920;
    const int ImgHeight = 1080;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 0;
    const cv::Mat K2 = (cv::Mat_<double>(3, 3)
            << 1010.1193051331736 / 1.4, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831 /
                                                                       1.0, 577.4709247205907, 0.0, 0.0, 1.0);
    cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K2, D, imageSize, alpha, imageSize, 0);
    cout << NewCameraMatrix << endl;
//
//    Mat sourceImage = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
//    namedWindow("Source", 1);
//    for (int i = 0; i < inDim.size(); ++i) {
//        circle(sourceImage, inDim[i], 3, Scalar(0, 255, 0), -1, 8);
//    }
    // imshow("Source",sourceImage);
    // 计算得到原始图像中像素点坐标在畸变校正后的图像中的坐标

    for (int i = 0; i < inDim.size(); i++) {
        cv::Matx33d K;
        cv::Vec4d D;
        /*[599.1767578125, 0, 1051.742061275407;
         0, 989.3424682617188, 571.4513472305662;
         0, 0, 1]
         * */
//        K(0, 0) = 599.1767578125;
//        K(0, 1) = 0;
//        K(0, 2) = 1051.742061275407;
//        K(1, 0) = 0;
//        K(1, 1) = 989.3424682617188;
//        K(1, 2) = 571.4513472305662;
//        K(2, 0) = 0;
//        K(2, 1) = 0;
//        K(2, 2) = 1;
        // 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0
        K(0, 0) = 1010.1193051331736;
        K(0, 1) = 0;
        K(0, 2) = 1007.7481675024154;
        K(1, 0) = 0;
        K(1, 1) = 1009.4943272753831;
        K(1, 2) = 577.4709247205907;
        K(2, 0) = 0;
        K(2, 1) = 0;
        K(2, 2) = 1;

        // -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457
        D(0) = -0.06663067168381479;
        D(1) = 0.0009026610617662017;
        D(2) = -0.007498635027107796;
        D(3) = 0.0019139336144852457;
        ProjectPointFromOriginToUndistorted(K, D, inDim[i], inDim[i]);
    }
    for (int i = 0; i < inDim.size(); ++i) {
        cout << inDim[i] << endl;
    }

    // 计算得到旋转向量和平移向量，将旋转向量转换为旋转矩阵
    cv::solvePnP(outDim, inDim, K, D2, rvec, tvec);
    cv::Rodrigues(rvec, R);

    T.create(4, 4, CV_64FC1);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec.at<double>(0, 0);
    T.at<double>(1, 3) = tvec.at<double>(1, 0);
    T.at<double>(2, 3) = tvec.at<double>(2, 0);
    T.at<double>(3, 0) = T.at<double>(3, 1) = T.at<double>(3, 2) = 0;
    T.at<double>(3, 3) = 1;
//    cout<<R<<endl;
//    cout<<T<<endl;
    // 第一个方法计算出来的RT矩阵,和这个方法算出来的一样的
    // RT_[0.002919347595132238, -0.9999813796559083, 0.005358894576756645, -57.04754470101124;
    // -0.4945244400306179, -0.00610144434352311, -0.8691423074441414, 2433.772627727664;
    // 0.8691588207123151, -0.0001127758346887164, -0.4945330440522541, 1376.410228531058]

    // 验证计算得到的外参的准确性，通过利用计算得到的外参将世界坐标投影至像素坐标系
    // 并将结果与输入的像素坐标对比来验证
    for (int i = 0; i < inDim.size(); i++) {
        // 世界坐标系变换至相机坐标系
        float ux, uy;
        cv::Mat pt_cam, pt_world;
        pt_world.create(4, 1, CV_64FC1);
        pt_world.at<double>(0, 0) = outDim[i].x;
        pt_world.at<double>(1, 0) = outDim[i].y;
        pt_world.at<double>(2, 0) = outDim[i].z;
        pt_world.at<double>(3, 0) = 1;

        pt_cam = T * pt_world;

        // 相机坐标系变换至像素坐标系
        float x = pt_cam.at<double>(0, 0);
        float y = pt_cam.at<double>(1, 0);
        float z = pt_cam.at<double>(2, 0);

        if (z < 1e-6)
            z = 1.f;

        float r = sqrtf(x * x + y * y);  // 低精度快速的sqrt

        if (r < 1e-6) {
            // 位于光心
            ux = K.at<double>(0, 2);
            uy = K.at<double>(1, 2);
        } else {
            float theta = atan2f(r, z);  // 求两个向量的夹角

            float theta_d = 1.f;
            float res = 1.f; // 公式里面提取了一个公因式，然后多了个1，所以这边是1
            for (int i = 0; i < 4; ++i) {
                theta_d *= theta * theta;
                res += D.at<double>(i, 0) * theta_d;
            }

            res *= theta; // 公因式得多乘一个Θ
            // 也就是实际可以是先去畸变，再去通过内参转换到像素坐标系
            ux = x * K.at<double>(0, 0) * res / r + K.at<double>(0, 2);
            uy = y * K.at<double>(1, 1) * res / r + K.at<double>(1, 2);
        }
        printf("ux, uy: %f %f\n", ux, uy);
    }

    // 将图像透视变换至地面
    int img_w = 1920;
    int img_h = 1080;
    float world_w = 70000;// 单位为mm
    float scale = img_w / world_w;
    cv::Mat dst_img(img_h, img_w, CV_8UC3);
    cv::Mat src_img = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/jibian/1053.jpg");
    for (int v = 0; v < img_h; v++) {
        for (int u = 0; u < img_w; u++) {
            // 目标图像到世界坐标系的映射
            float world_x, world_y, world_z;
            float ux, uy;
            world_x = (u - 0.5 * img_w) / scale;  // 为什么要减去0.5*我，除以scale
            world_y = -(v - 0.5 * img_h) / scale;
            world_z = 0;

            cv::Mat pt_cam, pt_world;
            pt_world.create(4, 1, CV_64FC1);
            pt_world.at<double>(0, 0) = world_x;
            pt_world.at<double>(1, 0) = world_y;
            pt_world.at<double>(2, 0) = world_z;
            pt_world.at<double>(3, 0) = 1;

            pt_cam = T * pt_world;

            float x = pt_cam.at<double>(0, 0);
            float y = pt_cam.at<double>(1, 0);
            float z = pt_cam.at<double>(2, 0);

            if (z < 1e-6)  // 可能是太小了默认就是1？atan计算？
                z = 1.f;

            float r = sqrtf(x * x + y * y);

            if (r < 1e-6) {
                ux = K.at<double>(0, 2);
                uy = K.at<double>(1, 2);
            } else {
                float theta = atan2f(r, z);

                float theta_d = 1.f;
                float res = 1.f; // 公式里面提取了一个公因式，然后多了个1，所以这边是1
                for (int i = 0; i < 4; ++i) {
                    theta_d *= theta * theta;
                    res += D.at<double>(i, 0) * theta_d;
                }

                res *= theta;  // 公因式得多乘一个Θ

                ux = x * K.at<double>(0, 0) * res / r + K.at<double>(0, 2);  // 行
                uy = y * K.at<double>(1, 1) * res / r + K.at<double>(1, 2);  // 列
            }
            // 这边就得到了真实的3D点在像素坐标系下的坐标，如果计算出来的
            if (ux < 0 || ux > src_img.cols - 1
                || uy < 0 || uy > src_img.rows - 1) {
                dst_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);  // 8U 类型的 RGB 彩色图像  (0-255)
            } else// 只是为了验证畸变校正流程，为方便这里用了最近邻差值
            {
                dst_img.at<cv::Vec3b>(v, u) = src_img.at<cv::Vec3b>((int) uy, (int) ux);
            }
        }
    }
    cv::imwrite("dst.jpg", dst_img);

}

// 测试3D点能不能在转回像素坐标，也是验证的第一步
void test_solvepnp3() {
    std::vector<cv::Point2f> pixel_uv_vec;
    std::vector<cv::Point3f> point_world_vec;
    cv::Mat rvec, tvec, R, T;

    cv::Mat K(3, 3, CV_64FC1);
    K.at<double>(0, 0) = 290.62;
    K.at<double>(0, 1) = 0;
    K.at<double>(0, 2) = 643.44;
    K.at<double>(1, 0) = 0;
    K.at<double>(1, 1) = 290.62;
    K.at<double>(1, 2) = 365.61;
    K.at<double>(2, 0) = 0;
    K.at<double>(2, 1) = 0;
    K.at<double>(2, 2) = 1;

    cv::Mat D(4, 1, CV_64FC1);
    D.at<double>(0, 0) = 0.165618;
    D.at<double>(1, 0) = 0.020838;
    D.at<double>(2, 0) = -0.023782;
    D.at<double>(3, 0) = 0.002512;

    point_world_vec.push_back(cv::Point3f(-3100, 0, 0));
    point_world_vec.push_back(cv::Point3f(-2250, 850, 0));
    point_world_vec.push_back(cv::Point3f(-1400, 0, 0));
    point_world_vec.push_back(cv::Point3f(-2250, -850, 0));
    point_world_vec.push_back(cv::Point3f(1400, 0, 0));
    point_world_vec.push_back(cv::Point3f(2250, 850, 0));
    point_world_vec.push_back(cv::Point3f(3100, 0, 0));
    point_world_vec.push_back(cv::Point3f(2250, -850, 0));

    pixel_uv_vec.push_back(cv::Point2f(298, 373));
    pixel_uv_vec.push_back(cv::Point2f(419, 338));
    pixel_uv_vec.push_back(cv::Point2f(441, 380));
    pixel_uv_vec.push_back(cv::Point2f(266, 444));
    pixel_uv_vec.push_back(cv::Point2f(835, 383));
    pixel_uv_vec.push_back(cv::Point2f(862, 341));
    pixel_uv_vec.push_back(cv::Point2f(987, 380));
    pixel_uv_vec.push_back(cv::Point2f(1013, 453));

    // 计算得到原始图像中像素点坐标在畸变校正后的图像中的坐标
    for (int i = 0; i < pixel_uv_vec.size(); i++) {
        cv::Matx33d K;
        cv::Vec4d D;
        K(0, 0) = 290.62;
        K(0, 1) = 0;
        K(0, 2) = 643.44;
        K(1, 0) = 0;
        K(1, 1) = 290.62;
        K(1, 2) = 365.61;
        K(2, 0) = 0;
        K(2, 1) = 0;
        K(2, 2) = 1;
        D(0) = 0.165618;
        D(1) = 0.020838;
        D(2) = -0.023782;
        D(3) = 0.002512;
        ProjectPointFromOriginToUndistorted(K, D, pixel_uv_vec[i], pixel_uv_vec[i]);
    }

    // 计算得到旋转向量和平移向量，将旋转向量转换为旋转矩阵
    cv::solvePnP(point_world_vec, pixel_uv_vec, K, cv::Mat(), rvec, tvec);
    cv::Rodrigues(rvec, R);

    T.create(4, 4, CV_64FC1);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec.at<double>(0, 0);
    T.at<double>(1, 3) = tvec.at<double>(1, 0);
    T.at<double>(2, 3) = tvec.at<double>(2, 0);
    T.at<double>(3, 0) = T.at<double>(3, 1) = T.at<double>(3, 2) = 0;
    T.at<double>(3, 3) = 1;

    // 验证计算得到的外参的准确性，通过利用计算得到的外参将世界坐标投影至像素坐标系
    // 并将结果与输入的像素坐标对比来验证
    for (int i = 0; i < point_world_vec.size(); i++) {
        // 世界坐标系变换至相机坐标系
        float ux, uy;
        cv::Mat pt_cam, pt_world;
        pt_world.create(4, 1, CV_64FC1);
        pt_world.at<double>(0, 0) = point_world_vec[i].x;
        pt_world.at<double>(1, 0) = point_world_vec[i].y;
        pt_world.at<double>(2, 0) = point_world_vec[i].z;
        pt_world.at<double>(3, 0) = 1;

        pt_cam = T * pt_world;

        // 相机坐标系变换至像素坐标系.
        float x = pt_cam.at<double>(0, 0);
        float y = pt_cam.at<double>(1, 0);
        float z = pt_cam.at<double>(2, 0);

        if (z < 1e-6)
            z = 1.f;

        float r = sqrtf(x * x + y * y);

        if (r < 1e-6) {
            // 位于光心
            ux = K.at<double>(0, 2);
            uy = K.at<double>(1, 2);
        } else {
            float theta = atan2f(r, z);

            float theta_d = 1.f;
            float res = 1.f;
            for (int i = 0; i < 4; ++i) {
                theta_d *= theta * theta;
                res += D.at<double>(i, 0) * theta_d;
            }

            res *= theta;

            ux = x * K.at<double>(0, 0) * res / r + K.at<double>(0, 2);
            uy = y * K.at<double>(1, 1) * res / r + K.at<double>(1, 2);
        }
        printf("ux, uy: %f %f\n", ux, uy);
    }
}

void get_new_image() {
    cv::Mat img, undistortImg;
    cv::Matx33d K, P;
    cv::Vec4d D;
    cv::Mat mapX, mapY;
    img = cv::imread("img.bmp");

    K(0, 0) = 348.52;
    K(0, 1) = 0;
    K(0, 2) = 640.19;
    K(1, 0) = 0;
    K(1, 1) = 348.52;
    K(1, 2) = 358.56;
    K(2, 0) = 0;
    K(2, 1) = 0;
    K(2, 2) = 1;

    D(0) = 0.066258;
    D(1) = 0.039769;
    D(2) = -0.026906;
    D(3) = 0.003342;

    P = K;
    P(0, 0) /= 1.5;
    P(1, 1) /= 1.5;

    // 这个也是一种去畸变的方法
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(),
                                         P, cv::Size(img.cols, img.rows), CV_16SC2, mapX, mapY);

    cv::remap(img, undistortImg, mapX, mapY, CV_INTER_LINEAR);
    cv::imshow("src", img);
    cv::imshow("corrected", undistortImg);
    cv::imwrite("corrected3.bmp", undistortImg);
    cv::waitKey();
}

void test_solvepnp5() {
    cv::Mat rvec, tvec, R, T;
    // 点是按照1   2
    //       4   3
    float threeDim[4][3] = {{3000, 1500,  0}, // x y z的顺序
                            {3000, -1680, 0},
                            {6000, -1680, 0},
                            {6000, 1500,  0}};  // 这个点是距离光心的位置？
//    float twoDim[4][2] = {{628,  797},
//                          {1395, 802},
//                          {1258, 501},
//                          {763,  497}};
    // 原图经过变换算出来的点： 原始内参下的点    这个是鱼眼模型算出来的点
    // [597.415, 814.534]
    // [1430.94, 822.685]
    // [1265.53, 498.746]
    // [757.977, 495.399]
//    float twoDim[4][2] = {{651,  809},
//                          {1233, 822},
//                          {1108, 490},
//                          {761,  483}};  // 去畸变的点 除以1.4
    float twoDim[4][2] = {{685,  811},
                          {1194, 822},
                          {1086, 491},
                          {782,  484}};  // 去畸变的点 除以1.6

    vector<Point3f> outDim;
    vector<Point2f> inDim;
    vector<float> distCoeff(0);

    for (int i = 0; i < 4; i++) {
        outDim.push_back(Point3f(threeDim[i][0], threeDim[i][1], threeDim[i][2]));
        inDim.push_back(Point2f(twoDim[i][0], twoDim[i][1]));
    }
    // const cv::Mat K = (cv::Mat_<double>(3, 3)
    //         << 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0);
    const cv::Mat D = (cv::Mat_<double>(4, 1)
            << -0.0526858350541784, -0.01873269061565343, 0.0060846931831152, -0.0016727061237763216);
    const cv::Mat D2 = (cv::Mat_<double>(5, 1)
            << 0, 0, 0, 0, 0);
    const int ImgWidth = 1920;
    const int ImgHeight = 1080;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 0;
    const cv::Mat K2 = (cv::Mat_<double>(3, 3)
            << 1003.9989013289942 / 1.6, 0.0, 926.3763250309561, 0.0, 1004.1132782586517 / 1.0, 546.1004237610695, 0.0, 0.0, 1.0);
    cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K2, D, imageSize, alpha, imageSize, 0);
    cout << NewCameraMatrix << endl;
//
//    Mat sourceImage = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
//    namedWindow("Source", 1);
//    for (int i = 0; i < inDim.size(); ++i) {
//        circle(sourceImage, inDim[i], 3, Scalar(0, 255, 0), -1, 8);
//    }
    // imshow("Source",sourceImage);
    // 计算得到原始图像中像素点坐标在畸变校正后的图像中的坐标

//    for (int i = 0; i < inDim.size(); i++) {
//        cv::Matx33d K;
//        cv::Vec4d D;
//        /*[599.1767578125, 0, 1051.742061275407;
//         0, 989.3424682617188, 571.4513472305662;
//         0, 0, 1]
//         * */
//        K(0, 0) = 599.1767578125;
//        K(0, 1) = 0;
//        K(0, 2) = 1051.742061275407;
//        K(1, 0) = 0;
//        K(1, 1) = 989.3424682617188;
//        K(1, 2) = 571.4513472305662;
//        K(2, 0) = 0;
//        K(2, 1) = 0;
//        K(2, 2) = 1;
//        // 1010.1193051331736, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831, 577.4709247205907, 0.0, 0.0, 1.0
////        K(0, 0) = 1010.1193051331736;
////        K(0, 1) = 0;
////        K(0, 2) = 1007.7481675024154;
////        K(1, 0) = 0;
////        K(1, 1) = 1009.4943272753831;
////        K(1, 2) = 577.4709247205907;
////        K(2, 0) = 0;
////        K(2, 1) = 0;
////        K(2, 2) = 1;
//
//        // -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457
//        D(0) = 0;
//        D(1) = 0;
//        D(2) = 0;
//        D(3) = 0;
//        ProjectPointFromOriginToUndistorted(K, D, inDim[i], inDim[i]);
//    }
    for (int i = 0; i < inDim.size(); ++i) {
        cout << inDim[i] << endl;
    }

    // 计算得到旋转向量和平移向量，将旋转向量转换为旋转矩阵
    cv::solvePnP(outDim, inDim, NewCameraMatrix, D2, rvec, tvec);
    cv::Rodrigues(rvec, R);

    T.create(4, 4, CV_64FC1);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec.at<double>(0, 0);
    T.at<double>(1, 3) = tvec.at<double>(1, 0);
    T.at<double>(2, 3) = tvec.at<double>(2, 0);
    T.at<double>(3, 0) = T.at<double>(3, 1) = T.at<double>(3, 2) = 0;
    T.at<double>(3, 3) = 1;
//    cout<<R<<endl;
//    cout<<T<<endl;
    // 第一个方法计算出来的RT矩阵,和这个方法算出来的一样的
    // RT_[0.002919347595132238, -0.9999813796559083, 0.005358894576756645, -57.04754470101124;
    // -0.4945244400306179, -0.00610144434352311,inDim -0.8691423074441414, 2433.772627727664;
    // 0.8691588207123151, -0.0001127758346887164, -0.4945330440522541, 1376.410228531058]

    // 验证计算得到的外参的准确性，通过利用计算得到的外参将世界坐标投影至像素坐标系
    // 并将结果与输入的像素坐标对比来验证
    for (int i = 0; i < inDim.size(); i++) {
        // 世界坐标系变换至相机坐标系
        float ux, uy;
        cv::Mat pt_cam, pt_world;
        pt_world.create(4, 1, CV_64FC1);
        pt_world.at<double>(0, 0) = outDim[i].x;
        pt_world.at<double>(1, 0) = outDim[i].y;
        pt_world.at<double>(2, 0) = outDim[i].z;
        pt_world.at<double>(3, 0) = 1;

        pt_cam = T * pt_world;

        // 相机坐标系变换至像素坐标系
        float x = pt_cam.at<double>(0, 0);
        float y = pt_cam.at<double>(1, 0);
        float z = pt_cam.at<double>(2, 0);

        if (z < 1e-6)
            z = 1.f;

        float r = sqrtf(x * x + y * y);

        if (r < 1e-6) {
            // 位于光心
            ux = NewCameraMatrix.at<double>(0, 2);
            uy = NewCameraMatrix.at<double>(1, 2);
        } else {
            ux = x * NewCameraMatrix.at<double>(0, 0) / z + NewCameraMatrix.at<double>(0, 2);
            uy = y * NewCameraMatrix.at<double>(1, 1) / z + NewCameraMatrix.at<double>(1, 2);
        }
        printf("ux, uy: %f %f  -> x,y: %f %f\n", ux, uy, inDim[i].x, inDim[i].y);
    }

    // 将图像透视变换至地面
//    int img_w = 1920 / 3;
//    int img_h = 1080 / 3;
    int img_w = 1920;
    int img_h = 1080;
    float world_w = 70000;// 单位为mm
    float scale = img_w / world_w;
    cv::Mat dst_img(img_h, img_w, CV_8UC3);
    cv::Mat src_img = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/2023-02-15-16-34-50-qujibian/frame0000.jpg");
    double count_matmul = 0;
    double count_copy = 0;
    double count_sqrt = 0;
    double gen_points = 0;
    auto start = std::chrono::system_clock::now();
#pragma omp for
    for (int v = 0; v < img_h; v++) {
        for (int u = 0; u < img_w; u++) {
            auto before_gen = std::chrono::system_clock::now();
            // 目标图像到世界坐标系的映射
            float world_x, world_y, world_z;
            float ux, uy;
            world_x = (u - 0.5 * img_w) / scale;
            world_y = -(v - 0.5 * img_h) / scale;
            world_z = 0;

            cv::Mat pt_cam, pt_world;
            pt_world.create(4, 1, CV_64FC1);
            pt_world.at<double>(0, 0) = world_x;
            pt_world.at<double>(1, 0) = world_y;
            pt_world.at<double>(2, 0) = world_z;
            pt_world.at<double>(3, 0) = 1;
            auto after_gen = std::chrono::system_clock::now();
            gen_points += std::chrono::duration_cast<std::chrono::nanoseconds>(after_gen - before_gen).count();

            auto before_matmul = std::chrono::system_clock::now();
            pt_cam = T * pt_world;
            auto after_matmul = std::chrono::system_clock::now();
            count_matmul += std::chrono::duration_cast<std::chrono::nanoseconds>(after_matmul - before_matmul).count();

            float x = pt_cam.at<double>(0, 0);
            float y = pt_cam.at<double>(1, 0);
            float z = pt_cam.at<double>(2, 0);

            if (z < 1e-6)
                z = 1.f;

            auto before_sqrt = std::chrono::system_clock::now();
            float r = sqrtf(x * x + y * y);
            auto after_sqrt = std::chrono::system_clock::now();
            count_sqrt += std::chrono::duration_cast<std::chrono::nanoseconds>(after_sqrt - before_sqrt).count();

            auto before_copy = std::chrono::system_clock::now();
            if (r < 1e-6) {
                ux = NewCameraMatrix.at<double>(0, 2);
                uy = NewCameraMatrix.at<double>(1, 2);
            } else {
                ux = x * NewCameraMatrix.at<double>(0, 0) / z + NewCameraMatrix.at<double>(0, 2);
                uy = y * NewCameraMatrix.at<double>(1, 1) / z + NewCameraMatrix.at<double>(1, 2);
            }
            if (ux < 0 || ux > src_img.cols - 1
                || uy < 0 || uy > src_img.rows - 1) {
                dst_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
            } else// 只是为了验证畸变校正流程，为方便这里用了最近邻差值
            {
                dst_img.at<cv::Vec3b>(v, u) = src_img.at<cv::Vec3b>((int) uy, (int) ux);
            }
            auto end_copy = std::chrono::system_clock::now();
            count_copy += std::chrono::duration_cast<std::chrono::nanoseconds>(end_copy - before_copy).count();
        }
    }
    auto end = std::chrono::system_clock::now();
    cout << "gen points costs:" << gen_points << "us" << endl;
    cout << "the pt_cam = T * pt_world; costs:" << count_matmul << "us" << endl;
    cout << "sqrt costs:" << count_sqrt << "us" << endl;
    cout << "the copy costs:" << count_copy << "us" << endl;
    std::cout << "transform time is " // 只统计模型预测时间, 不包含图像预处理后处理
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " us" << std::endl;

//    cv::Mat src_img2 = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/res_0108.jpg");
//    Point2f srcPoints[4];
//    Point2f dstPoints[4];
//    srcPoints[0] = Point2f(742, 1080);
//    srcPoints[1] = Point2f(1373, 1080);
//    srcPoints[2] = Point2f(1080, 108);
//    srcPoints[3] = Point2f(1020, 108);
//
//    dstPoints[0] = Point2f(742, 1080);
//    dstPoints[1] = Point2f(1373, 1080);
//    dstPoints[2] = Point2f(1373, 108);
//    dstPoints[3] = Point2f(742, 108);
//    Mat r, warp;
//    r = getPerspectiveTransform(srcPoints, dstPoints);
//    warpPerspective(src_img2, warp, r, cv::Size(1920, 1080));
//    // cv::imshow("src",src_img);
//    cv::imwrite("warp.jpg", warp);
    cv::imwrite("dst_qujibian.jpg", dst_img);
}

// 验证了3D->2D是没什么问题的，用的是去畸变的像素点，solvepnp用的畸变系数为0，2D->3D也验证了
void test_solvepnp5_2() {
    // 看博客好像是坐标系没什么关系，对这个3D点，比如我就选择车头正中间的地面上的点作为世界坐标系的原点
    float threeDim[4][3] = {{3000, 1500,  0}, // x y z的顺序
                            {3000, -1680, 0},
                            {6000, -1680, 0},
                            {6000, 1500,  0}};  // 这个点是距离光心的位置？
    // float twoDim[4][2] = { {628,797}, {1395,802}, {1258,501}, {763,497}};  // 原图
    float twoDim[4][2] = {{685,  811},
                          {1194, 822},
                          {1086, 491},
                          {782,  484}};  // 去畸变的点 除以1.6
    float lane[18][2] = {
//            {731,  1075},
//            {752,  1022},
//            {763,  970},
//            {783,  917},
//            {794,  865},
//            {810,  812},
//            {827,  760},
//            {836,  707},
//            {852,  655},
//            {869,  602},
//            {883,  550},
//            {899,  497},
//            {918,  445},
//            {934,  392},
//            {953,  340},
//            {971,  287},
//            {989,  235},
//            {1007, 182},
            {299, 1075},
            {333, 1022},
            {368, 970},
            {411, 917},
            {446, 865},
            {484, 812},
            {523, 760},
            {564, 707},
            {601, 655},
            {638, 602},
            {672, 550},
            {709, 497},
            {752, 445},
            {786, 392},
            {821, 340},
            {860, 287},
            {896, 235},
            {930, 182},
    };
    float lane2[18][2] = {
//            {1353,  1075},
//            {1348,  1022},
//            {1332,  970},
//            {1318,  917},
//            {1312,  865},
//            {1298,  812},
//            {1281,  760},
//            {1265,  707},
//            {1247,  655},
//            {1232,  602},
//            {1213,  550},
//            {1202,  497},
//            {1181,  445},
//            {1163,  392},
//            {1142,  340},
//            {1128,  287},
//            {1108,  235},
//            {1095, 182},
            {1796, 1075},
            {1761, 1022},
            {1724, 970},
            {1689, 917},
            {1650, 865},
            {1612, 812},
            {1576, 760},
            {1540, 707},
            {1504, 655},
            {1467, 602},
            {1432, 550},
            {1394, 497},
            {1367, 445},
            {1320, 392},
            {1280, 340},
            {1240, 287},
            {1203, 235},
            {1166, 182},
    };


    vector<Point2f> lane_;
    vector<Point2f> lane2_;
    for (int i = 0; i < 18; i++) {
        lane_.push_back(Point2f(lane[i][0], lane[i][1]));
        lane2_.push_back(Point2f(lane2[i][0], lane2[i][1]));
    }
    vector<Point3f> outDim;
    vector<Point2f> inDim;
    vector<float> distCoeff(0);

    for (int i = 0; i < 4; i++) {
        outDim.push_back(Point3f(threeDim[i][0], threeDim[i][1], threeDim[i][2]));
        inDim.push_back(Point2f(twoDim[i][0], twoDim[i][1]));
    }

//    const cv::Mat D = (cv::Mat_<double>(4, 1)
//            << -0.06663067168381479, 0.0009026610617662017, -0.007498635027107796, 0.0019139336144852457);
    const cv::Mat D = (cv::Mat_<double>(4, 1)
            << -0.0526858350541784, -0.01873269061565343, 0.0060846931831152, -0.0016727061237763216);
    const int ImgWidth = 1920;
    const int ImgHeight = 1080;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 0;
//    const cv::Mat K2 = (cv::Mat_<double>(3, 3)
//            << 1010.1193051331736 / 1.4, 0.0, 1007.7481675024154, 0.0, 1009.4943272753831 /
//                                                                       1.0, 577.4709247205907, 0.0, 0.0, 1.0);
    const cv::Mat K2 = (cv::Mat_<double>(3, 3)
            << 1003.9989013289942 / 1.6, 0.0, 926.3763250309561, 0.0, 1004.1132782586517 / 1.0, 546.1004237610695, 0.0, 0.0, 1.0);
    cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K2, D, imageSize, alpha, imageSize, 0);

    Mat sourceImage = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/2023-02-15-16-34-50-qujibian/frame0000.jpg");
    namedWindow("Source", 1);
    for (int i = 0; i < inDim.size(); ++i) {
        circle(sourceImage, inDim[i], 3, Scalar(0, 255, 0), -1, 8);
    }
    // imshow("Source",sourceImage);


//    Mat cameraMatrix(3,3,CV_32F);
//    float tempMatrix[3][3] = { { 2697.6,0 ,597.4 }, { 0, 2682,515.6 }, { 0, 0 ,1} };
//    for (int i = 0; i < 3;i++)
//    {
//        for (int j = 0; j < 3;j++)
//        {
//            cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
//        }
//    }

    Mat rvec1, tvec1;
    solvePnP(outDim, inDim, NewCameraMatrix, Mat(), rvec1, tvec1);
    cout << rvec1 << endl;
    cout << tvec1 << endl;
    cout << "11111111----------------------------------11111111" << endl;
    cv::Mat rvecM1(3, 3, cv::DataType<double>::type);  //旋转矩阵
    Rodrigues(rvec1, rvecM1);
    cout << rvecM1 << endl;
    cout << tvec1 << endl;
    cout<<NewCameraMatrix<<endl;

    // 此处用于求相机位于坐标系内的旋转角度,2D-3D的转换并不用求,
    // 这边几个角度没什么用
    const double PI = 3.1415926;
    double thetaZ = atan2(rvecM1.at<double>(1, 0), rvecM1.at<double>(0, 0)) / PI * 180;
    double thetaY = atan2(-1 * rvecM1.at<double>(2, 0), sqrt(rvecM1.at<double>(2, 1) * rvecM1.at<double>(2, 1)
                                                             + rvecM1.at<double>(2, 2) * rvecM1.at<double>(2, 2))) /
                    PI * 180;
    double thetaX = atan2(rvecM1.at<double>(2, 1), rvecM1.at<double>(2, 2)) / PI * 180;
    cout << "theta x  " << thetaX << endl << "theta Y: " << thetaY << endl << "theta Z: " << thetaZ << endl;


    ///根据公式求Zc，即s
    cv::Mat imagePoint = cv::Mat::ones(3, 1, cv::DataType<double>::type);
    cv::Mat tempMat, tempMat2;
    //输入一个2D坐标点，便可以求出相应的s
    imagePoint.at<double>(0, 0) = 904;
    imagePoint.at<double>(1, 0) = 490;
    double zConst = 0;//实际坐标系的距离
    //计算参数s
    double s;
    tempMat = rvecM1.inv() * NewCameraMatrix.inv() * imagePoint;  // M1矩阵
    tempMat2 = rvecM1.inv() * tvec1;  // M2矩阵
    s = zConst + tempMat2.at<double>(2, 0);
    s /= tempMat.at<double>(2, 0);
    cout << "s : " << s << endl;

    // **********************************
    ///3D to 2D
    cv::Mat worldPoints = Mat::ones(4, 1, cv::DataType<double>::type);
    worldPoints.at<double>(0, 0) = 6000;
    worldPoints.at<double>(1, 0) = 1500;
    worldPoints.at<double>(2, 0) = 0;
    cout << "world Points :  " << worldPoints << endl;
    Mat image_points = Mat::ones(3, 1, cv::DataType<double>::type);
    //setIdentity(image_points);

    // 下面这个流程能不能替换到透视变换那边
    Mat RT_;
    hconcat(rvecM1, tvec1, RT_);
    cout << "RT_" << RT_ << endl;
    image_points = NewCameraMatrix * RT_ * worldPoints;
    Mat D_Points = Mat::ones(3, 1, cv::DataType<double>::type);
    D_Points.at<double>(0, 0) = image_points.at<double>(0, 0) / image_points.at<double>(2, 0);
    D_Points.at<double>(1, 0) = image_points.at<double>(1, 0) / image_points.at<double>(2, 0);
    //cv::projectPoints(worldPoints, rvec1, tvec1, cameraMatrix1, distCoeffs1, imagePoints);
    cout << "3D to 2D:   " << D_Points << endl;

    //camera_coordinates
    Mat camera_cordinates = -rvecM1.inv() * tvec1;

    for (int i = 0; i < 18; ++i) {
        imagePoint.at<double>(0, 0) = lane_[i].x;;
        imagePoint.at<double>(1, 0) = lane_[i].y;
        double zConst = 0;//实际坐标系的距离
        //计算参数s
        double s;
        tempMat = rvecM1.inv() * NewCameraMatrix.inv() * imagePoint;
        tempMat2 = rvecM1.inv() * tvec1;
        s = zConst + tempMat2.at<double>(2, 0);
        s /= tempMat.at<double>(2, 0);
        // cout << "s : " << s << endl;
        cv::Mat imagePoint_your_know = cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
        imagePoint_your_know.at<double>(0, 0) = lane_[i].x;
        imagePoint_your_know.at<double>(1, 0) = lane_[i].y;
        Mat wcPoint = rvecM1.inv() * (NewCameraMatrix.inv() * s * imagePoint_your_know - tvec1);
        Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
        // cout << "2D to 3D :" << worldPoint << endl;

        imagePoint.at<double>(0, 0) = lane2_[i].x;;
        imagePoint.at<double>(1, 0) = lane2_[i].y;
        double zConst2 = 0;//实际坐标系的距离
        //计算参数s
        double s2;
        tempMat = rvecM1.inv() * NewCameraMatrix.inv() * imagePoint;
        tempMat2 = rvecM1.inv() * tvec1;
        s2 = zConst2 + tempMat2.at<double>(2, 0);
        s2 /= tempMat.at<double>(2, 0);
        // cout << "s : " << s << endl;
        cv::Mat imagePoint_your_know2 = cv::Mat::ones(3, 1, cv::DataType<double>::type); //u,v,1
        imagePoint_your_know2.at<double>(0, 0) = lane2_[i].x;
        imagePoint_your_know2.at<double>(1, 0) = lane2_[i].y;
        Mat wcPoint2 = rvecM1.inv() * (NewCameraMatrix.inv() * s * imagePoint_your_know2 - tvec1);
        Point3f worldPoint2(wcPoint2.at<double>(0, 0), wcPoint2.at<double>(1, 0), wcPoint2.at<double>(2, 0));
        // cout << "2D to 3D :" << worldPoint2 << endl;
        cout << "2D to 3D :" << worldPoint.y - worldPoint2.y << endl;
    }
//    int img_w = 1920;
//    int img_h = 1080;
//    cv::Mat dst_img(img_h,img_w,CV_8UC3);
//    cv::Mat src_img = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
//    cv::warpPerspective(src_img,dst_img,rvecM1*tvec1,cv::Size(img_w,img_h));
//
//
//
//    imshow("Source", sourceImage);
//    imshow("Source2", dst_img);
//    waitKey(0);

}

// Mat.at<存储类型名称>(行，列)[通道]
int main() {
    // test_solvepnp4();
    // test_solvepnp2();
    // test_solvepnp3();
    // test_solvepnp();
    // OpenCVFisheyeImageUndistortion();
    test_solvepnp5();
    test_solvepnp5_2();
    Mat *img = 0;
//     Mat src = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/qujibian/1053.jpg");
    Mat src = imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/2023-02-15-16-34-50-qujibian/frame0000.jpg");
    img = &src;
    Mat src2 = src.clone();
    namedWindow("original image", WINDOW_AUTOSIZE);
    cv::setMouseCallback("original image", onMouse, reinterpret_cast<void *> (img));//注册鼠标操作(回调)函数
    imshow("original image", src);

    waitKey();
    return 0;
}

