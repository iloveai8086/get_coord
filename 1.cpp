//
// Created by ros on 8/16/22.
//


void test_solvepnp4() {
    cv::Mat rvec, tvec, R, T;
    float threeDim[4][3] = {{3000, 1580,  0},
                            {3000, -1720, 0},
                            {6000, -1720, 0},
                            {6000, 1580,  0}};  // 这个点是距离光心的位置？
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







