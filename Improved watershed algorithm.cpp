#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("E:\\1.bmp"); // 读取图像
    Mat scrGray, scrmedian;
    cvtColor(img, scrGray, COLOR_BGR2GRAY);//转化为灰度图像
    medianBlur(scrGray, scrmedian, 3);//中值滤波
    // 二值化处理
    Mat binaryImg;
    threshold(scrmedian, binaryImg, 127, 255, THRESH_BINARY);
    // 连通域分析
    Mat labels;
    Mat stats;
    Mat centroids;
    int n = connectedComponentsWithStats(binaryImg, labels, stats, centroids);
    // 绘制每个连通域的凸包
    vector<double>PerArea(n, 0);//编程需注意变量作用域
    vector<double>PerArea2(n, 0);
    //要添加忽略连通域面积较小的，否则把噪点也给弄进去了
    for (int i = 1; i < n; i++)
    {
        // 提取当前连通域的像素坐标
        vector<Point> points;
        for (int row = stats.at<int>(i, cv::CC_STAT_TOP); row < stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT); row++)
        {
            for (int col = stats.at<int>(i, cv::CC_STAT_LEFT); col < stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH); col++)
            {
                if (labels.at<int>(row, col) == i)
                {
                    points.push_back(Point(col, row));//为什么col和row的顺序换了？
                }
            }
        }
        // 计算当前连通域的凸包
        vector<Point> hull;
        convexHull(points, hull);
        if (points.size() < 20)//15这个值很关键，目前尝试15很合适
        {
            PerArea[i] = 0;
            PerArea2[i] = 0;
        }
        else
        {
            double area = contourArea(hull);//凸包面积有存在0的情况
            double Yarea = stats.at<int>(i, cv::CC_STAT_AREA);
            PerArea[i] = Yarea / area;
            PerArea2[i] = Yarea / area;
        }
    }

    int p = n;//怕把n的值给修改了
    for (int i = 1; i < p; i++)
    {
        if (PerArea[i] == 0)
        {
            PerArea.erase(PerArea.begin() + i);//一旦删除向量元素个数删减
            p--;
            i--;//这里修改了一下，这样才能把0全部删除
        }
        cout << "面积比" << PerArea[i] << endl;
    }

    int k = ceil(PerArea.size() / 2);
    nth_element(PerArea.begin(), PerArea.begin() + k, PerArea.end());//此时PerArea中元素排列变化了
    double THRA = PerArea[k];//自适应阈值取中值
    cout << "粘连阈值" << THRA << endl;
    //用阈值筛选粘连组织
    Mat newcopy = Mat::zeros(binaryImg.size(), CV_8UC1);
    for (int i = 0; i < n; i++)
    {
        if (PerArea2[i] > THRA || PerArea2[i] == 0)
        {
            for (int row = 0; row < newcopy.rows; row++)
            {
                for (int col = 0; col < newcopy.cols; col++)
                {
                    if (labels.at<int>(row, col) == i)
                    {
                        newcopy.at<uchar>(row, col) = 0;
                    }
                }
            }
        }
        else
        {
            for (int row = 0; row < newcopy.rows; row++)
            {
                for (int col = 0; col < newcopy.cols; col++)
                {
                    if (labels.at<int>(row, col) == i)
                    {
                        newcopy.at<uchar>(row, col) = 255;//满足条件的设为白255
                    }
                }
            }
        }
    }
    imshow("newcopy2", newcopy);
    //开运算
    Mat element;
    element = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
    Mat dstImage;
    erode(newcopy, dstImage, element);
    dilate(dstImage, dstImage, element);
    imshow("形态学处理后", dstImage);
    //进一步将粘连组织分为等轴粘连、片层粘连和复杂粘连
    Mat labels1;
    Mat stats1;
    Mat centroids1;
    int m = connectedComponentsWithStats(dstImage, labels1, stats1, centroids1);

    Mat labels2;
    Mat stats2;
    Mat centroids2;
    int u = connectedComponentsWithStats(newcopy, labels2, stats2, centroids2);

    vector<int>labelnew(m);
    vector<int>labelnew2(m);
    for (int i = 0; i < m; i++) {
        int row = floor(centroids1.at<double>(i, 0));
        int col = floor(centroids1.at<double>(i, 1));
        int t = labels2.at<int>(col, row);//col和row不能弄错

        double xdistance = ((centroids1.at<double>(i, 0)) - (centroids2.at<double>(t, 0))) * ((centroids1.at<double>(i, 0)) - (centroids2.at<double>(t, 0)));
        double ydistance = ((centroids1.at<double>(i, 1)) - (centroids2.at<double>(t, 1))) * ((centroids1.at<double>(i, 1)) - (centroids2.at<double>(t, 1)));
        double distance = sqrt(xdistance + ydistance);
        //偏差小于MORPH_ELLIPSE, Size(12, 12)尺寸一半即可认为是纯等轴粘连
        if (distance >= 3.0)
        {
            labelnew[i] = t;
            labelnew2[i] = t;
        }
        else
        {
            labelnew[i] = 0;
            labelnew2[i] = t;//如何把i取出来，就用容器
        }
        cout << "有用标记：" << labelnew[i] << endl;
        cout << "中心距离" << distance << endl;
    }

    Mat DZZLImage = Mat::zeros(newcopy.size(), CV_8UC1);
    for (int i = 1; i < m; i++) {
        if (labelnew[i] != labelnew2[i]) {
            for (int row = 0; row < newcopy.rows; row++)
            {
                for (int col = 0; col < newcopy.cols; col++)
                {
                    if (labels2.at<int>(row, col) == labelnew2[i])
                    {
                        DZZLImage.at<uchar>(row, col) = 255;
                    }
                }
            }
        }
    }
    imshow("等轴粘连", DZZLImage);

    //删除容器中等于0的元素
    int j = m;
    for (int i = 1; i < j; i++)
    {
        if (labelnew[i] == 0)
        {
            labelnew.erase(labelnew.begin() + i);//一旦删除向量元素个数删减
            j--;
            i--;//这里修改了一下，这样才能把0全部删除
        }
    }

    Mat ZLImage = Mat::zeros(newcopy.size(), CV_8UC1);//每次都要copy到个新的矩阵为的是不改变旧矩阵，后续容易对比

    for (int i = 0; i < u; i++)
    {
        if (count(labelnew.begin(), labelnew.end(), i))//目前这个i与前面的t同意义，指的是newcopy里的联通阈
        {
            for (int row = 0; row < newcopy.rows; row++)
            {
                for (int col = 0; col < newcopy.cols; col++)
                {
                    if (labels2.at<int>(row, col) == i)
                    {
                        ZLImage.at<uchar>(row, col) = 255;
                    }
                }
            }
        }
    }
    imshow("复杂粘连", ZLImage);
    //进一步区分等轴粘连和除去等轴和复杂粘连的片层粘连·· 

    Mat PCZLImage = Mat::zeros(binaryImg.size(), CV_8UC1);

    for (int i = 0; i < u; i++)
    {
        if (count(labelnew2.begin(), labelnew2.end(), i))
        {
            for (int row = 0; row < newcopy.rows; row++)
            {
                for (int col = 0; col < newcopy.cols; col++)
                {
                    if (labels2.at<int>(row, col) == i)
                    {
                        PCZLImage.at<uchar>(row, col) = 0;
                    }
                }
            }
        }
        else
            for (int row = 0; row < newcopy.rows; row++)
            {
                for (int col = 0; col < newcopy.cols; col++)
                {
                    if (labels2.at<int>(row, col) == i && i != 0)
                    {
                        PCZLImage.at<uchar>(row, col) = 255;
                    }
                }
            }
    }

    imshow("片层粘连", PCZLImage);

    for (int row = 0; row < newcopy.rows; row++)
    {
        for (int col = 0; col < newcopy.cols; col++)
        {
            if (dstImage.at<uchar>(row, col) != 0)
            {
                dstImage.at<uchar>(row, col) = 255;
            }
        }
    }


    Mat NEWImage = Mat::zeros(newcopy.size(), CV_8UC1);
    for (int row = 0; row < newcopy.rows; row++)
    {
        for (int col = 0; col < newcopy.cols; col++)
        {
            if (dstImage.at<uchar>(row, col) != ZLImage.at<uchar>(row, col) && DZZLImage.at<uchar>(row, col) != 255)
            {
                NEWImage.at<uchar>(row, col) = 255;
            }
        }
    }

    imshow("其他情况", NEWImage);

    Mat labels21, stats21, centroids21;
    int nLabels21 = connectedComponentsWithStats(NEWImage, labels21, stats21, centroids21, 4, CV_32S);

    // 创建一个新图像，用于显示结果
    Mat result_new = Mat::zeros(NEWImage.size(), CV_8UC1);

    // 定义面积阈值
    const int areaThreshold = 12;

    for (int i = 1; i < nLabels21; i++) {
        int area_new = stats21.at<int>(i, CC_STAT_AREA);

        // 如果连通域面积大于阈值，则保留该连通域
        if (area_new > areaThreshold) {
            // 提取并保留该连通域
            for (int r = 0; r < labels21.rows; ++r) {
                for (int c = 0; c < labels21.cols; ++c) {
                    if (labels21.at<int>(r, c) == i) {
                        result_new.at<uchar>(r, c) = 255;
                    }
                }
            }
        }
    }
    imshow("result_new", result_new);


    Mat aftersplit = Mat::zeros(result_new.size(), CV_8UC1);
    for (int row = 0; row < result_new.rows; row++)
    {
        for (int col = 0; col < result_new.cols; col++)
        {
            if (result_new.at<uchar>(row, col) == 0 && ZLImage.at<uchar>(row, col) == 255) {
                aftersplit.at<uchar>(row, col) = 255;
            }

        }
    }
    imshow("aftersplit", aftersplit);

    //将待分割图像转变为3通道图像；
    Mat dstImage_new = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int row = 0; row < dstImage.rows; row++) {
        for (int col = 0; col < dstImage.cols; col++) {
            if (dstImage.at<uchar>(row, col) != 0) {
                // 将该像素设置为白色
                dstImage_new.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
            }
        }
    }

    //进行多重腐蚀，寻找注水点；
    Mat labels_new, stats22, centroids22;
    int nLabels = connectedComponentsWithStats(dstImage, labels_new, stats22, centroids22);
    vector<Point2f>zhixin;

    for (int i = 1; i < nLabels; i++)
    {
        Mat mask = Mat::zeros(labels_new.size(), CV_8UC1);
        for (int row = 0; row < labels_new.rows; row++)
        {
            for (int col = 0; col < labels_new.cols; col++)
            {
                if (labels_new.at<int>(row, col) == i)
                {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }

        Mat element = getStructuringElement(2, Size(3, 3));

        int maxNumberOfComponents = 0;
        vector<Point2f> maxComponentsCenters;
        int erosionCount = 0;

        while (true) {
            erode(mask, mask, element); // 应用腐蚀
            erosionCount++;

            // 查找连通域
            Mat labels23, stats23, centroids23;
            int numberOfComponents = connectedComponentsWithStats(mask, labels23, stats23, centroids23);

            if (numberOfComponents <= 1)
            { // 当没有或仅有背景连通域时停止
                break;
            }

            if (numberOfComponents > maxNumberOfComponents)
            {
                maxNumberOfComponents = numberOfComponents;
                maxComponentsCenters.clear();
                for (int i = 1; i < centroids23.rows; i++) {
                    maxComponentsCenters.push_back(Point2f(centroids23.at<double>(i, 0), centroids23.at<double>(i, 1)));
                }
            }

            if (cv::countNonZero(mask) == 0) { // 如果图像完全被腐蚀
                break;
            }
        }

        for (size_t i = 0; i < maxComponentsCenters.size(); i++) {
            cout << "Center " << i + 1 << ": " << maxComponentsCenters[i] << endl;
        }

        zhixin.insert(zhixin.end(), maxComponentsCenters.begin(), maxComponentsCenters.end());

    }

    for (size_t i = 0; i < zhixin.size(); ++i)
    {
        printf("Point %zu: (%.2f, %.2f)\n", i + 1, zhixin[i].x, zhixin[i].y);
    }

    Mat markers = Mat::zeros(dstImage.size(), CV_32SC1);
    for (int i = 0; i < zhixin.size(); i++) {
        int x = static_cast<int>(zhixin.at(i).x);
        int y = static_cast<int>(zhixin.at(i).y);
        markers.at<int>(y, x) = i;
    }

    // 在原图基础上应用分水岭算法
    watershed(dstImage_new, markers);

    Mat result = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= zhixin.size())
            {
                result.at<Vec3b>(i, j) = Vec3b(0, 255, 0); // 使用绿色标记分割区域
            }
        }
    }

    for (const Point2f& pt : zhixin) {
        // 在pt坐标处绘制一个小圆圈代表点
        // 参数依次为：目标图像、中心点、半径、颜色、填充
        circle(result, pt, 5, Scalar(0, 0, 255), FILLED);
    }

    imshow("afterwatershed", result);

    waitKey(0);
    return (0);
}