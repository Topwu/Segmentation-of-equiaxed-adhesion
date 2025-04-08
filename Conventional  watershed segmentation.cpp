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

#define WindowName "阈值选取实例"

const int g_nMaxAlphaValue = 255;//Alpha的最大值
int g_nAlphaValueSlider;//滑动条对应的变量

Mat dstShow, g_dstImage, dstImage;


void on_Trackbar(int, void*)
{
    g_dstImage = Mat::zeros(dstShow.size(), CV_8UC1);
    for (int row = 0; row < dstShow.rows; row++)
    {
        for (int col = 0; col < dstShow.cols; col++)
        {

            if (dstShow.at<uchar>(row, col) >= g_nAlphaValueSlider)
            {
                g_dstImage.at<uchar>(row, col) = 255;
            }
        }
    }

    //显示效果图
    imshow(WindowName, g_dstImage);

    Mat dstImage_new = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int row = 0; row < dstImage.rows; row++) {
        for (int col = 0; col < dstImage.cols; col++) {
            if (dstImage.at<uchar>(row, col) != 0) {
                // 将该像素设置为白色
                dstImage_new.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
            }
        }
    }

    // 寻找连通域
    Mat labels22, stats22, centroids22;
    int nLabels = connectedComponentsWithStats(g_dstImage, labels22, stats22, centroids22, 8, CV_32S);

    // 创建标记图像
    Mat markers = Mat::zeros(g_dstImage.size(), CV_32SC1);
    for (int i = 1; i < nLabels; i++) {
        int x = static_cast<int>(centroids22.at<double>(i, 0));
        int y = static_cast<int>(centroids22.at<double>(i, 1));
        markers.at<int>(y, x) = i;
    }

    // 在原图基础上应用分水岭算法
    watershed(dstImage_new, markers);

    // 创建显示结果的图像
    Mat result = Mat::zeros(g_dstImage.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= nLabels) {
                result.at<Vec3b>(i, j) = Vec3b(0, 255, 0); // 使用绿色标记分割区域
            }
        }
    }

    // 显示原图与结果
    imshow("Original Image", dstImage_new);
    imshow("Watershed Result", result);

}

int main()
{
    Mat img = imread("E:/1.bmp"); // 读取图像
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

    //dstImage为开运算后经连通域标记后的图像，以下转化为二值图像；
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

    //复杂粘连中排除开运算后剩余的等轴部分；
    Mat NEWImage = Mat::zeros(newcopy.size(), CV_8UC1);
    for (int row = 0; row < newcopy.rows; row++)
    {
        for (int col = 0; col < newcopy.cols; col++)
        {
            if (dstImage.at<uchar>(row, col) != ZLImage.at<uchar>(row, col))
            {
                NEWImage.at<uchar>(row, col) = 255;
            }
        }
    }

    imshow("其他情况", NEWImage);

    //距离变换；
    Mat imageThin(dstImage.size(), CV_32FC1);
    distanceTransform(dstImage, imageThin, DIST_L2, 3);
    dstShow = Mat::zeros(dstImage.size(), CV_8UC1);

    for (int row = 0; row < imageThin.rows; row++)
    {
        for (int col = 0; col < imageThin.cols; col++)
        {
            dstShow.at<uchar>(row, col) = imageThin.at<float>(row, col);

        }
    }
    normalize(dstShow, dstShow, 0, 255, NORM_MINMAX);

    imshow("dist_norm", dstShow);

    g_nAlphaValueSlider = 127;

    //创建窗体
    namedWindow(WindowName, 1);

    //在创建的窗体中创建一个滑动条控件
    char TrackbarName[50] = "选取阈值";

    createTrackbar(TrackbarName, WindowName, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar);

    //结果在回调函数中显示
    on_Trackbar(g_nAlphaValueSlider, 0);



    waitKey(0);
    return (0);
}