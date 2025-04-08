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
    Mat img = imread("E:\\1.bmp"); // ��ȡͼ��
    Mat scrGray, scrmedian;
    cvtColor(img, scrGray, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ��
    medianBlur(scrGray, scrmedian, 3);//��ֵ�˲�
    // ��ֵ������
    Mat binaryImg;
    threshold(scrmedian, binaryImg, 127, 255, THRESH_BINARY);
    // ��ͨ�����
    Mat labels;
    Mat stats;
    Mat centroids;
    int n = connectedComponentsWithStats(binaryImg, labels, stats, centroids);
    // ����ÿ����ͨ���͹��
    vector<double>PerArea(n, 0);//�����ע�����������
    vector<double>PerArea2(n, 0);
    //Ҫ��Ӻ�����ͨ�������С�ģ���������Ҳ��Ū��ȥ��
    for (int i = 1; i < n; i++)
    {
        // ��ȡ��ǰ��ͨ�����������
        vector<Point> points;
        for (int row = stats.at<int>(i, cv::CC_STAT_TOP); row < stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT); row++)
        {
            for (int col = stats.at<int>(i, cv::CC_STAT_LEFT); col < stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH); col++)
            {
                if (labels.at<int>(row, col) == i)
                {
                    points.push_back(Point(col, row));//Ϊʲôcol��row��˳���ˣ�
                }
            }
        }
        // ���㵱ǰ��ͨ���͹��
        vector<Point> hull;
        convexHull(points, hull);
        if (points.size() < 20)//15���ֵ�ܹؼ���Ŀǰ����15�ܺ���
        {
            PerArea[i] = 0;
            PerArea2[i] = 0;
        }
        else
        {
            double area = contourArea(hull);//͹������д���0�����
            double Yarea = stats.at<int>(i, cv::CC_STAT_AREA);
            PerArea[i] = Yarea / area;
            PerArea2[i] = Yarea / area;
        }
    }

    int p = n;//�°�n��ֵ���޸���
    for (int i = 1; i < p; i++)
    {
        if (PerArea[i] == 0)
        {
            PerArea.erase(PerArea.begin() + i);//һ��ɾ������Ԫ�ظ���ɾ��
            p--;
            i--;//�����޸���һ�£��������ܰ�0ȫ��ɾ��
        }
        cout << "�����" << PerArea[i] << endl;
    }

    int k = ceil(PerArea.size() / 2);
    nth_element(PerArea.begin(), PerArea.begin() + k, PerArea.end());//��ʱPerArea��Ԫ�����б仯��
    double THRA = PerArea[k];//����Ӧ��ֵȡ��ֵ
    cout << "ճ����ֵ" << THRA << endl;
    //����ֵɸѡճ����֯
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
                        newcopy.at<uchar>(row, col) = 255;//������������Ϊ��255
                    }
                }
            }
        }
    }
    imshow("newcopy2", newcopy);
    //������
    Mat element;
    element = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
    Mat dstImage;
    erode(newcopy, dstImage, element);
    dilate(dstImage, dstImage, element);
    imshow("��̬ѧ�����", dstImage);
    //��һ����ճ����֯��Ϊ����ճ����Ƭ��ճ���͸���ճ��
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
        int t = labels2.at<int>(col, row);//col��row����Ū��

        double xdistance = ((centroids1.at<double>(i, 0)) - (centroids2.at<double>(t, 0))) * ((centroids1.at<double>(i, 0)) - (centroids2.at<double>(t, 0)));
        double ydistance = ((centroids1.at<double>(i, 1)) - (centroids2.at<double>(t, 1))) * ((centroids1.at<double>(i, 1)) - (centroids2.at<double>(t, 1)));
        double distance = sqrt(xdistance + ydistance);
        //ƫ��С��MORPH_ELLIPSE, Size(12, 12)�ߴ�һ�뼴����Ϊ�Ǵ�����ճ��
        if (distance >= 3.0)
        {
            labelnew[i] = t;
            labelnew2[i] = t;
        }
        else
        {
            labelnew[i] = 0;
            labelnew2[i] = t;//��ΰ�iȡ��������������
        }
        cout << "���ñ�ǣ�" << labelnew[i] << endl;
        cout << "���ľ���" << distance << endl;
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
    imshow("����ճ��", DZZLImage);

    //ɾ�������е���0��Ԫ��
    int j = m;
    for (int i = 1; i < j; i++)
    {
        if (labelnew[i] == 0)
        {
            labelnew.erase(labelnew.begin() + i);//һ��ɾ������Ԫ�ظ���ɾ��
            j--;
            i--;//�����޸���һ�£��������ܰ�0ȫ��ɾ��
        }
    }

    Mat ZLImage = Mat::zeros(newcopy.size(), CV_8UC1);//ÿ�ζ�Ҫcopy�����µľ���Ϊ���ǲ��ı�ɾ��󣬺������׶Ա�

    for (int i = 0; i < u; i++)
    {
        if (count(labelnew.begin(), labelnew.end(), i))//Ŀǰ���i��ǰ���tͬ���壬ָ����newcopy�����ͨ��
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
    imshow("����ճ��", ZLImage);
    //��һ�����ֵ���ճ���ͳ�ȥ����͸���ճ����Ƭ��ճ������ 

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

    imshow("Ƭ��ճ��", PCZLImage);

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

    imshow("�������", NEWImage);

    Mat labels21, stats21, centroids21;
    int nLabels21 = connectedComponentsWithStats(NEWImage, labels21, stats21, centroids21, 4, CV_32S);

    // ����һ����ͼ��������ʾ���
    Mat result_new = Mat::zeros(NEWImage.size(), CV_8UC1);

    // ���������ֵ
    const int areaThreshold = 12;

    for (int i = 1; i < nLabels21; i++) {
        int area_new = stats21.at<int>(i, CC_STAT_AREA);

        // �����ͨ�����������ֵ����������ͨ��
        if (area_new > areaThreshold) {
            // ��ȡ����������ͨ��
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

    //�����ָ�ͼ��ת��Ϊ3ͨ��ͼ��
    Mat dstImage_new = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int row = 0; row < dstImage.rows; row++) {
        for (int col = 0; col < dstImage.cols; col++) {
            if (dstImage.at<uchar>(row, col) != 0) {
                // ������������Ϊ��ɫ
                dstImage_new.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
            }
        }
    }

    //���ж��ظ�ʴ��Ѱ��עˮ�㣻
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
            erode(mask, mask, element); // Ӧ�ø�ʴ
            erosionCount++;

            // ������ͨ��
            Mat labels23, stats23, centroids23;
            int numberOfComponents = connectedComponentsWithStats(mask, labels23, stats23, centroids23);

            if (numberOfComponents <= 1)
            { // ��û�л���б�����ͨ��ʱֹͣ
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

            if (cv::countNonZero(mask) == 0) { // ���ͼ����ȫ����ʴ
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

    // ��ԭͼ������Ӧ�÷�ˮ���㷨
    watershed(dstImage_new, markers);

    Mat result = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= zhixin.size())
            {
                result.at<Vec3b>(i, j) = Vec3b(0, 255, 0); // ʹ����ɫ��Ƿָ�����
            }
        }
    }

    for (const Point2f& pt : zhixin) {
        // ��pt���괦����һ��СԲȦ�����
        // ��������Ϊ��Ŀ��ͼ�����ĵ㡢�뾶����ɫ�����
        circle(result, pt, 5, Scalar(0, 0, 255), FILLED);
    }

    imshow("afterwatershed", result);

    waitKey(0);
    return (0);
}