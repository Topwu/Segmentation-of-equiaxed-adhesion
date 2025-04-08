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

#define WindowName "��ֵѡȡʵ��"

const int g_nMaxAlphaValue = 255;//Alpha�����ֵ
int g_nAlphaValueSlider;//��������Ӧ�ı���

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

    //��ʾЧ��ͼ
    imshow(WindowName, g_dstImage);

    Mat dstImage_new = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int row = 0; row < dstImage.rows; row++) {
        for (int col = 0; col < dstImage.cols; col++) {
            if (dstImage.at<uchar>(row, col) != 0) {
                // ������������Ϊ��ɫ
                dstImage_new.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
            }
        }
    }

    // Ѱ����ͨ��
    Mat labels22, stats22, centroids22;
    int nLabels = connectedComponentsWithStats(g_dstImage, labels22, stats22, centroids22, 8, CV_32S);

    // �������ͼ��
    Mat markers = Mat::zeros(g_dstImage.size(), CV_32SC1);
    for (int i = 1; i < nLabels; i++) {
        int x = static_cast<int>(centroids22.at<double>(i, 0));
        int y = static_cast<int>(centroids22.at<double>(i, 1));
        markers.at<int>(y, x) = i;
    }

    // ��ԭͼ������Ӧ�÷�ˮ���㷨
    watershed(dstImage_new, markers);

    // ������ʾ�����ͼ��
    Mat result = Mat::zeros(g_dstImage.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= nLabels) {
                result.at<Vec3b>(i, j) = Vec3b(0, 255, 0); // ʹ����ɫ��Ƿָ�����
            }
        }
    }

    // ��ʾԭͼ����
    imshow("Original Image", dstImage_new);
    imshow("Watershed Result", result);

}

int main()
{
    Mat img = imread("E:/1.bmp"); // ��ȡͼ��
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

    //dstImageΪ���������ͨ���Ǻ��ͼ������ת��Ϊ��ֵͼ��
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

    //����ճ�����ų��������ʣ��ĵ��Ჿ�֣�
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

    imshow("�������", NEWImage);

    //����任��
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

    //��������
    namedWindow(WindowName, 1);

    //�ڴ����Ĵ����д���һ���������ؼ�
    char TrackbarName[50] = "ѡȡ��ֵ";

    createTrackbar(TrackbarName, WindowName, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar);

    //����ڻص���������ʾ
    on_Trackbar(g_nAlphaValueSlider, 0);



    waitKey(0);
    return (0);
}