#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

int step_slider = 5;
int step_slider_max = 20;
int jitter_slider = 3;
int jitter_slider_max = 10;
int raio_slider = 3;
int raio_slider_max = 10;

int top_slider = 10;
int top_slider_max = 200;

char TrackbarName[50];

Mat image, border, points;

vector<int> yrange;
vector<int> xrange;

int width, height, r,g,b;
int x, y;

void on_trackbar_canny(int, void*){
  Canny(image, border, top_slider, 3*top_slider);  

  xrange.resize(height/step_slider);
  yrange.resize(width/step_slider);
  
  iota(xrange.begin(), xrange.end(), 0); 
  iota(yrange.begin(), yrange.end(), 0);

  for(uint i=0; i<xrange.size(); i++){
    xrange[i]= xrange[i]*step_slider+step_slider/2;
  }

  for(uint i=0; i<yrange.size(); i++){
    yrange[i]= yrange[i]*step_slider+step_slider/2;
  }

  points = Mat(height, width, CV_8UC3, CV_RGB(255,255,255));

  random_shuffle(xrange.begin(), xrange.end());

  for(auto i : xrange){
    random_shuffle(yrange.begin(), yrange.end());
    for(auto j : yrange){
      if(jitter_slider) x = i+rand()%(2*jitter_slider)-jitter_slider+1;
      else x = i;
      if(jitter_slider) y = j+rand()%(2*jitter_slider)-jitter_slider+1;
      else y = j;
      b = image.at<Vec3b>(x,y)[0];
      g = image.at<Vec3b>(x,y)[1];
      r = image.at<Vec3b>(x,y)[2];
      circle(points,cv::Point(y,x),raio_slider,CV_RGB(r,g,b),-1,CV_AA);
    }
  }

  for(int i = 0;i<height;i++){
    for(int j = 0;j<width;j++){
//      int border_radius = border.at<uchar>(i,j)*(raio_slider-2)/255;
      //border_radius = (border_radius>0 ? border_radius : 1);
      int border_radius = border.at<uchar>(i,j)*(top_slider/40 + 1)/255;
      b = image.at<Vec3b>(i,j)[0];
      g = image.at<Vec3b>(i,j)[1];
      r = image.at<Vec3b>(i,j)[2];
      circle(points,cv::Point(j,i),border_radius,CV_RGB(r,g,b),-1,CV_AA);
    }
  }
  imshow("cannypoints",points);
}

int main(int argc, char** argv){
  
  image= imread(argv[1],CV_LOAD_IMAGE_COLOR);
  
  if(!image.data){
    cout << "nao abriu" << argv[1] << endl;
    cout << argv[0] << " imagem.jpg";
    exit(0);
  }

  namedWindow("cannypoints",WINDOW_NORMAL);
  imshow("cannypoints",image);

  srand(time(0));

  width=image.size().width;
  height=image.size().height;

  sprintf( TrackbarName, "Threshold inferior x %d", top_slider_max );
  createTrackbar( TrackbarName, "cannypoints", &top_slider, top_slider_max, on_trackbar_canny );

  sprintf( TrackbarName, "step x %d",  step_slider_max );
  createTrackbar( TrackbarName, "cannypoints", &step_slider, step_slider_max, on_trackbar_canny );

  sprintf( TrackbarName, "Jitter x %d", jitter_slider_max );
  createTrackbar( TrackbarName, "cannypoints", &jitter_slider, jitter_slider_max, on_trackbar_canny );

  sprintf( TrackbarName, "Raio x %d", top_slider_max );
  createTrackbar( TrackbarName, "cannypoints", &raio_slider, raio_slider_max, on_trackbar_canny );

  waitKey();
  return 0;
}
