#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
  Mat image;
  int width, height;
  VideoCapture cap;
  vector<Mat> planes;
  Mat histR_antigo, histR_novo;
  int nbins = 64;
  float range[] = {0, 256};
  const float *histrange = { range };
  bool uniform = true;
  bool acummulate = false;
  double D;
  int counter(0);
  cap.open(0);
  char key;

  if(!cap.isOpened()){
    cout << "cameras indisponiveis";
    return -1;
  }

  width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  cout << "largura = " << width << endl;
  cout << "altura  = " << height << endl;
  
  cap >> image;
  split (image, planes);
  calcHist(&planes[0], 1, 0, Mat(), histR_antigo, 1, &nbins, &histrange, uniform, acummulate);

  while(1){
    cap >> image;
    split (image, planes);
    calcHist(&planes[0], 1, 0, Mat(), histR_novo, 1, &nbins, &histrange, uniform, acummulate);
    D = norm(histR_novo,histR_antigo,NORM_L2); 
    
    if(D>5000){
    	calcHist(&planes[0], 1, 0, Mat(), histR_antigo, 1, &nbins, &histrange, uniform, acummulate);
    	cout<<"ALARME ATIVADO\n";
    }
    counter++;
    if(counter>18000){
    	calcHist(&planes[0], 1, 0, Mat(), histR_antigo, 1, &nbins, &histrange, uniform, acummulate);
    }

    //cout<<D<<endl;
    
    imshow("image", image);
    key = waitKey(30);
    if(key==105) imwrite("in.png",image);
    else if(key==111) imwrite("out.png",image);
    else if(key>=0) break;
  }
  return 0;
}