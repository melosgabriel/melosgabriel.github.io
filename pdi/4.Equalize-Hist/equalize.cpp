#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  Mat image, out;
  int width, height;
  bool color;
  vector<Mat> channels;
  cout<<"Voce quer uma imagem colorida?\n";
  cin>>color;
  char key;
  
  if(argc>1){
    image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    if(!image.data){
      cout<<"Nao abriu a imagem!\n";
      return -1;
    }

    width  = image.cols;
    height = image.rows;

    cout << "largura = " << width << endl;
    cout << "altura  = " << height << endl;

    if(color){
      cvtColor(image,out,CV_RGB2HSV);
      split(out,channels);
      equalizeHist(channels[2], channels[2]);
      merge(channels,out);
      cvtColor(out,out,CV_HSV2RGB);
    }
    else{
      cvtColor(image,image,CV_RGB2GRAY);
      equalizeHist(image, out );
    }
    namedWindow("source");
    namedWindow("output");
    imshow("souce", image);
    imshow("output",out);
    imwrite("output.png",out);
    waitKey();
      
  } 
  else{
    VideoCapture cap;
    cap.open(0);

    if(!cap.isOpened()){
      cout << "cameras indisponiveis";
      return -1;
    }

    width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    cout << "largura = " << width << endl;
    cout << "altura  = " << height << endl;

    cout << "Pressione p para printar e ESC para sair!\n";
    while(1){
      cap >> image;
      flip(image,image,1);

      if(color){
        cvtColor(image,out,CV_BGR2HSV);
        split(out,channels);
        equalizeHist(channels[2], channels[2]);
        merge(channels,out);
        cvtColor(out,out,CV_HSV2BGR);
      }
      else{
        cvtColor(image,image,CV_BGR2GRAY);
        equalizeHist(image, out );
      }
      namedWindow("source");
      namedWindow("output");
      imshow("souce", image);
      imshow("output",out);
      key = waitKey(30);
      if(key==27) break;
      if(key==112){
        imwrite("saida.png",out);
        imwrite("entrada.png",image);
      }
    }  
  }
  
  return 0;
}

