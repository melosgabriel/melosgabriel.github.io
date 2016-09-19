#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

double d;
int L1_slider = 0;
int L1_slider_max;

int L2_slider = 0;
int L2_slider_max;

int d_slider = 0;
int d_slider_max = 100;

int contrast_slider = 0;
int contrast_slider_max = 100;
int previous_contrast = 0;

Mat im,aux,result;

char TrackbarName[50];

double alpha(double x,double l1,double l2,double d1){
	float retorno;
	float k11 = (x-l1)/d1;
	float k22 = (x-l2)/d1;
	retorno = 0.5*(tanh(k11) - tanh(k22));
	return retorno;
}

void juntar(Mat& src1 , Mat& src2){
	for(int i=2;i<src1.rows;i++){
		double alfa = alpha(i,L1_slider,L2_slider,d);
		addWeighted(src1.row(i),alfa, src2.row(i),1-alfa,0.0,result.row(i));
	}
}

void on_trackbar_d(int, void*){
	d = (double) d_slider;
	juntar(im,aux);
	result.convertTo(result, CV_8UC3);
	imshow("TiltShift",result);
}

void on_trackbar_L1(int, void*){
	juntar(im,aux);
	result.convertTo(result, CV_8UC3);
	imshow("TiltShift",result);
}

void on_trackbar_L2(int, void*){
	juntar(im,aux);
	result.convertTo(result, CV_8UC3);
	imshow("TiltShift",result);
}

float media[] = {1,1,1,
			     1,1,1,
				 1,1,1};
Mat mask = Mat(3,3,CV_32F,media),mask1;
     
int main(int argvc, char** argv){
  	VideoCapture video(argv[1]);      //Abrindo Arquivo de entrada
  	VideoWriter output_cap("output.avi", video.get(CV_CAP_PROP_FOURCC), video.get(CV_CAP_PROP_FPS),
               			   cv::Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT))); //Arquivo de saída

  	if(!video.isOpened()) return -1;            //Teste para validar entradas e saídas
  	if(!output_cap.isOpened()) return -1;

  	int ratio, counter(0);
  	cout<<"Qual a razão que você quer no StopMotion?\n"; //Calculando a razão do StopMotion
  	cin>>ratio;

  	bool tempo;
  	cout<<"Manter tempo original? 1 - sim; 0 - Não\n";
  	cin>>tempo;

  	scaleAdd(mask, 1/9.0, Mat::zeros(3,3,CV_32F), mask1);
	mask = mask1;

	video.read(im);
	aux = im.clone();
	im.convertTo(im,CV_32FC3);
  	aux.convertTo(aux,CV_32FC3);
	for(int i=0;i<30;i++){
  		filter2D(aux, aux, im.depth(), mask, Point(1,1), 0);
  	}

  	result = im.clone();
	L1_slider_max = im.rows;
  	L2_slider_max = im.rows;
  	cout<<"width: "<<im.cols<<endl;
  	cout<<"height: "<<im.rows<<endl;

  		
	namedWindow("TiltShift",WINDOW_NORMAL);
	sprintf( TrackbarName, "decaimento x %d", d_slider_max );
  	createTrackbar( TrackbarName, "TiltShift", &d_slider, d_slider_max, on_trackbar_d );
  	on_trackbar_d(d_slider, 0 );
  
  	sprintf( TrackbarName, "Linha Superior x %d", L1_slider_max );
  	createTrackbar( TrackbarName, "TiltShift", &L1_slider, L1_slider_max, on_trackbar_L1 );
  	on_trackbar_L1(L1_slider, 0 );
  
  	sprintf( TrackbarName, "Linha Inferior x %d", L2_slider_max );
  	createTrackbar( TrackbarName, "TiltShift", &L2_slider, L2_slider_max, on_trackbar_L2 );
  	on_trackbar_L2(L2_slider, 0 );
  	waitKey(0);
  	destroyWindow("TiltShift");
  	waitKey(1);

  	while(1){
  		if(!video.read(im)) break;
  		counter++;
	  	if(counter == ratio){
	  		counter = 0;
	  		aux = im.clone();
		  	for(int i=0;i<30;i++){
	  			filter2D(aux, aux, im.depth(), mask, Point(1,1), 0);
	  		}
	  		juntar(im,aux);
	  		output_cap.write(result);
	  	}
	  	if(tempo) output_cap.write(result);
  	}
  	
  	waitKey(0);
  	return 0;
}
