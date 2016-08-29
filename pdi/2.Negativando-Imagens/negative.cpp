#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
	Mat im = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(!im.data) cout<<"Nao abriu a imagem!\n";

	CvPoint cp1, cp2;
	namedWindow("janela",WINDOW_AUTOSIZE);
	cout<<"Tamanho da imagem: "<<im.cols<<'x'<<im.rows<<endl;
	cout<<"Diga dois pontos: \nX1: ";
	cin>>cp1.y; cout<<"Y1: "; cin>>cp1.x;
	cout<<"X2: "; cin>>cp2.y; cout<<"Y2: "; cin>>cp2.x;
	int maiorx,menorx,maiory,menory;
	if(cp1.x>=cp2.x){
		maiorx = cp1.x;
		menorx = cp2.x;
	}
	else{
		maiorx = cp2.x;
		menorx = cp1.x;	
	} 
	if(cp1.y>=cp2.y){
		maiory = cp1.y;
		menory = cp2.y;	
	}
	else{
		maiory = cp2.y;
		menory = cp1.y;	
	}
	for(int i=menorx;i<maiorx;i++){
		for(int j=menory;j<maiory;j++){
			im.at<Vec3b>(i,j)[0] = 255 - im.at<Vec3b>(i,j)[0];
			im.at<Vec3b>(i,j)[1] = 255 - im.at<Vec3b>(i,j)[1];
			im.at<Vec3b>(i,j)[2] = 255 - im.at<Vec3b>(i,j)[2];
		}
	}
	imshow("janela",im);
	imwrite("saida.jpg",im);
	waitKey();

}