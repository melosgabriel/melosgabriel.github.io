#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>

using namespace cv;
using namespace std;

struct coordenada{
    int x,y;
    inline coordenada() : x(0),y(0) {}
};

void sfill(int x,int y, unsigned char cor, Mat &image){
    coordenada C; coordenada D; stack <coordenada> pilha; 
    C.x = x; C.y = y; pilha.push(C);
    unsigned char cor2 = image.at<uchar>(x,y); 
    //unsigned char cor2 = 0;
    while(!pilha.empty()){
        x = pilha.top().x; y = pilha.top().y;
        pilha.pop();
        if(x<image.size().height && y<image.size().width){
	        if(image.at<uchar>(x,y) != cor){
			    if(image.at<uchar>(x+1,y) == cor2){
			        C.x = x+1; C.y = y;
			        pilha.push(C);
			    }	
			    if(image.at<uchar>(x,y+1) == cor2){
			        D.x = x; D.y = y+1;
			            pilha.push(D);
			    }	
		        if(x!=0){
		        	if(image.at<uchar>(x-1,y) == cor2){
		            	C.x = x-1; C.y = y;
		            	pilha.push(C);
		        	}	
		        }
		        if(y!=0){
			        if(image.at<uchar>(x,y-1) == cor2){
			            D.x = x; D.y = y-1;
			            pilha.push(D);
			        }	
		        }
		        
		        image.at<uchar>(x,y) = cor;
	    	}
    	}
    }

}

int main(int argc, char *argv[]){
   Mat image; CvPoint p;
   image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
   int mLinhas = image.rows; int nColunas = image.cols;
   
   int counter = 32, nBubbles[4] = {0,0,0,0};

   //Remove bubbles that touch the borders
   for(int i=0;i<mLinhas;i++){
   	if(image.at<uchar>(i,0) == 255) sfill(i,0,0,image);
   	if(image.at<uchar>(i,nColunas-1) == 255) sfill(i,nColunas-1,0,image);
   }
   for(int j=0;j<nColunas;j++){
   	if(image.at<uchar>(0,j) == 255) sfill(0,j,0,image);
   	if(image.at<uchar>(mLinhas-1,j) == 255) sfill(mLinhas-1,j,0,image);
   }
   //Search for bubbles
   for(int i = 0;i<mLinhas;i++){
   	for(int j = 0;j<nColunas;j++){
   		if(image.at<uchar>(i,j) == 255){
   			sfill(i,j,counter,image);
   			counter+=5;
   		}
   	}
   }
   imshow("image",image);
   waitKey();
   
   //Counting Bubbles
   image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
   //Remove bubbles that touch the borders
   for(int i=0;i<mLinhas;i++){
   	if(image.at<uchar>(i,0) == 255) sfill(i,0,0,image);
   	if(image.at<uchar>(i,nColunas-1) == 255) sfill(i,nColunas-1,0,image);
   }
   for(int j=0;j<nColunas;j++){
   	if(image.at<uchar>(0,j) == 255) sfill(0,j,0,image);
   	if(image.at<uchar>(mLinhas-1,j) == 255) sfill(mLinhas-1,j,0,image);
   }
   //Filling background
   sfill(0,0,254,image);
   //finding bubbles and labeling them as 1
   for(int i = 0;i<mLinhas;i++){
   	for(int j = 0;j<nColunas;j++){
   		if(image.at<uchar>(i,j) == 255){
   			sfill(i,j,1,image);
   		}
   	}
   }
   //Finding bubbles with holes and labeling them 1+number of holes
   for(int i = 0;i<mLinhas;i++){
   	for(int j = 0;j<nColunas;j++){
   		if(image.at<uchar>(i,j) == 0 && image.at<uchar>(i,j-1) != 0 && image.at<uchar>(i,j-1) != 254){
   			sfill(i,j-1,image.at<uchar>(i,j-1)+1,image);
   			sfill(i,j,254,image);
   		}
   	}
   }
   //Counting bubbles
   for(int i = 0;i<mLinhas;i++){
   	for(int j = 0;j<nColunas;j++){
   		if(image.at<uchar>(i,j) != 254){
   			nBubbles[(int)image.at<uchar>(i,j)-1]++;
   			sfill(i,j,254,image);
   		}
   	}
   }
   for(int i=0;i<4;i++){
   	cout<<"A imagem tem "<<nBubbles[i] <<" bolhas com "<<i<<" buracos\n";
   }
   return 0;
}
