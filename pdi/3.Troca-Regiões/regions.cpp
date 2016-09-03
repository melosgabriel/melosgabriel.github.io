#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

int main(int argc,char* argv[]){
	Mat im= imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(!im.data){
		cout<<"Nao abriu a imagem!\n";
		return -1;
	}
	Mat result = im.clone();
	
	Rect regions[4] = {Rect(0,0,im.cols/2,im.rows/2),Rect(im.cols/2,0,im.cols/2,im.rows/2),
		Rect(0,im.rows/2,im.cols/2,im.rows/2),Rect(im.cols/2,im.rows/2,im.cols/2,im.rows/2)};
	
	int vetOptions[24][4] = { {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
							  {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
							  {2, 1, 3, 0}, {2, 1, 0, 3}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 3, 1, 0}, {2, 3, 0, 1},
							  {3, 2, 1, 0}, {3, 2, 0, 1}, {3, 1, 2, 0}, {3, 1, 0, 2}, {3, 0, 1, 2}, {3, 0, 2, 1}};

	int option; namedWindow("janela");
	do{
		cout<<"Escolha uma opção:\n1. Trocar aleatoriamente\n2. Escolher ordem\n3. Mostrar\n-- Terminar (-1) \n";
		cin>>option;
		if(option==1){
			srand(time(NULL));
			int order = rand() % 24;
			for(int i = 0; i < 4; i++ ){
				switch (vetOptions[order][i]){
					case 0:
					im(regions[i]).copyTo(result(regions[0]));
					break;
					case 1:
					im(regions[i]).copyTo(result(regions[1]));
					break;
					case 2:
					im(regions[i]).copyTo(result(regions[2]));
					break;
					case 3:
					im(regions[i]).copyTo(result(regions[3]));
					break;

				}
			}
		}
		if(option==2){
			int order[4];
			cout<<"1 2\n3 4\n\nRegiao 1: ";
			cin>>order[0];
			cout<<"Regiao 2: ";
			cin>>order[1];
			cout<<"Regiao 3: ";
			cin>>order[2];
			cout<<"Regiao 4: ";
			cin>>order[3];
			for(int i = 0; i < 4; i++ ){
				im(regions[order[i]-1]).copyTo(result(regions[i]));
			}		
		}
		if(option==3){
			imshow("janela",result);
			waitKey();
		}
	}while(option!=-1);
}