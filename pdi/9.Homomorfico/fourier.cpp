#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace cv;
using namespace std;

void deslocaDFT(Mat& image ){
  Mat tmp, A, B, C, D;
  image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
  int cx = image.cols/2;
  int cy = image.rows/2;
  A = image(Rect(0, 0, cx, cy));
  B = image(Rect(cx, 0, cx, cy));
  C = image(Rect(0, cy, cx, cy));
  D = image(Rect(cx, cy, cx, cy));
  A.copyTo(tmp);  D.copyTo(A);  tmp.copyTo(D);
  C.copyTo(tmp);  B.copyTo(C);  tmp.copyTo(B);
}



int main(int argc, char* argv[]){
  Mat im, imFiltered, filter,padded;
  int dft_M, dft_N;
  im = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  imshow("original",im);
  if(!im.data){
    cout<<"Nao abriu a imagem!\n";
    return -1;
  }

  dft_M = getOptimalDFTSize(im.rows);
  dft_N = getOptimalDFTSize(im.cols);
  copyMakeBorder(im, padded, 0, dft_M - im.rows, 0, dft_N - im.cols, BORDER_CONSTANT, Scalar::all(0));
  imFiltered = padded.clone();
  cout<<"original: "<<im.rows<<'x'<<im.cols<<endl;
  cout<<"padded: "<<padded.rows<<'x'<<padded.cols<<endl;

  vector<Mat> planos;
  Mat zeros = Mat_<float>::zeros(padded.size());
  Mat realInput = Mat_<float>(padded);
  Mat complex; 
  planos.push_back(realInput);
  planos.push_back(zeros);
  merge(planos, complex);

  dft(complex, complex);
  planos.clear();
  split(complex,planos);
  magnitude(planos[0],planos[1],planos[0]);
  planos[0]+=Scalar::all(1);
  log(planos[0],planos[0]);
  deslocaDFT(planos[0]);
  normalize(planos[0],planos[0],0,1,CV_MINMAX);

  imshow("dft",planos[0]);
  if (waitKey(0) == 27){
    imwrite("espectro.png",planos[0]);
  }
  return 0;
}

