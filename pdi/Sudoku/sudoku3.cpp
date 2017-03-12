#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/core/mat.hpp"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>
#include <sstream>
using namespace cv;
using namespace std;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

Mat findGrid(Mat);
int recognize_digit(Mat& im,tesseract::TessBaseAPI& tess);
void remove_borders(Mat &);
bool solve(Mat);

void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255)){ 
	if(line[1]!=0) { 
		float m = -1/tan(line[1]); 
		float c = line[0]/sin(line[1]); 
		cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width+c), rgb);
	} 
	else{ 
		cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb); 
	} 
}

void mergeRelatedLines(vector<Vec2f>*, Mat &);

int main(int argc, char* argv[]){
	Mat sudoku_original,sudoku_gray,sudoku_thresh;
	if(argc == 1) return 0;
	
	sudoku_original = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	cvtColor(sudoku_original,sudoku_gray,CV_BGR2GRAY);

	tesseract::TessBaseAPI tess;
    if (tess.Init("/opt/local/share/tessdata/", "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
	//Thresholding
	GaussianBlur(sudoku_gray,sudoku_thresh,Size(3,3),0);
	//threshold(sudoku_thresh,sudoku_thresh,127,255,THRESH_OTSU+THRESH_BINARY);
	adaptiveThreshold(sudoku_thresh,sudoku_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
	//imshow("thresh",sudoku_thresh); waitKey(0);
	//Applying Dilation to fill holes
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
	//Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
	bitwise_not(sudoku_thresh,sudoku_thresh);
	
	Mat sudoku_grid = sudoku_thresh.clone();
	
	dilate(sudoku_grid, sudoku_grid, kernel);
	sudoku_grid = findGrid(sudoku_grid);

	erode(sudoku_grid,sudoku_grid,kernel);
	//erode(sudoku_thresh,sudoku_thresh,kernel);
	//imshow("grid",sudoku_grid); waitKey(0);
	vector<Vec2f> lines; 
	HoughLines(sudoku_grid, lines, 1, CV_PI/180, 180);

	for(int i=0;i<lines.size();i++) { drawLine(lines[i], sudoku_thresh, CV_RGB(0,0,128)); }
	imshow("window2",sudoku_thresh); waitKey(0);

	mergeRelatedLines(&lines,sudoku_grid);


	Vec2f topEdge = Vec2f(1000,1000); double topYIntercept=100000, topXIntercept=0;
	Vec2f bottomEdge = Vec2f(-1000,-1000); double bottomYIntercept=0, bottomXIntercept=0;
	Vec2f leftEdge = Vec2f(1000,1000); double leftXIntercept=100000, leftYIntercept=0;
	Vec2f rightEdge = Vec2f(-1000,-1000); double rightXIntercept=0, rightYIntercept=0;

	for(int i=0;i<lines.size();i++){         
		Vec2f current = lines[i];         
		float p=current[0];         
		float theta=current[1];         
		if(p==0 && theta==-100) continue;
		double xIntercept, yIntercept;
		xIntercept = p/cos(theta);
		yIntercept = p/(cos(theta)*sin(theta));
		if(theta>CV_PI*80/180 && theta<CV_PI*100/180){
			if(p<topEdge[0]) topEdge = current;
			if(p>bottomEdge[0]) bottomEdge = current;
		}
		else if(theta<CV_PI*10/180 || theta>CV_PI*170/180){
			if(xIntercept>rightXIntercept){
				rightEdge = current;
				rightXIntercept = xIntercept;
			}
			else if(xIntercept<=leftXIntercept){
				leftEdge = current;
				leftXIntercept = xIntercept;
			}
		}
	}

	Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
	int height=sudoku_original.size().height;
	int width=sudoku_original.size().width;
	if(leftEdge[1]!=0){         
		left1.x=0;        
		left1.y=leftEdge[0]/sin(leftEdge[1]);         
		left2.x=width;    
		left2.y=-left2.x/tan(leftEdge[1]) + left1.y;     
	}     
	else{         
		left1.y=0;        
		left1.x=leftEdge[0]/cos(leftEdge[1]);         
		left2.y=height;    
		left2.x=left1.x - height*tan(leftEdge[1]);     
	}     
	if(rightEdge[1]!=0){         
		right1.x=0;        
		right1.y=rightEdge[0]/sin(rightEdge[1]);         
		right2.x=width;    
		right2.y=-right2.x/tan(rightEdge[1]) + right1.y;     
	}     
	else{         
		right1.y=0;        
		right1.x=rightEdge[0]/cos(rightEdge[1]);         
		right2.y=height;   
		right2.x=right1.x - height*tan(rightEdge[1]);     
	}     
	bottom1.x=0;    
	bottom1.y=bottomEdge[0]/sin(bottomEdge[1]);     
	bottom2.x=width;bottom2.y=-bottom2.x/tan(bottomEdge[1]) + bottom1.y;     
	top1.x=0;        
	top1.y=topEdge[0]/sin(topEdge[1]);     
	top2.x=width;    
	top2.y=-top2.x/tan(topEdge[1]) + top1.y;

	drawLine(topEdge, sudoku_gray, CV_RGB(0,0,0));     
	drawLine(bottomEdge, sudoku_gray, CV_RGB(0,0,0));     
	drawLine(leftEdge, sudoku_gray, CV_RGB(0,0,0));     
	drawLine(rightEdge, sudoku_gray, CV_RGB(0,0,0));

	double leftA = left2.y-left1.y;     
	double leftB = left1.x-left2.x;     
	double leftC = leftA*left1.x + leftB*left1.y;     	
	double rightA = right2.y-right1.y;     
	double rightB = right1.x-right2.x;     
	double rightC = rightA*right1.x + rightB*right1.y;     
	double topA = top2.y-top1.y;     
	double topB = top1.x-top2.x;     
	double topC = topA*top1.x + topB*top1.y;    
	double bottomA = bottom2.y-bottom1.y;     
	double bottomB = bottom1.x-bottom2.x;     
	double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;      
	double detTopLeft = leftA*topB - leftB*topA;     
	CvPoint ptTopLeft = cvPoint((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);      
	double detTopRight = rightA*topB - rightB*topA;     
	CvPoint ptTopRight = cvPoint((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);     
	double detBottomRight = rightA*bottomB - rightB*bottomA;     
	CvPoint ptBottomRight = cvPoint((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);
	double detBottomLeft = leftA*bottomB-leftB*bottomA;     
	CvPoint ptBottomLeft = cvPoint((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);

	int maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);     
	int temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);     
	if(temp>maxLength) maxLength = temp;     
	temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x) + (ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);     
	if(temp>maxLength) maxLength = temp;     
	temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);     
	if(temp>maxLength) maxLength = temp;     
	maxLength = sqrt((double)maxLength);

	Point2f src[4], dst[4]; 
	src[0] = ptTopLeft; dst[0] = Point2f(0,0); 
	src[1] = ptTopRight; dst[1] = Point2f(maxLength-1, 0); 
	src[2] = ptBottomRight; dst[2] = Point2f(maxLength-1, maxLength-1); 
	src[3] = ptBottomLeft; dst[3] = Point2f(0, maxLength-1);

	cout<<src[0].x<<','<<src[0].y<<"<=>"<<dst[0].x<<','<<dst[0].y<<endl;
	cout<<src[1].x<<','<<src[1].y<<"<=>"<<dst[1].x<<','<<dst[1].y<<endl;
	cout<<src[2].x<<','<<src[2].y<<"<=>"<<dst[2].x<<','<<dst[2].y<<endl;
	cout<<src[3].x<<','<<src[3].y<<"<=>"<<dst[3].x<<','<<dst[3].y<<endl;


	Mat sudoku_undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
	Mat sudoku_undistorted_gray = Mat(Size(maxLength, maxLength), CV_8UC1);
	cv::warpPerspective(sudoku_original,sudoku_undistorted,cv::getPerspectiveTransform(src,dst),Size(maxLength,maxLength));
	cv::warpPerspective(sudoku_gray,sudoku_undistorted_gray,cv::getPerspectiveTransform(src,dst),Size(maxLength,maxLength));
	
	//erode(sudoku_undistorted,sudoku_undistorted,kernel);
	GaussianBlur(sudoku_undistorted_gray,sudoku_undistorted_gray,Size(5,5),0);
   	imshow("window1",sudoku_gray); waitKey(0);
   	adaptiveThreshold(sudoku_undistorted_gray,sudoku_undistorted_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 2);
   	//dilate(sudoku_undistorted_gray,sudoku_undistorted_gray,kernel);
   	//erode(sudoku_undistorted_gray,sudoku_undistorted_gray,kernel);
   	//erode(sudoku_undistorted_gray,sudoku_undistorted_gray,kernel);
   	floodFill(sudoku_undistorted_gray,Point(10,0),0);
   	imshow("window",sudoku_undistorted_gray); waitKey(0);
    Mat values = Mat::zeros(9,9,CV_8UC1);
    for(int i = 0;i<9;i++){
    	for(int j = 0;j<9;j++){
    		int number;

    		Mat crote2 = sudoku_undistorted_gray.clone(); 
	    	Rect roi(i*sudoku_undistorted.rows/9,j*sudoku_undistorted.cols/9,sudoku_undistorted.rows/9,sudoku_undistorted.cols/9);
    		Mat crote = sudoku_undistorted_gray(roi).clone(); bitwise_not(crote,crote);
    	    remove_borders(crote);
    	    int largest_contour_index=0;
 			int largest_area=0;


 			//if(!largest_contour_index) continue;
 			rectangle(crote2,roi,Scalar(0,255,0),1,8,0);
    		imshow("crote",crote);
    		waitKey(0);
    		number = recognize_digit(crote,tess);
	    	cout<<"Quadrado("<<j<<','<<i<<"): "<<number<<endl;
	        values.at<uchar>(j,i) = number;
    	}
    }
    cout<<"\n\n\n\n";
    for(int i=0;i<9;i++){
    	for(int j=0;j<9;j++){
    		cout<<int(values.at<uchar>(i,j))<<' ';
    	}
    	cout<<endl;
    }
    cout<<"\n\n\n\n";
    Mat values_solved = values.clone();
    solve(values_solved);
    for(int i=0;i<9;i++){
    	for(int j=0;j<9;j++){
    		cout<<int(values_solved.at<uchar>(i,j))<<' ';
    	}
    	cout<<endl;
    }
    for(int i = 0;i<9;i++){
    	for(int j = 0;j<9;j++){
    		if(values.at<uchar>(j,i)==0)
	    	putText(sudoku_undistorted, to_string(values_solved.at<uchar>(j,i)), 
	    		Point(i*sudoku_undistorted.rows/9 + 6,(j+1)*sudoku_undistorted.cols/9 - 6),
	    		FONT_HERSHEY_SIMPLEX, 0.6, Scalar (255,0,0),2);
    	}
    }
	//imshow("crote",sudoku_thresh);
	//waitKey(0);
	//imshow("sudoku",sudoku_original);
	//waitKey(0);
	imshow("sudoku",sudoku_undistorted);
	waitKey(0);
	//bitwise_not(sudoku_undistorted,sudoku_undistorted);
	//imwrite("trainar.png",sudoku_undistorted);
}

Mat findGrid(Mat sudoku_thresh){
	int count=0;
	int max=-1;
	Point maxPt;
	for(int y=0;y<sudoku_thresh.size().height;y++){ 
		for(int x=0;x<sudoku_thresh.size().width;x++){ 
			if(sudoku_thresh.at<uchar>(y,x)>=128){ 
				int area = floodFill(sudoku_thresh, Point(x,y), 64); 
				if(area>max){ 
					maxPt = Point(x,y);
					max = area; 
				} 
			} 
		} 
	}

	floodFill(sudoku_thresh,maxPt,255);
	
	for(int y=0;y<sudoku_thresh.size().height;y++){ 
		for(int x=0;x<sudoku_thresh.size().width;x++){ 
			if(sudoku_thresh.at<uchar>(y,x)==64 && x!=maxPt.x && y!=maxPt.y){ 
				floodFill(sudoku_thresh, Point(x,y), 0); 
			} 
		}
	}
	return sudoku_thresh;
}

void mergeRelatedLines(vector<Vec2f> *lines, Mat &img){
	vector<Vec2f>::iterator current;
	for(current=lines->begin();current!=lines->end();current++){
		if((*current)[0]==0 && (*current)[1]==-100) continue;
		float p1 = (*current)[0];
		float theta1 = (*current)[1];
		Point pt1current, pt2current;
		if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180){
			pt1current.x =0;
			pt1current.y = p1/sin(theta1);
			pt2current.x = img.size().width;
			pt2current.y = -pt2current.x/tan(theta1) + p1/sin(theta1);
		}
		else{
			pt1current.y = 0;
			pt1current.x = p1/cos(theta1);
			pt2current.y = img.size().height;
			pt2current.x = -pt2current.y/tan(theta1) + p1/cos(theta1);
		}
		vector<Vec2f>::iterator pos;
		for(pos=lines->begin();pos!=lines->end();pos++){
			if(*current==*pos) continue;
			if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180){
				float p = (*pos)[0];
				float theta = (*pos)[1];
				Point pt1,pt2;
				if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180){
					pt1.x = 0;
					pt1.y = p/sin(theta);
					pt2.x = img.size().width;
					pt2.y = -pt2.x/tan(theta) + p/sin(theta);
				}
				else{
					pt1.y = 0;
					pt1.x = p/cos(theta);
					pt2.y = img.size().height;
					pt2.x = -pt2.y/tan(theta) + p/cos(theta);
				}
				if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) && 
				   ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64)) {
					(*current)[0] = ((*current)[0]+(*pos)[0])/2;
					(*current)[1] = ((*current)[1]+(*pos)[1])/2;
					(*pos)[0] = 0;
					(*pos)[1] = -100;
				}
			}
		}
	}
}


int recognize_digit(Mat& im,tesseract::TessBaseAPI& tess)
{
    tess.SetImage((uchar*)im.data, im.size().width, im.size().height, im.channels(), (int)im.step1());
    tess.Recognize(0);
    const char* out = tess.GetUTF8Text();
    cout<<"Tesseract: "<<out[0]<<endl;
    if (out)
        if(out[0]=='1' or out[0]=='I' or out[0]=='i' or out[0]=='/' or out[0]=='|' or out[0]=='l' or out[0]=='t')
            return 1;
        else if(out[0]=='2')
            return 2;
        else if(out[0]=='3')
            return 3;
        else if(out[0]=='4')
            return 4;
        else if(out[0]=='5' or out[0]=='S' or out[0]=='s')
            return 5;
        else if(out[0]=='6')
            return 6;
        else if(out[0]=='7')
            return 7;
        else if(out[0]=='8' or out[0]=='B')
            return 8;
        else if(out[0]=='9')
            return 9;
        else
            return 0;
    else
        return 0;
}

/*void remove_borders(Mat & image){
	int limiar(200);
	for(int i=0;i<image.rows;i++){
   	if(image.at<uchar>(i,0) == 0) if(floodFill(image.clone(),Point(0,i),255) < limiar) floodFill(image,Point(0,i),255);
   	if(image.at<uchar>(i,image.cols-1) == 0) if(floodFill(image.clone(),Point(image.cols-1,i),255) <limiar) floodFill(image,Point(image.cols-1,i),255);
   }
   for(int j=0;j<image.cols;j++){
   	if(image.at<uchar>(0,j) == 0) if(floodFill(image.clone(),Point(j,0),255) < limiar) floodFill(image,Point(j,0),255);
   	if(image.at<uchar>(image.rows-1,j) == 0) if(floodFill(image,Point(j,image.rows-1),255) < limiar) floodFill(image,Point(j,image.rows-1),255);
   }
}*/

void remove_borders(Mat & image){
	int limiar;
	/*for(int i=0;i<image.rows;i++){
   		if(image.at<uchar>(i,0) == 0) if(floodFill(image.clone(),Point(0,i),255) < limiar) floodFill(image,Point(0,i),255);
   		if(image.at<uchar>(i,tam/6) == 0) if(floodFill(image.clone(),Point(0,i),255) < limiar) floodFill(image,Point(0,i),255);
   		if(image.at<uchar>(i,image.cols-1) == 0) if(floodFill(image.clone(),Point(image.cols-1,i),255) <limiar) floodFill(image,Point(image.cols-1,i),255);
   		if(image.at<uchar>(i,image.cols-tam/6) == 0) if(floodFill(image.clone(),Point(image.cols-1,i),255) <limiar) floodFill(image,Point(image.cols-1,i),255);
   }
	for(int j=0;j<image.cols;j++){
   		if(image.at<uchar>(0,j) == 0) if(floodFill(image.clone(),Point(j,0),255) < limiar) floodFill(image,Point(j,0),255);
   		if(image.at<uchar>(image.rows-1,j) == 0) if(floodFill(image,Point(j,image.rows-1),255) < limiar) floodFill(image,Point(j,image.rows-1),255);
   }*/
	int tam = (image.rows>image.cols ? image.rows : image.cols);
	limiar = int(tam*tam*0.15);
   	for(int i=0;i<image.rows;i++){
	  	for(int j=0;j<image.cols;j++){
	  		if(i<tam/10 || i>image.rows-tam/10) if(floodFill(image.clone(),Point(i,j),255) < limiar) floodFill(image,Point(i,j),255);
	  		if(j<tam/10 || j>image.cols-tam/10) if(floodFill(image.clone(),Point(i,j),255) < limiar) floodFill(image,Point(i,j),255);
	  		//cout<<"i: "<<i<<"j: "<<j<<endl;
   		}
	}
}

bool used_in_row(Mat img, int num, int row){
	for(int i =0; i<img.cols;i++){
		if(img.at<uchar>(row,i) == num) return true;
	}
	return false;
}

bool used_in_col(Mat img, int num, int col){
	for(int j=0;j<img.rows;j++){
		if(img.at<uchar>(j,col) == num) return true;
	}
	return false;
}

/*0 1 2
  3 4 5
  6 7 8*/
bool used_in_box(Mat img, int num, int start_row, int start_col){
	for(int j=start_col;j<start_col+3;j++){
		for(int i=start_row;i<start_row+3;i++){
			if(img.at<uchar>(i,j) == num) return true;
		}
	}
	return false;
}

bool isSafe(Mat img, int row, int col, int num){
	if(!used_in_row(img,num,row) && !used_in_col(img,num,col) && !used_in_box(img,num,row-row%3,col-col%3))
		return true;
	else return false;
}

bool FindUnassignedLocation(Mat grid, int &row, int &col)
{
    for (row = 0; row < grid.rows; row++)
        for (col = 0; col < grid.cols; col++)
            if (grid.at<uchar>(row,col) == 0)
                return true;
    return false;
}

bool solve(Mat img){
	bool solved = true;
	int i,j;
	if(!FindUnassignedLocation(img,i,j)) return true;
	for (int k=0;k<10;k++){
		if(isSafe(img,i,j,k)){
			img.at<uchar>(i,j) = k;
			if(solve(img)) return true;
			img.at<uchar>(i,j) = 0;
		}
	}
	/*for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if(img.at<uchar>(i,j) == 0){
				for(int k=1;k<10;k++){
					if(isSafe(img,i,j,k)){
						img.at<uchar>(i,j) = k;
						if(solve(img)) return true;
						img.at<uchar>(i,j) = 0;
					}
				}
			}
		}
	}*/
	//out = img.clone(); 
	return false;
}


/*bool solve(Mat grid)
{
    int row, col;
 
    // If there is no unassigned location, we are done
    if (!FindUnassignedLocation(grid, row, col))
       return true; // success!
 
    // consider digits 1 to 9
    for (int num = 1; num <= 9; num++)
    {
        // if looks promising
        if (isSafe(grid, row, col, num))
        {
            // make tentative assignment
            grid.at<uchar>(row,col) = num;
 
            // return, if success, yay!
            if (solve(grid))
                return true;
 
            // failure, unmake & try again
            grid.at<uchar>(row,col) = 0;
        }
    }
    return false; // this triggers backtracking
}*/
 

