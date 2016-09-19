#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc,char* argv[]){
  // Load input video
  int counter = 0;
  cv::VideoCapture input_cap(argv[1]);
  if (!input_cap.isOpened())
  {
          std::cout << "!!! Input video could not be opened" << std::endl;
          return -1;
  }

  // Setup output video
  cv::VideoWriter output_cap(argv[2], 
                 input_cap.get(CV_CAP_PROP_FOURCC),
                 input_cap.get(CV_CAP_PROP_FPS),
                 cv::Size(input_cap.get(CV_CAP_PROP_FRAME_WIDTH),
                 input_cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

  if (!output_cap.isOpened())
  {
          std::cout << "!!! Output video could not be opened" << std::endl;
          return -1;
  }


  // Loop to read from input and write to output
  cv::Mat frame;

  while (true)
  {       
      if (!input_cap.read(frame))             
          break;
      if ((counter >= 10*input_cap.get(CV_CAP_PROP_FPS) && counter <= 20*input_cap.get(CV_CAP_PROP_FPS)) ||
          (counter >= 35*input_cap.get(CV_CAP_PROP_FPS) && counter <= 40*input_cap.get(CV_CAP_PROP_FPS))){
        output_cap.write(frame);
        //counter =0;
      }
      counter++;
  }

  input_cap.release();
  output_cap.release();
}