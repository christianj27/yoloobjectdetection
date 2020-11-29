VideoCapture interface("C:\Users\Christian Jeremia\Documents\darkflow-master\test_video_day.mp4");
int counter = 0;
Mat frame;
while(cap >> frame){
   if(frame.empty()){
      break;
   }
   counter++
}