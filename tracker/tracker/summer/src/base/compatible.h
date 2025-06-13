// Used to achieve compatibility between different versions of libraries

//--------- OpenCV version 3 and 4
#define OPENCV_VERSION_4

#ifdef OPENCV_VERSION_4
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_RGB2GRAY cv::COLOR_RGB2GRAY
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CHAIN_APPROX_NONE cv::CHAIN_APPROX_NONE
#define CV_DIST_L2 cv::DIST_L2
#define CV_RETR_LIST cv::RETR_LIST
#define CV_RETR_EXTERNAL cv::RETR_EXTERNAL
#define CV_SORT_EVERY_ROW cv::SORT_EVERY_ROW
#define CV_SORT_ASCENDING cv::SORT_ASCENDING
#define CV_THRESH_BINARY cv::THRESH_BINARY 

#endif 
