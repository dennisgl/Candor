
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <cstdlib>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, Mat cq );
void extractSkin( Mat &input, Mat &output);
void voronoiGen( Mat &dist, Mat &dist8u, Mat &labels );
void getFgBgMask( Mat &fg, Mat &bg, Mat &bgfgMask );


/** Global variables */
String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
//String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/semi_palm.xml";
String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);

int th_cr_range[2] = {135,180};
int th_cb_range[2] = {85,125};

int cr_range[2] = {135,180};
int cb_range[2] = {85,125};


int cr_min_std = 10;
int cr_max_std = 25;
int cb_min_std = 25;
int cb_max_std = 25;





int main(int argc, char *argv[])
{
    
	int i,j,k;
    
	//initModule_video();
    //setUseOptimized(true);
	//setNumThreads(8);
    
    
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
	
    Mat edges;
    
    
	
	int color_dist[255*3][3];
	int cd_count=0;
    
	for ( i=255;i>0;i--,cd_count++ ) {
		color_dist[cd_count][2] = 0;
		color_dist[cd_count][1] = 255;
		color_dist[cd_count][0] = i;
	}
    
	for ( i=0;i<255;i++,cd_count++ ) {
		color_dist[cd_count][2] = i;
		color_dist[cd_count][1] = 255;
		color_dist[cd_count][0] = 0;
	}
    
	for ( i=255;i>0;i--,cd_count++ ) {
		color_dist[cd_count][2] = 255;
		color_dist[cd_count][1] = i;
		color_dist[cd_count][0] = 0;
	}
    
    // Get background
    /*
     Mat bg;
     for(;;) {
     cap >> bg ;
     imshow("bg", bg);
     if ( waitKey(-1) == 'g' ) {
     destroyWindow("bg");
     break;
     }
     }
     */
    
    
    
	namedWindow("Original",1);
    /*
	cvCreateTrackbar("cr_min", "Original", &cr_range[0], 255 );
	cvCreateTrackbar("cr_max", "Original", &cr_range[1], 255 );
	cvCreateTrackbar("cb_min", "Original", &cb_range[0], 255 );
	cvCreateTrackbar("cb_max", "Original", &cb_range[1], 255 );
    */
    
    cvCreateTrackbar("0: cr_min_std", "Original", &cr_min_std, 200 );
    cvCreateTrackbar("1: cr_max_std", "Original", &cr_max_std, 200 );
	cvCreateTrackbar("2: cb_min_std", "Original", &cb_min_std, 200 );
    cvCreateTrackbar("3: cb_max_std", "Original", &cb_max_std, 200 );
    
	Mat inputf, frame, fgmask;
	BackgroundSubtractorMOG2 fgbg;
    
    fgbg.set("nmixtures", 3);
    fgbg.set("detectShadows", false);
    
    int ndiv = 25 ; // color quantization
    cvCreateTrackbar("4: ndiv", "Original", &ndiv, 100 );
    
    namedWindow("frame",1);
    int ar=0, ag=0, ab=0;
    cvCreateTrackbar("0: ar", "frame", &ar, 255 );
    cvCreateTrackbar("1: ag", "frame", &ag, 255 );
    cvCreateTrackbar("2: ab", "frame", &ab, 255 );
    
    
    double lastMaxArea = -1;
    bool isFirstFrame = true;
    
    for(;;)
    {
        try{
            cap >> inputf; // get a new frame from camera
            cv::resize(inputf, frame, Size(), 0.618, 0.618);
            flip(frame,frame,1);
            
            
            frame += Scalar( ab, ag, ar );
            
            imshow("frame", frame);
            
            //while(waitKey(30) != 27);
            
            // begin color quantization
            
            Mat cq_a, cq_b;
            
            frame.copyTo(cq_a);
            
            for ( int i = 0 ; i < 1 ; i++ ) {
                double x1,x2;
                cv::minMaxLoc(cq_a,&x1,&x2,NULL,NULL);
                int div=x2/ndiv;
                cq_a.convertTo( cq_b, cq_a.type(), 1.0/div, 0);
                cq_b.convertTo( cq_b, cq_b.type(), div, 0);
                //imshow("cq output image",cq_b);
             
            }
            
            //pyrMeanShiftFiltering(cq_a, cq_b, 20, 20, 1, TermCriteria( TermCriteria::MAX_ITER+TermCriteria::EPS,5,3));
            
            int input_x = frame.cols;
            int input_y = frame.rows;
            rectangle( cq_b, cvPoint( 0, 0), cvPoint( input_x, input_y), Scalar( 255, 255, 0 ), 5, 8, 0 );
            
            
            imshow("cq output image",cq_b);
            //cv::waitKey(0);
            // end color quantization
            
            
            
            detectAndDisplay(frame,cq_b);
            //if ( frame.empty() ) continue;
            
            GaussianBlur(frame, frame, Size(7,7), 1.5, 1.5);
            
            //fgbg(frame, fgmask);
            //Mat bg_sub;
            //frame.copyTo(bg_sub,fgmask);
            
            
            
            extractSkin(frame, edges ) ;
            //dilate(edges,edges,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
            GaussianBlur(edges, edges, Size(5,5), 1.5, 1.5);

            
            
            imshow("Original", edges);
            
            //rectangle(edges, Point(0,0), Point(edges.size().width,edges.size().height), Scalar( 0, 0, 0 ), 10, 8, 0 );
            //detectAndDisplay( frame );
            
            Mat edgesOri = Mat(edges);
            cvtColor( edges, edges, CV_BGR2GRAY );
            
			
            edges = edges > 0;
            
            //imshow("Threshold", edges);

            
            
            //skel begin
            morphologyEx(edges, edges, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
            
            Mat skimg = edges;
            int skElemSize = skimg.rows*skimg.cols*skimg.channels();
            
            Mat element = getStructuringElement(MORPH_CROSS,Size(3,3));
            bool done = false;
            //Mat skel(skimg.size(), CV_8UC1);
            Mat skel = cv::Mat::zeros(skimg.size().height, skimg.size().width, CV_8UC1);
            double limiter_factor = 1;
            int limiter = skElemSize*limiter_factor;
            
            int zeros;
            while( !done ) {
                Mat m_erode;
                erode(skimg,m_erode,element);
                Mat m_temp;
                dilate(m_erode,m_temp,element);
                subtract(skimg,m_temp,m_temp);
                //imshow("subtract", m_temp);
                bitwise_or(skel,m_temp,skel);
                //imshow("Skel", skel);
                skimg = m_erode;
                //imshow("Ori", skimg);
                
                zeros = skElemSize - countNonZero(skimg);
                //cout << zeros << endl;d
                //while(waitKey(30) < 0) ;
                
                if (zeros>=limiter) done = true;
            }
            
            imshow("Skel", skel);
             
             // skel end
             
            
            // DT begin
            
            Mat dtimg ;
            Mat labels ;
            
            distanceTransform(edges, dtimg, labels, CV_DIST_L2, 3);
            
            
            Mat skel32;
            skel.convertTo(skel32,CV_32F);
            //cout << ((dtimg.type() == skel32.type())? "Type Matched":"Oops") << endl;
            

             

            
            double dt_min_value, dt_max_value;
            Point dt_min_point, dt_max_point;
            minMaxLoc(dtimg, &dt_min_value, &dt_max_value, &dt_min_point, &dt_max_point);
            printf("[dt]min/max: (%f/%f)\n", dt_min_value, dt_max_value);
            
            Mat dt_mask(frame.size(), CV_8U);
            frame.copyTo(dt_mask);
            multiply(dt_mask, dt_mask, dt_mask, 0);
            
            ellipse( dt_mask, dt_max_point+Point(0,-dt_max_point.y*0.2), Size( dt_max_value*3, dt_max_value*3 ), 0, 0, 360, Scalar( 255, 255, 255 ), -1, 8, 0 );
            
            Mat dt_crop;
            frame.copyTo(dt_crop,dt_mask);
            imshow("SK2", dt_crop);
            
            normalize(dtimg, dtimg, 0.0, 1.0, NORM_MINMAX);
             
             // DT end
             
            
            
            
            //---canny&contour begin
            
            Mat edges_crop;
            edgesOri.copyTo(edges_crop,dt_mask);
            
            ellipse( edges_crop, dt_max_point+Point(0,-dt_max_point.y*0.2), Size( dt_max_value*3, dt_max_value*3 ), 0, 0, 360, Scalar( 0, 255, 255 ), 2, 8, 0 );
            
            cvtColor(edges_crop,edges_crop, CV_BGR2GRAY);
            edges_crop = edges_crop>0;
            
            imshow("tmp",edges_crop);
            
            
            
            int thresh = 150;
            Mat canny_output;
            Canny( edges, canny_output, thresh, thresh*2, 3 );
            //Canny( edges_crop, canny_output, thresh, thresh*2, 3 );
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            Mat contourCanvas;
            
            if (isFirstFrame) {
                contourCanvas = Mat::zeros( canny_output.size(), CV_8UC3 );
            }
            
            int max_contour=0;
            
            if (contours.size()<=0) max_contour = -1;
            
            for( int i = 0; i < contours.size(); i++ )
            {
                if ( fabs(contourArea(contours[i],false)) < 300 ) continue;
                //if ( fabs(contourArea(contours[i],false)) > fabs(contourArea(contours[max_contour],false)) )
                if ( contourArea(contours[i],false) > contourArea(contours[max_contour],false) )
                    max_contour = i ;
                /*
                // draw each contour
                Scalar color = Scalar( 255,0,255 );
                drawContours( contourCanvas, contours, i, color, 2, 8, hierarchy, 0, Point() );
                
                
                
                
                Rect boundRect = boundingRect( contours[i] );
                
                std::ostringstream sstream;
                sstream << contourArea(contours[i],false);
                string str_area = sstream.str();
                
                putText(contourCanvas, str_area, cvPoint(boundRect.x+boundRect.width/2, boundRect.y+boundRect.height/2), FONT_HERSHEY_SIMPLEX, 0.6f, Scalar( 0,255,255 ));
                
                imshow("Contours", contourCanvas);
                //cout << "now: " << i << " ( " << boundRect.width << ", " << boundRect.height << ")"<< endl;
                
                waitKey(100);
                */
                
            }
            
            //while(waitKey(30)!=27);
            
            if ( max_contour != -1 ) {
                bool foundLegitMax = false;
                double nowMaxArea = contourArea(contours[max_contour],false);
                
                
                
                if ( lastMaxArea == -1 ) {
                    foundLegitMax = true;
                }
                else if ( nowMaxArea > lastMaxArea ) {
                    if ( nowMaxArea/lastMaxArea < 20 ) {
                        foundLegitMax = true;
                    }
                } else {
                    if ( lastMaxArea/nowMaxArea <= 2.5 ) {
                        foundLegitMax = true;
                    }
                }
                
                
                if ( foundLegitMax ) {
                    contourCanvas = Mat::zeros( canny_output.size(), CV_8UC3 );
                    
                    lastMaxArea = nowMaxArea;
                    Scalar color = Scalar( 50,50,255 );
                    drawContours( contourCanvas, contours, max_contour, color, CV_FILLED, 8, hierarchy, 0, Point() );
                    
                    std::ostringstream sstream;
                    sstream << nowMaxArea;
                    string str_area = sstream.str();
                    
                    putText(contourCanvas, str_area, cvPoint(10, 20), FONT_HERSHEY_SIMPLEX, 0.6f, Scalar( 0,255,255 ));
                    
                    imshow("Contours", contourCanvas);
                }
                 
            }
            
            
            
            //while(waitKey(30)!=27);
            
            //---canny&contour end
            
            
            
            Mat dt_skel;
            dtimg.copyTo(dt_skel, skel);
            dilate(dt_skel,dt_skel,getStructuringElement(MORPH_CROSS,Size(5,5)));
            
            dt_skel.convertTo(dt_skel,CV_8U,255);
            cvtColor(dt_skel,dt_skel,CV_GRAY2BGR);
            
            
            for(int i = 0; i < dt_skel.rows; i++)
            {
                for(int j = 0; j < dt_skel.cols; j++)
                {
                    Vec3b &bgrPixel = dt_skel.at<Vec3b>(i, j);
                    if ( !(bgrPixel.val[0]==0 && bgrPixel.val[1] == 0 && bgrPixel.val[2]==0) ) {
                        bgrPixel.val[0] = color_dist[bgrPixel.val[0]*3][0];
                        bgrPixel.val[1] = color_dist[bgrPixel.val[1]*3][1];
                        bgrPixel.val[2] = color_dist[bgrPixel.val[2]*3][2];
                    }
                    
                    /*
                     if ( !(bgrPixel.val[0]==0 && bgrPixel.val[1] == 0 && bgrPixel.val[2]==0) )
                     printf("(%d,%d,%d)\n",bgrPixel.val[0],bgrPixel.val[1],bgrPixel.val[2]);
                     */
                }
            }
            //while(waitKey(30) < 0) ;
            Mat level_skel;
            dt_skel.copyTo(level_skel,dt_crop);
            ellipse( level_skel, dt_max_point, Size( dt_max_value, dt_max_value ), 0, 0, 360, Scalar( 45, 55, 200 ), -1, 8, 0 );
            imshow("D2", level_skel);
            
            
            //cout << ((dt_skel.type() == CV_8UC3)? "Channel 3":"Oops") << endl;
            
            
            //while(waitKey(30) < 0) ;
            
            
            /*
             Mat canny_output;
             vector<vector<Point> > contours;
             vector<Vec4i> hierarchy;
             
             edges.copyTo(canny_output);
             findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
             Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
             drawContours( drawing, contours, -1, Scalar( 0, 255, 0), 2, 8, hierarchy, 1 , Point() );
             imshow("Contours", drawing);
             
             Canny(edges, edges, 0, 30, 3);
             imshow("Canny", edges);
             */
            
            
            if(waitKey(30) == 27) break;
		} catch(cv::Exception) {};
        
        if ( isFirstFrame ) isFirstFrame = false;
    }
    
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

void detectAndDisplay( Mat frame, Mat cq )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    cq.copyTo(frame);
    
    int max_face = -1;
    static double max_area = -1;
    double max_tmp = -1;
    for( int i = 0; i < faces.size(); i++ )
    {
        Rect face_area(faces[i].x,faces[i].y,faces[i].width,faces[i].height);
        
        
        double nowArea = face_area.area();
        if ( nowArea > max_tmp ) {
            max_face = i;
            max_tmp = nowArea;
        }
        
    }
    double delta= max_tmp/max_area;
    if ( delta >= 0.5 && delta <= 1.5 || max_area == -1 ) {
        max_area = max_tmp;
    } else {
        max_face = -1;
    }
        
    if ( faces.size() > 0 && max_face != -1 ) {
        Rect face_area(faces[max_face].x+faces[max_face].width*0.2,faces[max_face].y+faces[max_face].height*0.5,faces[max_face].width*0.6,faces[max_face].height*0.4);
        //Rect face_area(200,200,50,50);
        rectangle(frame, face_area, Scalar( 0, 0, 0 ), 10, 8, 0 );
        
        Mat face_roi_img;
        cq.copyTo(face_roi_img);
        face_roi_img = face_roi_img(face_area);
        imshow("roi",face_roi_img);
        
        cvtColor(face_roi_img, face_roi_img, CV_BGR2YCrCb);
        Scalar face_mean, face_std;
        meanStdDev(face_roi_img, face_mean, face_std);
        printf("<%.1f, %.1f, %.1f, %.1f>\n", face_mean.val[0],face_mean.val[1],face_mean.val[2],face_mean.val[3]);\
        printf("[%.1f, %.1f, %.1f, %.1f]\n", face_std.val[0],face_std.val[1],face_std.val[2],face_std.val[3]);
        
        double face_mean_adj = 1;
        double face_mean_cr = face_mean.val[1];
        double face_mean_cb = face_mean.val[2];
        
        cr_range[0] = (int)(face_mean_cr*face_mean_adj - face_std.val[2]*cr_min_std/10);
        cr_range[1] = (int)(face_mean_cr*face_mean_adj + face_std.val[2]*cr_max_std/10);
        cb_range[0] = (int)(face_mean_cb*face_mean_adj - face_std.val[1]*cb_min_std/10);
        cb_range[1] = (int)(face_mean_cb*face_mean_adj + face_std.val[1]*cb_max_std/10);
        
        
        //printf("{(%d, %d)  (%d, %d)}\n",cr_range[0],cr_range[1],cb_range[0],cb_range[1]);
        
        Point center( faces[max_face].x + faces[max_face].width*0.5, faces[max_face].y + faces[max_face].height*0.5 );
        ellipse( frame, center, Size( faces[max_face].width*0.6, faces[max_face].height*0.9), 0, 0, 360, Scalar( 0, 0, 0 ), -1, 8, 0 );
        
    }
    
    //-- Show what you got
    //imshow( "face", frame );
    
}

void extractSkin( Mat &input, Mat &output) {
	Mat input_YCrCb;
	cvtColor(input, input_YCrCb, CV_BGR2YCrCb);
    
    
	Mat skinMask;
	inRange(input_YCrCb, Scalar(0,cr_range[0], cb_range[0]), Scalar(256,cr_range[1], cb_range[1]),skinMask);
    
    //inRange(input_YCrCb, Scalar(0,th_cr_range[0], th_cb_range[0]), Scalar(256,th_cr_range[1], th_cb_range[1]),skinMask);
    
	input.copyTo(output,skinMask);
}

void voronoiGen( Mat &dist, Mat &dist8u, Mat &labels ) {
	static const Scalar colors[] =
    {
        Scalar(0,0,0),
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    
    
	dist8u.create(labels.size(), CV_8UC3);
	for( int i = 0; i < labels.rows; i++ )
	{
		const int* ll = (const int*)labels.ptr(i);
		const float* dd = (const float*)dist.ptr(i);
		uchar* d = (uchar*)dist8u.ptr(i);
		for( int j = 0; j < labels.cols; j++ )
		{
			int idx = ll[j] == 0 || dd[j] == 0 ? 0 : (ll[j]-1)%8 + 1;
			float scale = 1.f/(1 + dd[j]*dd[j]*0.0004f);
			int b = cvRound(colors[idx][0]*scale);
			int g = cvRound(colors[idx][1]*scale);
			int r = cvRound(colors[idx][2]*scale);
			d[j*3] = (uchar)b;
			d[j*3+1] = (uchar)g;
			d[j*3+2] = (uchar)r;
		}
	}
}

void getFgBgMask( Mat &fg, Mat &bg, Mat &bgfgMask ) {
	Mat fgYcc, bgYcc;
	cvtColor(fg, fgYcc, CV_RGB2YCrCb);
	cvtColor(bg, bgYcc, CV_RGB2YCrCb);
    
	Mat diff = fgYcc - bgYcc;
    vector<Mat> diffChannels;
    split(diff, diffChannels);
    
    // only operating on luminance for background subtraction...
    threshold(diffChannels[0], bgfgMask, 1, 255.0, THRESH_BINARY_INV);
    
    Mat kernel5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(bgfgMask, bgfgMask, MORPH_OPEN, kernel5x5);
    
    
}