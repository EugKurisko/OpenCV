#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

int main()
{
  // get images
    Mat image1 = imread("/home/eugen/Desktop/pictures/sample4.jpg");
    Mat image2 = imread("/home/eugen/Desktop/pictures/sample2.jpg");
    if(image1.empty() || image2.empty())
    {
        cout << "Can't open file\n";
        cin.get();
        return -1;
    }
    //Ptr<SiftFeatureDetector> detector = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    //Ptr<Feature2D> f2d = ORB::create();
    //Ptr<ORB> f2d = ORB::create(20, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    //Ptr<SiftFeatureDetector> f2d = ORB::create();
    //SiftFeatureDetector detector;
    //Ptr<FeatureDetector> detector = ORB::create();
    Ptr<FeatureDetector> detector = ORB::create(100, 1.2f, 8, 30, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    //Ptr<SiftDescriptorExtractor> detector = SIFT::create(100, 3, 0.04, 10, 1.6);
    //Ptr<OrbFeatureDetector> detector(25, 1.0f, 2, 10, 0, 2, 0, 10);
    //Ptr<SIFT> extractor = SIFT::create();
    //Ptr<xfeatures2d::SIFT> sift;
    //sift = cv::xfeatures2d::SIFT::create(0,4,0.04,10,1.6);
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    //-- Step 2: Calculate descriptors (feature vectors)    
    Mat descriptors_1, descriptors_2;
    //SiftDescriptorExtractor extr;
    Ptr<SiftDescriptorExtractor> extr = SIFT::create(100, 3, 0.04, 10, 1.6);
    //Ptr<SIFT> extr = SIFT::create();
    extr->compute(image1, keypoints1, descriptors_1);
    extr->compute(image2, keypoints2, descriptors_2);

    //-- Step 3: Matching descriptor vectors using BFMatcher :
    //BFMatcher matcher;
    BFMatcher  matcher;
    //FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    double max_dist = 0; double min_dist = 100;
    double dist;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

  vector<DMatch> good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { 
    if( matches[i].distance < max(1.56*min_dist, 0.02))
    { 
      good_matches.push_back( matches[i]); 
    }
  }
  //vector< DMatch > matches;
//   vector<vector<DMatch> > matches;
//   matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);
// vector<DMatch> match1;
// vector<DMatch> match2;

// for(int i=0; i<matches.size(); i++)
// {
//     match1.push_back(matches[i][0]);
//     match2.push_back(matches[i][1]);
// }

// double max_dist = 0; double min_dist = 100;
//   for( int i = 0; i < descriptors_1.rows; i++ )
//   { 
//     double dist = match1[i].distance;
//     if( dist < min_dist ) 
//       min_dist = dist;
//     if( dist > max_dist ) 
//       max_dist = dist;
//   }
//   for( int i = 0; i < descriptors_1.rows; i++ )
//   { 
//     double dist = match2[i].distance;
//     if( dist < min_dist ) 
//       min_dist = dist;
//     if( dist > max_dist ) 
//       max_dist = dist;
//   }
//  vector<DMatch> good_matches;
  // for( int i = 0; i < descriptors_1.rows; i++ )
  // { 
  //   if( matches[i].distance < max(0.88*min_dist, 0.02))
  //   { 
  //     good_matches.push_back(matches[i]); 
  //   }
  // }
  // for( int i = 0; i < descriptors_1.rows; i++ )
  // { 
  //   if( match2[i].distance < max(0.25*min_dist, 0.02))
  //   { 
  //     good_matches.push_back(match2[i]); 
  //   }
  // }
  namedWindow("", WINDOW_AUTOSIZE);
Mat img_matches1, img_matches2;
Scalar matchColor(0, 255, 255);
Scalar singlePointcolor(255, 255, 0);
drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches1, matchColor, singlePointcolor, vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches2);
imshow("test", img_matches1);
 waitKey(0);
    return 0;
}
  //matcher.match( descriptors_1, descriptors_2, matches );
  // double max_dist = 0; double min_dist = 100;
  // for( int i = 0; i < descriptors_1.rows; i++ )
  // { 
  //   double dist = matches[i].distance;
  //   if( dist < min_dist ) 
  //     min_dist = dist;
  //   if( dist > max_dist ) 
  //     max_dist = dist;
  // }
  // vector<DMatch> good_matches;

  // for( int i = 0; i < descriptors_1.rows; i++ )
  // { 
  //   if( matches[i].distance < max(2*min_dist, 0.02))
  //   { 
  //     good_matches.push_back( matches[i]); 
  //   }
  // }
  // Mat img_matches;
  //   // colors
  //   Scalar matchColor(0, 0, 255);
  //   Scalar singlePointcolor(255, 0, 0);
  //   drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches, matchColor, singlePointcolor, vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //   imshow("test", img_matches);
    // craete FD
  //   Ptr<FeatureDetector> detector = ORB::create(400);
  //   //Ptr<SURF> detector = SURF::create(400);
  //   //SurfFeatureDetector* detector;
  //   // keypoints
  //   //Ptr<SIFT> detector = SIFT::create();
  //   vector<KeyPoint> keypoints1, keypoints2;
  //   // find kp on images
  //   detector->detect(image1, keypoints1);
  //   detector->detect(image2, keypoints2);
  //   // create sift
  //   Ptr<SIFT> extractor  = SIFT::create(400);
  //   Mat descriptors1, descriptors2;
  //   // compute kp on images
  //   extractor->compute(image1, keypoints1, descriptors1);
  //   extractor->compute(image2, keypoints2, descriptors2);
  //   Ptr<BFMatcher> mathcer = BFMatcher::create();
  //   // create FBM to match descriptors
  //   //Ptr<FlannBasedMatcher> mathcer = FlannBasedMatcher::create();
  //   vector<DMatch> matches;
  //   // put matches in vector of matches
  //   mathcer->match(descriptors1, descriptors2, matches);
  //   double max_dist = 0; double min_dist = 100;

  // //Quick calculation of max and min distances between keypoints
  // for( int i = 0; i < descriptors1.rows; i++ )
  // { 
  //   double dist = matches[i].distance;
  //   if( dist < min_dist ) 
  //     min_dist = dist;
  //   if( dist > max_dist ) 
  //     max_dist = dist;
  // }

  // printf("-- Max dist : %f \n", max_dist );
  // printf("-- Min dist : %f \n", min_dist );

  // //Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  // //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very small
  // vector<DMatch> good_matches;

  // for( int i = 0; i < descriptors1.rows; i++ )
  // { 
  //   if( matches[i].distance < 2*min_dist)
  //   { 
  //     good_matches.push_back( matches[i]); 
  //   }
  // }
  //   //namedWindow("matches", 1);
  //   Mat img_matches;
  //   // colors
  //   Scalar matchColor(0, 0, 255);
  //   Scalar singlePointcolor(255, 0, 0);
  //   drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches, matchColor, singlePointcolor, vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //   imshow("test", img_matches);
  //   for( int i = 0; i < (int)good_matches.size(); i++ )
  //   { 
  //     printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); 
  //   }