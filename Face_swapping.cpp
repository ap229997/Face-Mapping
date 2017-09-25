// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image.  In
    particular, this program shows how you can take a list of images from the
    command line and display each on the screen with red boxes overlaid on each
    human face.

    The examples/faces folder contains some jpg images of people.  You can run
    this program on them and see the detections by executing the following command:
        ./face_detection_ex faces/*.jpg

    
    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  This type of object detector is fairly
    general and capable of detecting many types of semi-rigid objects in
    addition to human faces.  Therefore, if you are interested in making your
    own object detectors then read the fhog_object_detector_ex.cpp example
    program.  It shows how to use the machine learning tools which were used to
    create dlib's face detector. 


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <math.h>  


using namespace dlib;
using namespace std;
using namespace cv;
// ----------------------------------------------------------------------------------------

struct points{
float x;
float y;
};

struct node{
	int x;
	int y;
};

struct req{
	int a;
	int b;
	int c;
};


int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1)
        {
            cout << "Give some image files as arguments to this program." << endl;
            return 0;
        }

        frontal_face_detector detector = get_frontal_face_detector();
        image_window win1;
		image_window win2;
		image_window win3;
		image_window win4;
		image_window win5;
		shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        //Load both the images in img1 & img2 respectively
       
           // cout << "processing image " << argv[i] << endl;
		
			
		//Mat img_1 = imread(argv[1]);
		//Mat img_2 = imread(argv[2]);

		array2d<bgr_pixel> img1;
		array2d<bgr_pixel> img2;
		array2d<bgr_pixel> img3;
		//cv_image<bgr_pixel> img3;
            load_image(img1, argv[1]);
			load_image(img2, argv[2]);

			cout<<" "<<img1.nr()<<" "<<img1.nc()<<endl;
			cout<<" "<<img2.nr()<<" "<<img2.nc()<<endl;

            // Make the image bigger by a factor of two.  This is useful since
            // the face detector looks for faces that are about 80 by 80 pixels
            // or larger.  Therefore, if you want to find faces that are smaller
            // than that then you need to upsample the image as we do here by
            // calling pyramid_up().  So this will allow it to detect faces that
            // are at least 40 by 40 pixels in size.  We could call pyramid_up()
            // again to find even smaller faces, but note that every time we
            // upsample the image we make the detector run slower since it must
            // process a larger image.
           // pyramid_up(img1);///////////////hereherehereehjksnfffffknfffffffffnfnfnfnfnfnfnfnfnnfnfn
			//pyramid_up(img2);
            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces it can find in the image.
			//next line
           std::vector<rectangle> dets1 = detector(img1);
		   std::vector<rectangle> dets2 = detector(img2);

			 std::vector<full_object_detection> shapes1;
			 std::vector<full_object_detection> shapes2;
            for (unsigned long i = 0; i < dets1.size(); ++i)
                shapes1.push_back(pose_model(img1, dets1[i]));
		    for (unsigned long j = 0; j < dets1.size(); ++j)
                shapes2.push_back(pose_model(img2, dets2[j]));
	
			image_window win6;
			win6.clear_overlay();
			win6.set_image(img2);

	// Translation Scaling & Rotation of 2 images starts 
		std::vector<points> end_points1;
		std::vector<points> end_points2;
		
		double cx = shapes1[0].part(30).x();
		double cy = shapes1[0].part(30).y();
		double rms1=0;
		double rms2=0;
		for(int k=0; k<68; k++)
			{
				points p;
				p.x = shapes1[0].part(k).x() - cx;
				p.y = shapes1[0].part(k).y() - cy;
				rms1 += (p.x)*(p.x) + (p.y)*(p.y);
				end_points1.push_back(p);
			}


		for(int l=0; l<68; l++)
			{
				points q;
				q.x = shapes2[0].part(l).x() - cx;
				q.y = shapes2[0].part(l).y() - cy;
				rms2 += (q.x)*(q.x) + (q.y)*(q.y);
				end_points2.push_back(q);
			}

		rms1 = rms1/68;
		rms2 = rms2/68;
		rms1= sqrt(rms1);  // Standard Deviation 
		rms2= sqrt(rms2);  // Standard Deviation 

		//cout<<"standard deviation done"<<endl;

		for(int z=0;z<68;z++)
		{
		end_points1[z].x = (end_points1[z].x)/(rms1); // Scaling of 1st image 
		end_points1[z].y = (end_points1[z].y)/(rms1);
		end_points2[z].x = (end_points2[z].x)/(rms2); // Scaling of 2nd image 
		end_points2[z].y = (end_points2[z].y)/(rms2);
		}

		//cout<<"scaling done"<<endl; 
		// Now Rotation of images
		double num=0;
		double denom=0;
		double theta=0;
		for(int m=0; m<68;m++)
		{ num += (end_points2[m].x)*(end_points1[m].y) - (end_points2[m].y)*(end_points1[m].x);
		  denom += (end_points2[m].x)*(end_points1[m].x) + (end_points2[m].y)*(end_points1[m].y);

			}		

		theta= atan2(num,denom);
		
		array2d<bgr_pixel> img3_test;
		rotate_image(img1,img3,theta);
		rotate_image(img1,img3_test,theta);

		// Face alignment over.....now Colour correction starts
		double COLOUR_CORRECT_BLUR_FRAC = 0.6;
		int blur_amount=0;
		
		array<bgr_pixel> eyel_pixels;//pixel values of left and right eye
		array<bgr_pixel> eyer_pixels;

		eyel_pixels.resize(6);
		eyer_pixels.resize(6);
		array2d<bgr_pixel> img2_blur;
		array2d<bgr_pixel> img3_blur;

		 std::vector<rectangle> dets3 = detector(img3);
		   
			std::vector<full_object_detection> shapes3;
            for (unsigned long i = 0; i < dets3.size(); ++i)
                shapes3.push_back(pose_model(img3, dets3[i]));


			//Shifting starts
			int x,y;

			y=shapes3[0].part(30).y() - shapes2[0].part(30).y();
			x= shapes3[0].part(17).x() - shapes2[0].part(17).x();
			
			for(int i=0;i<img3.nr()-y;i++)
			{
				for(int j=0;j<img3.nc()-x;j++)
				{
					img3[i][j]=img3[i+y][j+x];
					img3_test[i][j]=img3_test[i+y][j+x];
				}
			}
		 //saving left eye coordinates
		cout<<"left eye started"<<endl;
		for(int a=36; a<=41;a++)
		{
		  int rx = shapes3[0].part(a).x();
		  int ry = shapes3[0].part(a).y();

		  assign_pixel(eyel_pixels[a-36],img3[ry][rx]);
		  
		}
		//cout<<"left eye done"<<endl;

		//saving right eye coordinates
		for(int a=42; a<=47;a++)
		{
		  //points s;
		  int sx = shapes3[0].part(a).x();
		  int sy = shapes3[0].part(a).y();
		  
		  bgr_pixel val;
		  val.blue=unsigned char (img3[sy][sx].blue);
		  val.green=unsigned char (img3[sy][sx].green);
		  val.red=unsigned char (img3[sy][sx].red);
		  
		  eyer_pixels[a-42].blue=val.blue;
		  eyer_pixels[a-42].red=val.red;
		  eyer_pixels[a-42].green=val.green;
		  //assign_pixel(eyer_pixels[a-42],val);
		}

		cout<<"blur started"<<endl;
		//blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(numpy.mean(eyel_pixels, axis=0) - numpy.mean(eyer_pixels, axis=0));
		double mean_arrayl[3];
		double mean_arrayr[3];
		double diff_array[3];

		for(int i=0; i<3; i++)
		{
			mean_arrayl[i]=0.0;
			mean_arrayr[i]=0.0;
		}

		  for (int b=0;b<6;b++)
		  {
			mean_arrayl[0] += (unsigned int)eyel_pixels[b].blue;
			mean_arrayr[0] += (unsigned int)eyer_pixels[b].blue;
		  }

		   for (int b=0;b<6;b++)
		  {
			mean_arrayl[1] += (unsigned int)eyel_pixels[b].green;
			mean_arrayr[1] += (unsigned int)eyer_pixels[b].green;
		  }
		   cout<<" "<<mean_arrayl[1]<<" "<<mean_arrayr[1]<<endl;

		    for (int b=0;b<6;b++)
		  {
			  mean_arrayl[2] += (unsigned int)eyel_pixels[b].red;
			  mean_arrayr[2] += (unsigned int)eyer_pixels[b].red;
		  }
		
			blur_amount = 0;
		for (int b=0;b<3;b++)
		{
			mean_arrayl[b] /= 3;
			mean_arrayr[b] /= 3;
			diff_array[b]=mean_arrayl[b]-mean_arrayr[b];
			blur_amount += (diff_array[b])*(diff_array[b]); // calculating norm
		}

		 blur_amount *= COLOUR_CORRECT_BLUR_FRAC ;
		// blur_amount should be positive and odd
		if (blur_amount % 2 == 0)
         blur_amount += 1 ;

		//cout<<"blur amount "<<blur_amount<<endl;
		double sigma = 0.3*((blur_amount)*0.5 - 1) + 0.8;
		cout<<" sigma: "<<sigma<<endl;
		
		dlib::gaussian_blur(img2,img2_blur,sigma/400,blur_amount); 
		dlib::gaussian_blur(img3,img3_blur,sigma/700,blur_amount);
		
		//cout<<blur_amount<<endl;
		cout<<int(img3_blur[10][10].blue)<<endl;
		cout<<int(img2_blur[10][10].blue)<<endl;
		cout<<img3.nr()<<endl;
		cout<<img3.nc()<<endl;
		//bgr_pixel temp;
		double temp2;

		win2.clear_overlay();
        win2.set_image(img2);

			cout<<"next blur done"<<endl;

		for(int i=0;i<img2.nr();i++)
		{
			for(int j=0;j<img2.nc();j++)
			{
				bgr_pixel temp;
				if(img3_blur[i][j].blue!=0)

				{
							temp2 =(double(img3[i][j].blue))*(double(img2_blur[i][j].blue))/(double(img3_blur[i][j].blue));
							temp.blue=(char)(temp2);
				}
				if(img3_blur[i][j].red!=0)
				{
						temp2=(double(img3[i][j].red))*(double(img2_blur[i][j].red))/(double(img3_blur[i][j].red));
						temp.red=(char)(temp2);
				}
				if(img3_blur[i][j].green!=0)
						temp2 = (double(img3[i][j].green))*(double(img2_blur[i][j].green))/(double(img3_blur[i][j].green));
				temp.green=(char)(temp2);
				assign_pixel(img3[i][j],temp);
				
			}
		}

		image_window win7;
		win7.clear_overlay();
		win7.set_image(img3);

		cout<<"final blur done"<<endl;

		shapes3.clear();
		dets3.clear();
		dets3 = detector(img3);
		for (unsigned long i = 0; i < dets3.size(); ++i)
                shapes3.push_back(pose_model(img3, dets3[i]));

		std::vector<node> end_points;

			for(int i=0; i<17; i++)
			{
				node q;
				q.x = shapes3[0].part(i).x();
				q.y = shapes3[0].part(i).y();
				end_points.push_back(q);
			}
			for(int i=26; i>=25; i--)
			{
				node q;
				q.x = shapes3[0].part(i).x();
				q.y = shapes3[0].part(i).y();
				end_points.push_back(q);
			}
			for(int i=18; i>=17; i--)
			{
				node q;
				q.x = shapes3[0].part(i).x();
				q.y = shapes3[0].part(i).y();
				end_points.push_back(q);
			}
			for(int i=0; i<1; i++)
			{
				node q;
				q.x = shapes3[0].part(i).x();
				q.y = shapes3[0].part(i).y();
				end_points.push_back(q);
			}

			std::vector<req> polygon;
			req line;
			for(int i=1; i<end_points.size(); i++)
			{
				line.a = end_points[i].y - end_points[i-1].y;
				line.b = end_points[i-1].x - end_points[i].x;
				line.c = end_points[i-1].y * ( end_points[i].x - end_points[i-1].x ) - end_points[i-1].x * ( end_points[i].y - end_points[i-1].y );
				polygon.push_back(line);

				cout<<" "<<end_points[i-1].x<<" "<<end_points[i-1].y<<" "<<end_points[i].x<<" "<<end_points[i].y<<endl;
				cout<<"      "<<line.a<<" "<<line.b<<" "<<line.c<<endl;

			}

			node test;

			test.x = shapes3[0].part(30).x();
			test.y = shapes3[0].part(30).y();

			double product,l1,l2;
			//std::vector<node> swap;
			
			for(int i=0; i< img3.nr(); i++)
			{
				for(int j=0; j< img3.nc(); j++)
				{
					int cnt=0;
					bool flag=0;
					for(int k=0; k<polygon.size(); k++)
					{
						l2 = test.x*polygon[k].a + test.y*polygon[k].b + polygon[k].c;
						l1 = j*polygon[k].a + i*polygon[k].b + polygon[k].c;
						product = l1*l2;

						if(product < 0) flag=1;
					}
					if(!flag) {
						node t;
						t.x=i;
						t.y=j;
						//swap.push_back(t);
						
					}
					else {
						img3[i][j].blue=0;
						img3[i][j].red=0;
						img3[i][j].green=0;
					}
					//cout<<endl;
				}
			}

			win1.clear_overlay();
            win1.set_image(img3);

			std::cout<<" polygon done"<<endl;

			
			/*int feather_amount = 50;
			dlib::gaussian_blur(img3_test,img3_new,5,feather_amount);
			*/
			
		//blackout the region on which mask is to be placed
			/*for(int i=0; i<img2.nr(); i++)
				for(int j=0; j<img2.nc(); j++)
				{
					if(img3[i][j].blue!=0 || img3[i][j].red!=0 || img3[i][j].green!=0) 
					{
						img2[i][j].blue=0;
						img2[i][j].green=0;
						img2[i][j].red=0;
					}
				}*/ 

			std::vector<rectangle> faces3 = detector(img2);
			std::vector<rectangle> faces5 = detector(img3);
			std::vector<full_object_detection> shapes5;
			shapes3.clear();
			for(int i=0; i<faces3.size(); i++)
				shapes3.push_back(pose_model(img2,faces3[i]));
			for(int i=0; i<faces5.size(); i++)
				shapes5.push_back(pose_model(img3,faces5[i]));

			int adjust_rows = (double(faces3[0].height())/double(faces5[0].height())) * double(img3.nr());
			int adjust_cols = (double(faces3[0].width())/double(faces5[0].width())) * double(img3.nc());

			array2d<bgr_pixel> new_image;
			new_image.set_size(adjust_rows,adjust_cols);
			resize_image(img3,new_image);
			Mat temp1 = toMat(new_image);
			cv_image<bgr_pixel> img6(temp1);

			std::vector<rectangle> faces6 = detector(img6);
			std::vector<full_object_detection> shapes6;
			for(int i=0; i<faces6.size(); i++)
				shapes6.push_back(pose_model(img6,faces6[i]));

			for(int i=faces3[0].top(); i<faces3[0].top()+faces3[0].height(); i++)
				for(int j=faces3[0].left(); j<faces3[0].left()+faces3[0].width(); j++)
				{
					if(img3[i][j].blue!=0 || img3[i][j].red!=0 || img3[i][j].green!=0) assign_pixel(img2[i][j],img3[i][j]);
				}
		
				std::vector<rectangle> faces7 = detector(img2);
				std::vector<full_object_detection> shapes7;
				for(int i=0; i<faces7.size(); i++)
					shapes7.push_back(pose_model(img2,faces7[i]));

				std::vector<chip_details> chip_locations;
				chip_details temp(faces3[0],faces3[0].area(),theta);
				chip_locations.push_back(temp);

				array2d<bgr_pixel> ext;
				extract_image_chip(img2,temp,ext);
			
			//win1.clear_overlay();
            //win1.set_image(img1);

			win2.clear_overlay();
            win2.set_image(img2);
			//win2.add_overlay(render_face_detections(shapes7));
			//win2.add_overlay(faces7,rgb_pixel(255,0,0));
			
			win3.clear_overlay();
            win3.set_image(img3);

			win4.clear_overlay();
			win4.set_image(img6);
			win4.add_overlay(faces6,rgb_pixel(255,0,0));
			win4.add_overlay(render_face_detections(shapes6));
		
			win5.clear_overlay();
			win5.set_image(ext);
		
		
            // Now we show the image on the screen and the face detections             
            win1.clear_overlay();
            win1.set_image(img1);
		    win1.add_overlay(render_face_detections(shapes1));

			win2.clear_overlay();
            win2.set_image(img2);
		    //win2.add_overlay(render_face_detections(shapes2));
		
			win3.clear_overlay();
            win3.set_image(img3);
			//win3.add_overlay(render_face_detections(shapes3));
            //win3.add_overlay(dets3, rgb_pixel(255,0,0));

			win4.clear_overlay();
            win4.set_image(img2_blur);

			win5.clear_overlay();
            win5.set_image(img3_blur);
			
            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

