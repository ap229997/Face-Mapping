// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

This example program shows how to find frontal human faces in an image and
estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, along the eyebrows, on
the eyes, and so forth.


This example is essentially just a version of the face_landmark_detection_ex.cpp
example modified to use OpenCV's VideoCapture object to read from a camera instead
of files.


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
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <iostream>

using namespace dlib;
using namespace cv;
using namespace std;

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

bool cmpr(points a, points b)
{
	if (a.x != b.x) return a.x < b.x;
	else return a.y < b.y;
}

cv_image<bgr_pixel> landmark(cv_image<bgr_pixel> img, std::vector<full_object_detection> shapes, std::vector<node> end_points, int check, bool what)
{
	std::vector<req> polygon;
	req line;
	for (int i = 1; i<end_points.size(); i++)
	{
		line.a = end_points[i].y - end_points[i - 1].y;
		line.b = end_points[i - 1].x - end_points[i].x;
		line.c = end_points[i - 1].y * (end_points[i].x - end_points[i - 1].x) - end_points[i - 1].x * (end_points[i].y - end_points[i - 1].y);
		polygon.push_back(line);

		cout << " " << end_points[i - 1].x << " " << end_points[i - 1].y << " " << end_points[i].x << " " << end_points[i].y << endl;
		cout << "      " << line.a << " " << line.b << " " << line.c << endl;

	}

	node test;

	test.x = shapes[0].part(check).x();
	test.y = shapes[0].part(check).y();

	double product, l1, l2;
	std::vector<node> swap;
	//cout<<"         "<<polygon.size()<<endl;
	for (int i = 0; i< img.nr(); i++)
	{
		for (int j = 0; j< img.nc(); j++)
		{
			int cnt = 0;
			bool flag = 0;
			for (int k = 0; k<polygon.size(); k++)
			{

				l2 = test.x*polygon[k].a + test.y*polygon[k].b + polygon[k].c;
				l1 = j*polygon[k].a + i*polygon[k].b + polygon[k].c;
				product = l1*l2;

				if (product < 0) flag = 1;
			}
			if (!what)
			{
				if (flag) {
					img[i][j].blue = 0;
					img[i][j].red = 0;
					img[i][j].green = 0;
				}
			}
			else
			{
				if (!flag) {
					img[i][j].blue = 0;
					img[i][j].red = 0;
					img[i][j].green = 0;
				}
			}

			//cout<<endl;
		}
	}
	polygon.clear();
	return img;
}

cv_image<bgr_pixel> rotate_to_fit(cv_image<bgr_pixel> img1, cv_image<bgr_pixel> img2, std::vector<full_object_detection> shapes1, std::vector<full_object_detection> shapes2)
{
	cout << "begin" << endl;
	std::vector<points> end_points1;
	std::vector<points> end_points2;
	cout << "centroid taken" << endl;
	double cx = shapes1[0].part(30).x();
	double cy = shapes1[0].part(30).y();
	double rms1 = 0;
	double rms2 = 0;
	for (int k = 0; k<68; k++)
	{
		points p;
		p.x = shapes1[0].part(k).x() - cx;
		p.y = shapes1[0].part(k).y() - cy;
		rms1 += (p.x)*(p.x) + (p.y)*(p.y);
		end_points1.push_back(p);
	}
	cout << "1st one done" << endl;

	cout << " " << shapes2.size() << endl;
	for (int l = 0; l<68; l++)
	{
		cout << "enter" << endl;
		points q;
		q.x = shapes2[0].part(l).x() - cx;
		q.y = shapes2[0].part(l).y() - cy;
		cout << "co-ord shift done" << endl;
		rms2 += (q.x)*(q.x) + (q.y)*(q.y);
		end_points2.push_back(q);
	}
	cout << "2nd one complete" << endl;

	rms1 = rms1 / 68;
	rms2 = rms2 / 68;
	rms1 = sqrt(rms1);  // Standard Deviation 
	rms2 = sqrt(rms2);  // Standard Deviation 

	cout << "standard deviation done" << endl;

	for (int z = 0; z<68; z++)
	{
		end_points1[z].x = (end_points1[z].x) / (rms1); // Scaling of 1st image 
		end_points1[z].y = (end_points1[z].y) / (rms1);
		end_points2[z].x = (end_points2[z].x) / (rms2); // Scaling of 2nd image 
		end_points2[z].y = (end_points2[z].y) / (rms2);
	}

	cout << "scaling done" << endl;
	// Now Rotation of images
	double num = 0;
	double denom = 0;
	double theta = 0;
	for (int m = 0; m<68; m++)
	{
		num += (end_points2[m].x)*(end_points1[m].y) - (end_points2[m].y)*(end_points1[m].x);
		denom += (end_points2[m].x)*(end_points1[m].x) + (end_points2[m].y)*(end_points1[m].y);

	}

	theta = atan2(num, denom);
	cout << "angle is: " << theta*(180 * 1.0) / (2 * 3.14) << endl;
	array2d<bgr_pixel> img3_test;
	rotate_image(img1, img3_test, theta);

	image_window win;
	win.clear_overlay();
	win.set_image(img3_test);

	//cout<<" "<<img1.nr()<<" "<<img1.nc()<<endl;
	//cout<<" "<<img3_test.nr()<<" "<<img3_test.nc()<<endl;

	Mat temp = toMat(img3_test);
	//cout<<" "<<temp.size()<<endl;
	cv_image<bgr_pixel> img3(temp);
	//cout<<" "<<img3.nr()<<" "<<img3.nc()<<endl;
	cout << "complete" << endl;

	return img3;
}

int main(int argc, char** argv)
{
	try
	{
		cv::VideoCapture cap(0);
		image_window win, win_new, win2;

		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		// Grab and process frames until the main window is closed by the user.
		
		Mat project_image;
		for (int rst = 0;; rst++)
		{
			// Grab a frame
			cv::Mat temp, temp_new, water, rain;
			cap >> temp;
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.

			cv_image<bgr_pixel> cimg(temp);
			

			// Detect faces 
			std::vector<rectangle> faces = detector(cimg);
			// Find the pose of each face.
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));

			std::vector<node> end_points;
			/*if (shapes.size() != 0)
			{
				cout << " " << shapes[0].num_parts() << endl;
				for (int i = 0; i < 17; i++)
				{
					node q;
					q.x = shapes[0].part(i).x();
					q.y = shapes[0].part(i).y();
					end_points.push_back(q);
				}

				std::vector<req> polygon;
				req line;
				for (int i = 1; i < end_points.size(); i++)
				{
					line.a = end_points[i].y - end_points[i - 1].y;
					line.b = end_points[i - 1].x - end_points[i].x;
					line.c = end_points[i - 1].y * (end_points[i].x - end_points[i - 1].x) - end_points[i - 1].x * (end_points[i].y - end_points[i - 1].y);
					polygon.push_back(line);

					//cout << " " << end_points[i - 1].x << " " << end_points[i - 1].y << " " << end_points[i].x << " " << end_points[i].y << endl;
					//cout << "      " << line.a << " " << line.b << " " << line.c << endl;

				}

				for (int chk = 1; chk < end_points.size(); chk++)
				{
					for (int i = min(end_points[chk].y, end_points[chk - 1].y); i <= max(end_points[chk].y, end_points[chk - 1].y); i++)
						for (int j = min(end_points[chk].x, end_points[chk - 1].x); j <= max(end_points[chk].x, end_points[chk - 1].x); j++)
						{
							if (i*polygon[chk - 1].a + j*polygon[chk - 1].b + polygon[chk - 1].c == 0)
							{
								for (int k = i - 20; k <= i + 20; k++)
								{
									cimg[k][j].blue = 0;
									cimg[k][j].green = 0;
									cimg[k][j].red = 255;
								}
							}
						}
				}
			}*/

			win2.clear_overlay();
			win2.set_image(cimg);
			win2.add_overlay(render_face_detections(shapes));
			//cout<<" "<<faces.size()<<endl;
			//cout<<" "<<shapes.size()<<endl;
			array2d<bgr_pixel> pattern1;
			array2d<bgr_pixel> pattern2;

			int fix_x = 0, fix_y = 0;
			for (int fc = 0; fc<faces.size(); fc++)
			{
				std::vector<node> end_points;

				for (int i = 0; i<17; i++)
				{
					node q;
					q.x = shapes[fc].part(i).x();
					q.y = shapes[fc].part(i).y();
					end_points.push_back(q);
				}
				for (int i = 26; i >= 25; i--)
				{
					node q;
					q.x = shapes[fc].part(i).x();
					q.y = shapes[fc].part(i).y();
					end_points.push_back(q);
				}
				for (int i = 18; i >= 17; i--)
				{
					node q;
					q.x = shapes[fc].part(i).x();
					q.y = shapes[fc].part(i).y();
					end_points.push_back(q);
				}
				for (int i = 0; i<1; i++)
				{
					node q;
					q.x = shapes[fc].part(i).x();
					q.y = shapes[fc].part(i).y();
					end_points.push_back(q);
				}

				//background subtraction
				cimg = landmark(cimg, shapes, end_points, 30, 0);

				win.clear_overlay();
				win.set_image(cimg);
				win.add_overlay(render_face_detections(shapes));
				win.add_overlay(faces,rgb_pixel(255,0,0));

				//processing pattern to be imposed


				//cout<<"   angle to rotate:  "<<(theta*180)/(2*3.14)<<endl;
				std::vector<rectangle> dets1 = detector(cimg);
				std::vector<full_object_detection> shapes1;
				for (unsigned long i = 0; i < dets1.size(); ++i)
					shapes1.push_back(pose_model(cimg, dets1[i]));


				double num = (shapes[fc].part(27).x() - shapes[fc].part(30).x())*1.0;
				double den = (shapes[fc].part(27).y() - shapes[fc].part(30).y())*1.0;
				double theta = atan2(num, den);

				int select;
				if(rst<=50) select=1;
				else if(rst>50 && rst<100) select=2;
				else select=3;
				load_image(pattern1, argv[select]);

				if(select!=1)
				{
				for (int i = 0; i<pattern1.nr(); i++)
					for (int j = 0; j<pattern1.nc(); j++)
					{
						if (j<pattern1.nc() / 2)
						{
							if (pattern1[i][j].blue > 100) pattern1[i][j].blue = 255;
							if (pattern1[i][j].red > 100 ) 
							{
								if(select==2) pattern1[i][j].red = 0;//255;
								else pattern1[i][j].red=255;
							}
							if (pattern1[i][j].green > 100) pattern1[i][j].green = 0;
						}
						else
						{
							if (pattern1[i][j].blue > 100) pattern1[i][j].blue = 0;
							if (pattern1[i][j].red > 100 ) 
							{
								if(select==2) pattern1[i][j].red = 0;//255;
								else pattern1[i][j].red=255;
							}
							if (pattern1[i][j].green > 100) pattern1[i][j].green = 255;
						}
					}
				}
				//voldemort mask
				/*for(int i=0; i<pattern1.nr(); i++)
				for(int j=0; j<pattern1.nc(); j++)
				{
				if(pattern1[i][j].blue > 200 && pattern1[i][j].green > 200 && pattern1[i][j].red > 200)
				{
				pattern1[i][j].blue = 0;
				pattern1[i][j].red = 0;
				pattern1[i][j].green = 0;
				}
				}*/


				rotate_image(pattern1, pattern2, 3.14 + theta);
				//rotate_image(pattern1,pattern2,0);
				Mat temp1 = toMat(pattern2);
				cv_image<bgr_pixel> pattern(temp1);

				int rows,cols;
				//deadpool,spidey...
				if(select==1)
				{
					rows = 1.25*faces[fc].height();
					cols = 1.25*faces[fc].width();
				}
				else
				{
					rows = 1.75*faces[fc].height();
					cols = 1.75*faces[fc].width();
				}

				cout << "resize start" << endl;
				array2d<bgr_pixel> oimg;
				oimg.set_size(rows, cols);
				resize_image(pattern, oimg);
				Mat check = toMat(oimg);
				cv_image<bgr_pixel> cimg1(check);
				cout << "resize done" << endl;

				assign_image(pattern2, cimg1);

				int i = cimg1.nr() / 2;
				int j = cimg1.nc() / 2;

				//deadpool
				int midr,midc;
				if(select==1)
				{
					midr = shapes[fc].part(27).y()+15;
					midc = shapes[fc].part(27).x();
				}
				else
				{
					midr = shapes[fc].part(30).y();
					midc = shapes[fc].part(30).x();
				}

				i = midr - i;
				j = midc - j;
				fix_x = i;
				fix_y = j;
				cout << " " << i << " " << j << endl;
				/*node q;
				q.x = i;
				q.y = j;
				toproject.push_back(q);*/

				if (i >= 0 && j >= 0)
				{
					for (int rx = 0; rx<cimg1.nr(); rx++)
						for (int ry = 0; ry<cimg1.nc(); ry++)
						{
							//if(ry<cimg1.nc()/2)
							/*{
								if (cimg1[rx][ry].blue < 100 || cimg1[rx][ry].red < 100 || cimg1[rx][ry].green < 100) cimg[i + rx][j + ry].blue = cimg1[rx][ry].blue;//255;//waterflow[rx][ry].blue;//cimg1[rx][ry].blue;
								if (cimg1[rx][ry].blue < 100 || cimg1[rx][ry].red < 100 || cimg1[rx][ry].green < 100) cimg[i + rx][j + ry].red = cimg1[rx][ry].red;//0;//waterflow[rx][ry].red;//cimg1[rx][ry].red ;
								if (cimg1[rx][ry].blue < 100 || cimg1[rx][ry].red < 100 || cimg1[rx][ry].green < 100) cimg[i + rx][j + ry].green = cimg1[rx][ry].green;//0;//waterflow[rx][ry].green;//cimg1[rx][ry].green;
							}*/
							//deadpool
							if(select==1)
							{
							if (cimg1[rx][ry].blue > 200 && cimg1[rx][ry].green > 200 && cimg1[rx][ry].red > 200) {
								cimg1[rx][ry].blue=0;
								cimg1[rx][ry].red=0;
								cimg1[rx][ry].green=0;
							}
							else assign_pixel(cimg[i + rx][j + ry], cimg1[rx][ry]);
							}
							else
							{
							if(cimg1[rx][ry].blue > 100 || cimg1[rx][ry].red > 100 || cimg1[rx][ry].green > 100) cimg[i+rx][j+ry].blue = cimg1[rx][ry].blue;
							if(cimg1[rx][ry].blue > 100 || cimg1[rx][ry].red > 100 || cimg1[rx][ry].green > 100) cimg[i+rx][j+ry].red = cimg1[rx][ry].red;
							if(cimg1[rx][ry].blue > 100 || cimg1[rx][ry].red > 100 || cimg1[rx][ry].green > 100) cimg[i+rx][j+ry].green = cimg1[rx][ry].green;
							}
							/*else
							{
							if(cimg1[rx][ry].blue > 100) cimg[i+rx][j+ry].blue = waterflow[i+rx][j+ry].blue;//cimg1[rx][ry].blue;
							if(cimg1[rx][ry].red > 100) cimg[i+rx][j+ry].red = waterflow[i+rx][j+ry].red;//cimg1[rx][ry].red ;
							if(cimg1[rx][ry].green > 100) cimg[i+rx][j+ry].green = waterflow[i+rx][j+ry].green;//cimg1[rx][ry].green;
							}
							/*if(cimg1[rx][ry].blue > 100) cimg[i+rx][j+ry].blue = cimg1[rx][ry].blue;
							if(cimg1[rx][ry].red > 100) cimg[i+rx][j+ry].red = cimg1[rx][ry].red ;
							if(cimg1[rx][ry].green > 100) cimg[i+rx][j+ry].green = cimg1[rx][ry].green;*/
						}

					array2d<bgr_pixel> final_image;
					//final_image.set_size(1340, 1840);
					//resize_image(cimg, final_image);
					win_new.clear_overlay();
					//win_new.set_image(final_image);
					win_new.set_image(cimg);
					win_new.add_overlay(render_face_detections(shapes));
					

					/*for(int rx=0; rx<cimg.nr(); rx++)
					for(int ry=0; ry<cimg.nc(); ry++)
					{
					if(cimg[rx][ry].blue==0 || cimg[rx][ry].red==0 || cimg[rx][ry].green==0)
					{
					cimg[rx][ry].blue = rainvideo[rx][ry].blue;
					cimg[rx][ry].red = rainvideo[rx][ry].red;
					cimg[rx][ry].green = rainvideo[rx][ry].green;
					}
					}*/
				}
				//cout<<" "<<dets1[0].height()<<" "<<dets1[0].width()<<endl;
				//cout<<" "<<cimg1.nr()<<" "<<cimg1.nc()<<endl;
			}

			cout << "mask done" << endl;
			

		}


		

	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}