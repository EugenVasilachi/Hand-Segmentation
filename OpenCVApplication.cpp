// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <math.h>

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


// -------------------------------------------- PROIECT ---------------------------------------

Mat loadImage(char path[MAX_PATH]) {
	Mat img = imread(path, IMREAD_COLOR);
	return img;
}

Mat convertToHSV(Mat img_rgb) {
	Mat img_hsv;
	cvtColor(img_rgb, img_hsv, COLOR_BGR2HSV);
	return img_hsv;
}

void splitHSV(const Mat& hsv, Mat& H_channel, Mat& S_channel)
{
	vector<Mat> hsv_channels;
	split(hsv, hsv_channels);
	H_channel = hsv_channels[0];
	S_channel = hsv_channels[1];
}

Mat createMask(const Mat& channel, int lower_threshold, int upper_threshold)
{
	Mat mask;
	inRange(channel, Scalar(lower_threshold), Scalar(upper_threshold), mask);
	return mask;
}

Mat removeSmallObjects(Mat& src, int minArea)
{
	// Etichetarea diferitelor componente
	Mat labels, stats, centroids;
	int nLabels = connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);
	Mat mask(labels.size(), CV_8UC1, Scalar(0));
	Mat surfSup = stats.col(4) > minArea;

	for (int i = 1; i < nLabels; i++)
	{
		if (surfSup.at<uchar>(i, 0))
		{
			mask = mask | (labels == i);
		}
	}

	return mask;
}

void noiceReduction(Mat& src) 
{
	Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3));

	erode(src, src, element1, Point(-1, -1), 2);
	dilate(src, src, element1, Point(-1, -1), 4);
	erode(src, src, element1, Point(-1, -1), 2);
}

Point2d calculateCenterOfMass(Moments& m)
{
	// Calcularea centrului de masa folosind momentele
	Point2d centerOfMass(m.m10 / m.m00, m.m01 / m.m00);

	return centerOfMass;
}


Vec2d calculateAxisDirection(Moments& m) {

	double angle = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);

	Vec2d axisDirection(cos(angle), sin(angle)); // Directia axei

	return axisDirection;
}

bool isHandPointingRight(const Vec2d& axisDirection) {
	// Verificam componenta y a directiei axei de alungire
	return axisDirection[1] < 0;
}

// calculeaza unghiul in radiani
// rad * 180 / PI -> grade
double calculateElongationAxisAtan(Moments& m) {
	double rad = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);
	return rad * 180 / CV_PI;
}


Mat translateAndRotate(Mat& src, Point2d& centerOfMass, double angle) {
	/*Mat translatedImage;
	Mat translationMat = (Mat_<double>(2, 3) << 1, 0, -centerOfMass.x, 0, 1, -centerOfMass.y);
	warpAffine(src, translatedImage, translationMat, src.size());*/

	Mat translatedAndRotatedImage;
	Mat rotationMat = getRotationMatrix2D(centerOfMass, angle, 1.0);;
	warpAffine(src, translatedAndRotatedImage, rotationMat, src.size());

	return translatedAndRotatedImage;
}

vector<int> calculateVerticalProjection(Mat& image) {
	vector<int> projection(image.rows, 0);
	for (int i = 0; i < image.rows; ++i) {
		projection[i] = countNonZero(image.row(i));
	}
	return projection;
}

// Functie pentru a gasi punctul de tranzitie de la palma la antebrat
int findTransitionPoint(const vector<int>& projection, int threshold, bool isRightOriented) {

	// Cand mana e indreptata catre dreapta, incepem verificarea proiectiei de sus (0)
	if (isRightOriented) {
		for (int i = 0; i < projection.size(); ++i) {
			if (abs(projection[i] - projection[i + 1]) > threshold) {
				return i; // Returneaza indexul unde variatia latimii nu mai este monotona
			}
		}
	} // Cand mana e indreptata catre stanga, incepem verificarea proiectiei de jos
	else {
		for (int i = projection.size() - 1; i > 0; --i) {
			if (abs(projection[i] - projection[i - 1]) > threshold) {
				return i;
			}
		}
	}

	return -1; // In cazul in care nu se gaseste un astfel de punct
}


void removeForearm(Mat& image, int transitionPoint, Vec2d& axisDirection) {
	// Presupunem ca tranzitia are loc pe verticala si eliminam tot ce este sub punctul de tranzitie
	if (isHandPointingRight(axisDirection)) {
		for (int i = 0; i < image.rows; ++i) {
			for (int j = 0; j < transitionPoint; ++j) {
				image.at<uchar>(i, j) = 0;
			}
		}
	}
	else {
		for (int i = 0; i < image.rows; ++i) {
			for (int j = transitionPoint; j < image.cols; ++j) {
				image.at<uchar>(i, j) = 0;
			}
		}
	}
}


void callBackFunctionForRegionGrowing(int event, int x, int y, int flags, void* param)
{
	vector<Mat>* channels = (vector<Mat>*)param;
	Mat img_rgb = (*channels)[0];
	Mat H = (*channels)[1] * 255 / 180;
	Mat S = (*channels)[2];
	if (event == CV_EVENT_LBUTTONDOWN) {
		//matricea de etichete
		Mat labels = Mat::zeros(H.size(), CV_8UC1);
		queue<Point> queueRegion;

		int T_h = 34;
		int T_s = 49;
		int w = 3;

		int k = 1;//ethiceta curenta

		//adaug element start in coada
		queueRegion.push(Point(x, y));

		double hue_avg = H.at<uchar>(y, x);
		double sat_avg = S.at<uchar>(y, x);

		// acesta primeste eticheta k
		labels.at<uchar>(y, x) = k;
		int N = 1;// numarul de pixeli din regiune

		//cat timp coada nu e goala
		while (!queueRegion.empty()) {
			// Retine poz. celui mai vechi element din coada
			Point oldest = queueRegion.front();
			// scoate element din coada
			queueRegion.pop();
			int xx = oldest.x;   // coordonatele lui
			int yy = oldest.y;
			// Pentru fiecare vecin al pixelului (xx, yy) ale carui coordonate
			for (int i = yy - w; i <= yy + w; i++) {
				for (int j = xx - w; j <= xx + w; j++) {
					// sunt in interiorul imaginii
					if (j > 0 && i > 0 && j < H.cols && i < H.rows && labels.at<uchar>(i, j) == 0) {

						// Daca abs(hue(vecin) – Hue_avg) < T si labels(vecin) == 0
						//double T = k * hue_avg;
						double h_diff = abs(H.at<uchar>(i, j) - hue_avg);
						double s_diff = abs(S.at<uchar>(i, j) - sat_avg);

						if (h_diff < T_h && s_diff < T_s) {
							// Aduga vecin la regiunea curenta
							queueRegion.push(Point(j, i));

							// labels(vecin) = k
							labels.at <uchar>(i, j) = k;

							// Actualizeaza hue_avg si sat_avg (medie ponderata)
							hue_avg = (N * hue_avg + abs(H.at<uchar>(i, j))) / (N + 1);
							sat_avg = (N * sat_avg + abs(S.at<uchar>(i, j))) / (N + 1);

							N++;
							T_h = hue_avg;
							T_s = sat_avg;
						}
					}
				}
			}
		}

		Mat dst = Mat::zeros(H.size(), CV_8UC1);
		//parcurg imaginea 
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				if (labels.at<uchar>(i, j) != 0) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
		//imshow("Region growing", dst);

		Mat post = dst.clone();
		noiceReduction(post);
		imshow("Postprocesare", post);

		int minArea = 3000;
		removeSmallObjects(post, minArea);

		Mat r(post.size(), CV_8UC1, Scalar(0));
		Mat mask = removeSmallObjects(post, minArea);
		post.copyTo(r, mask);
		imshow("Result region growing + postprocessing + small objects reductiop", r);

		// Calcularea momentelor imaginii binare
		Moments m = moments(r, true);

		Point2d centerOfMass = calculateCenterOfMass(m);
		cout << "Centrul de masa este la (" << centerOfMass.x << ", " << centerOfMass.y << ")" << endl;

		double angleAtan = calculateElongationAxisAtan(m);
		cout << "Unghiul de alungire ATAN este: " << angleAtan << endl;


		Mat translatedAndRotatedImage;
		translatedAndRotatedImage = translateAndRotate(r, centerOfMass, angleAtan);
		imshow("Translated and rotated image", translatedAndRotatedImage);


		vector<int> verticalProjection = calculateVerticalProjection(translatedAndRotatedImage);

		Vec2d axisDirection = calculateAxisDirection(m);
		bool isRightOriented;
		isHandPointingRight(axisDirection) ? isRightOriented = true : isRightOriented = false;


		// Gasirea punctului de tranzitie folosind un prag pentru schimbare
		int threshold = 25;
		int transitionPoint = findTransitionPoint(verticalProjection, threshold, isRightOriented);
		cout << "Punctul de tranzitie de la palma la antebrat este la randul: " << transitionPoint << endl;


		removeForearm(translatedAndRotatedImage, transitionPoint, axisDirection);
		imshow("Palm Only", translatedAndRotatedImage);

		Mat backRotatedImage = translateAndRotate(translatedAndRotatedImage, centerOfMass, -angleAtan);
		imshow("Rotita inapoi", backRotatedImage);

		for (int i = 0; i < backRotatedImage.rows; i++) {
			for (int j = 0; j < backRotatedImage.cols; j++) {
				if (backRotatedImage.at<uchar>(i, j) == 255) {
					img_rgb.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
				}
			}
		}

		imshow("FINAL RESULT", img_rgb);

	}

}

void handDetection() {
	Mat img_rgb;
	Mat img_hsv;
	Mat H, S;
	Mat combinedMask;

	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		img_rgb = loadImage(fname);

		img_hsv = convertToHSV(img_rgb);

		splitHSV(img_hsv, H, S);

		vector<Mat> hs_channels = { img_rgb, H, S };
		imshow("src", img_rgb);
		setMouseCallback("src", callBackFunctionForRegionGrowing, &hs_channels);

		waitKey(0);
	}
}






int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - RGB to HSV\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				handDetection();
				break;
		}
	}
	while (op!=0);
	return 0;
}