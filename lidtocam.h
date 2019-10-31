#pragma once
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <list>
#include <Eigen/Core>
#include <Eigen/Dense>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <chrono>

#define pi 3.1415926

#define er 6378137

using namespace cv;
using namespace std;

unsigned int currentFrame = 0;


//3x3 rectifying rotation to make image planes co-planar, R_rect_0X:3x3
Eigen::Matrix<double,4,4> R_rect_00;

//3x4 projection matrix after rectification, P_rect_02:3x4						
Eigen::Matrix<double,3,4>  P_rect_00;

//Transform from velo to cam0, T:4x4
Eigen::Matrix<double,4,4> T_velToCam;

//Imu to velo
Eigen::Matrix4d T_imuToVel;


Eigen::Matrix3d K;  //相机内参

Mat K_Mat;

//Eigen::Isometry3d IMU_inv0;

bool initial_Flag = false; 
bool addPoint = false;
int num = 0;


Mat	image ;
Mat image_gray;
Mat image_gray_last;

//vector<Point2f> predictPuv;

vector<Point2f> keypoints;
vector<Point2f> keypointsLast;
vector<Point3d> pnp3dPoints;
vector<Eigen::Matrix3d> PoseR;
vector<Eigen::Vector3d> Poset;
vector<Eigen::Matrix4d> PoseAll;

//vector<Point2f> pointLast;

//vector<Eigen::Isometry3d> imuPose;
//vector<Point2f> predictPoint;

//光流跟踪参数
double qualityLevel = 0.01;
double minDistance = 10;
int blockSize = 3;
double k = 0.04;
int maxConerNumber = 300;


vector<Eigen::Vector4d> lidarPointClouds;

//float imu_data_k[6] = {0};
//float imu_data_kplus1[6] = {0};

//double dt = 0.1;

//Eigen::Matrix3d R;
//Eigen::Vector3d Tr;

void reduceVector(vector<Point2f> &v, vector<unsigned char> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++){
        if (status[i]){
            v[j++] = v[i];
		}
	}
    v.resize(j);
}

void reduceVector(vector<Point3d> &v, vector<unsigned char> status)
{
	int j = 0;
	for(size_t i = 0;i<v.size();i++){
		if(status[i]){
			v[j++] = v[i];
		}
	}
	v.resize(j);
}

float distance(const Point2f &p , const Point2f &q){

	float dx =p.x - q.x;
	float dy = p.y - q.y;
	return sqrt(dx*dx+dy*dy);
}

void rejectWithF(vector<Point2f> &cur_keypoint, vector<Point2f> &prev_keypoint){
	vector<uchar> status;
	findFundamentalMat(cur_keypoint, prev_keypoint, FM_RANSAC, 1.00, 0.99,status);
	reduceVector(cur_keypoint,status);
	reduceVector(prev_keypoint,status);
}

string getFrameStr(unsigned int frame)
{
	if(frame>9999)
		return "00000"+to_string(frame);
	else if(frame>999)
		return "000000"+to_string(frame);
	else if(frame>99)
		return "0000000"+to_string(frame);
	else if(frame>9)
		return "00000000"+to_string(frame);
	else if(frame<=9)
		return "000000000"+to_string(frame);
}

void loadMatrix()
{
	R_rect_00 << 9.999239e-01, 9.837760e-03, -7.445048e-03, 0,
				-9.869795e-03 ,	9.999421e-01 ,-4.278459e-03 ,0,
				7.402527e-03, 4.351614e-03 ,9.999631e-01,0,
				0,0,0,1;

	P_rect_00 << 7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00 ,
				0.000000e+00, 7.215377e+02 ,1.728540e+02 ,0.000000e+00 ,
				0.000000e+00, 0.000000e+00, 1.000000e+00 ,0.000000e+00;
				
			K<< 7.215377e+02, 0.000000e+00, 6.095593e+02,
				0.000000e+00, 7.215377e+02 ,1.728540e+02 ,
				0.000000e+00, 0.000000e+00, 1.000000e+00 ;

	T_velToCam << 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
			 1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02,
			 9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01,
			 0 ,	0,	0,	1;
			 
	T_imuToVel << 9.999976e-01 ,7.553071e-04, -2.035826e-03,-8.086759e-01,
 			-7.854027e-04,9.998898e-01 ,-1.482298e-02,3.195559e-01,
 			 2.024406e-03 ,1.482454e-02 ,9.998881e-01,-7.997231e-01,
			  0,0,0,1;
	K_Mat = (Mat_<double>(3,3)<< 7.215377e+02, 0.000000e+00, 6.095593e+02,
								0.000000e+00, 7.215377e+02 ,1.728540e+02 ,
								0.000000e+00, 0.000000e+00, 1.000000e+00  );

}

Eigen::Vector3d transformProject(const Eigen::Vector4d& P_lidar)
{	Eigen::Vector3d z_P_uv = P_rect_00*R_rect_00*T_velToCam*P_lidar;
	return Eigen::Vector3d(  z_P_uv[0]/z_P_uv[2]  ,  z_P_uv[1]/z_P_uv[2] , 1 );
}

void testdebug()
{
	if(currentFrame >= 220&&currentFrame<=226){
		ofstream of;
		of.open(getFrameStr(currentFrame)+"dubeg.txt", ios::app);
		double average = 0;
		for(size_t i = 0;i<keypoints.size();i++){
			average += distance(keypoints[i],keypointsLast[i]);
			of<<float(i+1)<<" "<<distance(keypoints[i],keypointsLast[i])<<endl;
		}
		//of<<"average:"<<average/keypoints.size()<<endl;
		of.close();	
	}
}

//求图像点的归一化坐标
vector<Point2f> pixelToCam(const vector<Point2f>& point)
{
	vector<Point2f> cam_normalization;
	for(size_t i = 0;i< point.size();i++){
		Eigen::Vector3d pixel(point[i].x,point[i].y,1);
		Eigen::Vector3d cam(K.inverse()*pixel);
		Point2f tmp_point;
		tmp_point.x = cam[0];
		tmp_point.y = cam[1];

		cam_normalization.push_back(tmp_point);
	}
	return cam_normalization;
}



bool inImage(const Point2f& point)
{
	if(point.x >= 0 && point.x<= 1242 && point.y >= 0 && point.y <= 375){
		return true;
	}
	else{
		return false;
	}
}



//双向光流
void opticalTrack()
{
    vector<Point2f> next_keypoints; 
    vector<Point2f> prev_keypoints;	

	keypointsLast.clear();

	for(auto kp:keypoints){
		prev_keypoints.push_back(kp);
		keypointsLast.push_back(kp);
	}


	vector<unsigned char> status;
	vector<float> error;
	
	calcOpticalFlowPyrLK( image_gray_last ,  image_gray , prev_keypoints , next_keypoints , status ,error ,Size(21, 21), 3);

	vector<unsigned char> back_status;
	vector<Point2f> back_keypointsl;

	calcOpticalFlowPyrLK(image_gray,image_gray_last,next_keypoints, back_keypointsl,back_status,error,Size(21, 21), 3);

	for(size_t i = 0;i<status.size();i++){
		if( status[i] && back_status[i] && inImage(next_keypoints[i]) && distance(prev_keypoints[i],back_keypointsl[i])<0.5 ){
			status[i] = 1;
		}
		else{
			status[i] = 0;
		}
	}

	reduceVector(keypointsLast,status);
	reduceVector(next_keypoints,status);
	if(initial_Flag && !addPoint ){
		reduceVector(pnp3dPoints,status);
	}

	keypoints.clear();
	for(auto kp:next_keypoints){
		keypoints.push_back(kp);
	}
	
	//去掉
	/*
		vector<unsigned char> parallax_status;
	for(int i = 0;i<next_keypoints.size();i++){
		if(distance( next_keypoints[i],keypointsLast[i])>3){
			parallax_status.push_back(1);
		}
		else{
			parallax_status.push_back(0);	
		}
	}
	*/

}

void rejectWithDis()
{
	//去除距离过大的点
	vector<Point2f> next_keypoints;
	vector<Point2f> prev_keypoints;

	for(auto kp:keypoints){
		next_keypoints.push_back(kp);
	}
	for(auto kp:keypointsLast){
		prev_keypoints.push_back(kp);
	}

	auto iterkey = keypoints.begin();
	auto iterKeyLast = keypointsLast.begin();

	for(size_t i=0;i < next_keypoints.size();i++){
		if(distance(next_keypoints[i],prev_keypoints[i])>40){

			iterkey = keypoints.erase(iterkey);
			iterKeyLast = keypointsLast.erase(iterKeyLast);
			i++;
			continue;
		}
		iterkey++;
		iterKeyLast++;

	}
}

//画出两帧之间光流
void visualization()
{
	Mat image_show = image.clone();
	vector<Point2f> p1;
	vector<Point2f> p2;
	for(auto kp:keypoints){
		p1.push_back(kp);
	}
	for(auto kp:keypointsLast){
		p2.push_back(kp);
	}

	RNG rng(0);

	//cout<<keypoints.size()<<endl;
	//cout<<predictPoint.size()<<endl;

	int pointNum = 0;
	for(int i=0;i<keypoints.size();i++){

		//if(distance(p1[i],p2[i])<50){
			//pointNum++;
			Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
			circle(image_show,p1[i],2,Scalar(0,0,255),1);
			line(image_show,p2[i],p1[i],color,1);
			//line(image_show,p2[i],predictPoint[i],Scalar(0,255,0),1);
			
		//}
	}
	//cout<<"pointNum :"<<pointNum<<endl<<endl;
	imshow("Optical flow points",image_show);
	waitKey(10);
}

//三角测量
void triangulation(const Mat& R,const Mat& t,vector<Point3d>& points ,
					const vector<Point2f>& cam_last, const vector<Point2f>& cam_cur )
{
	Mat T1 = (Mat_<double>(3,4)<<1,0,0,0,0,1,0,0,0,0,1,0);  //以第一张图片为参考系
	Mat T2 = (Mat_<double>(3,4)<< R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
									R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0), 
									R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0) );  //第一张图片到第二张图片的运动T_21
	Mat pts_4d;
	triangulatePoints(T1, T2, cam_last, cam_cur, pts_4d);  //pts_4d表示输出的在第一张图片的齐次相机坐标，因此后面需要除以第四维坐标归一化，每一列对应一个坐标点

	for(size_t i = 0; i<pts_4d.cols; i++){
		Mat x = pts_4d.col(i);
		x /= x.at<float>(3,0);
		Point3d p(x.at<float>(0,0), x.at<float>(1,0), x.at<float>(2,0));
		points.push_back(p);
	}

}

void initialRT(const vector<Point2f>& keypointsLast,const vector<Point2f>& keypoints)
{

	//把特征点转为相机归一化坐标
	vector<Point2f> cam_last = pixelToCam(keypointsLast);
	vector<Point2f> cam_cur = pixelToCam(keypoints);

	Mat mask;
	Mat Essential_matrix = findFundamentalMat(cam_last,cam_cur,FM_RANSAC,0.3 / 460,0.99 ,mask); //归一化坐标时可求本质矩阵
	//cout<<"Essential_matrix :"<<Essential_matrix<<endl<<endl;
	//cout<<"mask:"<<mask<<endl;
	Mat R,t;
	Mat cameraMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	int inlier_cnt = recoverPose(Essential_matrix , cam_last,cam_cur,cameraMatrix,R,t,mask); //由本质矩阵求得相机相对运动Tk+1_k，返回内点个数
	/*
	//归一化平移向量
	double sum = sqrt(t.at<double>(0,0)*t.at<double>(0,0) + t.at<double>(1,0)*t.at<double>(1,0) + t.at<double>(2,0)*t.at<double>(2,0) );
	t.at<double>(0,0) /= t.at<double>(0,0)/sum;
	t.at<double>(1,0) /= t.at<double>(1,0)/sum;
	t.at<double>(2,0) /= t.at<double>(2,0)/sum;
	*/
	//cout<<"R is:"<<R<<endl<<endl;
	//cout<<"t is:"<<t<<endl<<endl;
	Eigen::Matrix3d eigenR;
	Eigen::Vector3d eigent;
	cv2eigen(R,eigenR);
	cv2eigen(t,eigent);
	PoseR.push_back(eigenR);
	Poset.push_back(eigent);

	if(inlier_cnt>15){   //
		cout<<"successed initial"<<endl<<endl;
		vector<Point3d> points;
		triangulation(R, t, points, cam_last, cam_cur);  //points必须是double类型

		for(size_t i = 0; i < points.size();i++){
			Mat keypoints_3d = R*(Mat_<double>(3,1)<<points[i].x, points[i].y, points[i].z)+t;
			Point3d tmp_p;
			tmp_p.x = keypoints_3d.at<double>(0,0);
			tmp_p.y = keypoints_3d.at<double>(1,0);
			tmp_p.z = keypoints_3d.at<double>(2,0);
			pnp3dPoints.push_back(tmp_p);

			//验证初始化误差

			keypoints_3d /= keypoints_3d.at<double>(2,0);
			cout<<"keypoints_3d:"<<K_Mat*keypoints_3d<<endl;
			cout<<"keypoints in second image:"<<keypoints[i]<<endl<<endl;
			
		}

		initial_Flag = true;
	}
	else{
		initial_Flag = false;
		cout<<"fail to initial,wait for next image."<<endl;
	}	

	//三角测量
	
}


void bundleAdjustment (
    const vector< Point3d > points_3d,
    const vector< Point2d > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    
    //Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    Block* solver_ptr = new Block ( std::unique_ptr<Block::LinearSolverType>(linearSolver) );     // 矩阵块求解器改

    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::unique_ptr<Block>(solver_ptr) ); //改

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3d p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const Point2d p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    //cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;
	Eigen::Isometry3d T( pose->estimate() );
	R = (Mat_<double>(3,3) << T(0,0),T(0,1),T(0,2),T(1,0),T(1,1),T(1,2),T(2,0),T(2,1),T(2,2));
	t =(Mat_<double>(3,1)<<T(0,3),T(1,3),T(2,3));

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d (pose->estimate()).matrix() <<endl<<endl;
}

void solvedPnP(vector<Point3d>& pnp3dPoints,
				 const vector<Point2f>& keypointsLast, 
				 	const vector<Point2f>& keypoints )
{	
	//PnP求相机运动
	Mat r,t;
	solvePnP(pnp3dPoints, keypoints, K_Mat, Mat(), r, t, false, SOLVEPNP_EPNP); 
	Mat R;
	Rodrigues(r, R);
	cout<<"R is:"<<R<<endl<<endl;
	cout<<"t is:"<<t<<endl<<endl;

	vector<Point2d> tmp_key;
	for(auto kp:keypoints){
		tmp_key.push_back(kp);
	}

	bundleAdjustment(pnp3dPoints,tmp_key,K_Mat,R,t);



	Eigen::Matrix3d eigenR;
	Eigen::Vector3d eigent;
	cv2eigen(R,eigenR);
	cv2eigen(t,eigent);
	PoseR.push_back(eigenR);
	Poset.push_back(eigent);

	//三角测量
	pnp3dPoints.clear();
	vector<Point2f> cam_last = pixelToCam(keypointsLast);
	vector<Point2f> cam_cur = pixelToCam(keypoints);
	vector<Point3d> points;
	triangulation(R, t, points, cam_last, cam_cur); 
	for(size_t i = 0; i < points.size();i++){
		Mat keypoints_3d = R*(Mat_<double>(3,1)<<points[i].x, points[i].y, points[i].z)+t;
		Point3d tmp_p;
		tmp_p.x = keypoints_3d.at<double>(0,0);
		tmp_p.y = keypoints_3d.at<double>(1,0);
		tmp_p.z = keypoints_3d.at<double>(2,0);
		pnp3dPoints.push_back(tmp_p);


		//验证pnp R,t误差

		keypoints_3d /= keypoints_3d.at<double>(2,0);
		cout<<"keypoints_3d:"<<K_Mat*keypoints_3d<<endl;
		cout<<"keypoints in second image:"<<keypoints[i]<<endl<<endl;
	}
	
}

void visualOdometry(const vector<Eigen::Matrix3d>& PoseR, const vector<Eigen::Vector3d>& Poset, vector<Eigen::Matrix4d>& PoseAll)
{
	//把第一帧作为参考系
	Eigen::Matrix4d poseck1_ck;
	//cout<<"OK"<<Poset[currentFrame][0]<<endl;


	poseck1_ck << PoseR[num](0, 0), PoseR[num](0, 1), PoseR[num](0, 2), Poset[num][0],
					PoseR[num](1, 0), PoseR[num](1, 1), PoseR[num](1, 2), Poset[num][1],
					PoseR[num](2, 0), PoseR[num](2, 1), PoseR[num](2, 2), Poset[num][2],
					0,	0,	0,	1;

	
	if(currentFrame == 1){
		PoseAll.push_back(poseck1_ck.inverse());
		//cout<<PoseAll[0];
	}
	else if(currentFrame>=2){
		
		PoseAll.push_back(PoseAll[num-1]*poseck1_ck.inverse());
		//cout<<PoseAll[num-1]<<endl;
	}

	ofstream ofs;
	ofs.open("point.txt", ios::app);
	ofs<< PoseAll[num](0,3)<<" "<<PoseAll[num](2,3)<<endl;
	ofs.close();




	num++;

	
}

void testRT(const vector<Eigen::Matrix3d>& PoseR, const vector<Eigen::Vector3d>& Poset,
			 const vector<Point2f>& keypoints, const vector<Point2f>& keypointsLast,const vector<Point3d> pnp3dPoints)
{
	for(size_t i = 0; i<keypoints.size();i++){
		Eigen::Vector3d x1(keypointsLast[i].x, keypointsLast[i].y, 1);
		Eigen::Vector3d p2 = K*(PoseR[currentFrame-1]*x1 + Poset[currentFrame-1]);
		cout<<"dx:"<<keypoints[i].x - p2[0]<<endl;
		cout<<"dy:"<<keypoints[i].y - p2[1]<<endl<<endl;
	}
}



/*
void readIMUDatak(string imu_name)
{
	//读取IMU数据
	ifstream fs;
	fs.open(imu_name,ios_base::in);
	if(!fs.is_open()){
		cout<<"wrong file path"<<endl;
		exit(0);
	}
	   
	for(int i=0;i<6;i++){
		fs>>imu_data_k[i];
	}
	fs.clear();
	fs.close();
}

void readIMUDatakplus1(string imu_namekplus1)
{
	//读取IMU数据
	ifstream fs1;
	fs1.open(imu_namekplus1,ios_base::in);
	if(!fs1.is_open()){
		cout<<"wrong file path"<<endl;
		exit(0);
	}
	   
	for(int i=0;i<6;i++){
		fs1>>imu_data_kplus1[i];
	}
	fs1.clear();
	fs1.close();
}

//计算IMU相对位姿Tk_k+1

Eigen::Isometry3d makeIMUPose(const float* imu_data_k , const float* imu_data_kplus1)
{
	//计算k时刻的IMU位姿Twk
	double scale = cos(imu_data_k[0]*pi/180.0);
	double mx = scale*imu_data_k[1]*pi*er/180;
	double my = scale*er*log(tan((90+imu_data_k[0])*pi/360));
	double mz = imu_data_k[2];
	Eigen::Vector3d IMU_t(mx,my,mz);

	Eigen::Matrix3d Rx;
	Rx<<1 ,0, 0, 0, cos(imu_data_k[3]), -sin(imu_data_k[3]), 0, sin(imu_data_k[3]) ,cos(imu_data_k[3]);
	Eigen::Matrix3d Ry;
	Ry<<cos(imu_data_k[4]), 0 ,sin(imu_data_k[4]), 0 ,1 ,0, -sin(imu_data_k[4]), 0 ,cos(imu_data_k[4]);
	Eigen::Matrix3d Rz;
	Rz<<cos(imu_data_k[5]), -sin(imu_data_k[5]), 0, sin(imu_data_k[5]), cos(imu_data_k[5]), 0,0 ,0 ,1;

	Eigen::Isometry3d IMU_T = Eigen::Isometry3d::Identity();
	IMU_T.rotate(Rz*Ry*Rx);
	IMU_T.pretranslate(IMU_t);

	//计算K+1时刻的IMU位姿Twk+1
	double scale1 = cos(imu_data_kplus1[0]*pi/180.0);
	double mx1 = scale1*imu_data_kplus1[1]*pi*er/180;
	double my1 = scale1*er*log(tan((90+imu_data_kplus1[0])*pi/360));
	double mz1 = imu_data_kplus1[2];
	Eigen::Vector3d IMU_t1(mx1,my1,mz1);
	
	Eigen::Matrix3d Rx1;
	Rx1<<1 ,0, 0, 0, cos(imu_data_kplus1[3]), -sin(imu_data_kplus1[3]), 0, sin(imu_data_kplus1[3]) ,cos(imu_data_kplus1[3]);
	Eigen::Matrix3d Ry1;
	Ry1<<cos(imu_data_kplus1[4]), 0 ,sin(imu_data_kplus1[4]), 0 ,1 ,0, -sin(imu_data_kplus1[4]), 0 ,cos(imu_data_kplus1[4]);
	Eigen::Matrix3d Rz1;
	Rz1<<cos(imu_data_kplus1[5]), -sin(imu_data_kplus1[5]), 0, sin(imu_data_kplus1[5]), cos(imu_data_kplus1[5]), 0,0 ,0 ,1;

	Eigen::Isometry3d IMU_T1 = Eigen::Isometry3d::Identity();
	IMU_T1.rotate(Rz1*Ry1*Rx1);
	IMU_T1.pretranslate(IMU_t1);
	//cout<<Rz1*Ry1*Rx1<<endl;
	//cout<<IMU_T1.matrix()<<endl;
	return IMU_T1.inverse()*IMU_T;
}

//返回向量的反对称矩阵
Eigen::Matrix3d vecToMat(const Eigen::Vector3d vec)
{
	Eigen::Matrix3d antisymMetryMat;
	antisymMetryMat<< 0 , -vec[2] , vec[1] ,
						vec[2] , 0 ,-vec[0] ,
							-vec[1] , vec[0] , 0;
	return antisymMetryMat;
}

//读取点云并转换为像素坐标
vector<Eigen::Vector3d> readPointClouds(const string pointClouds_name,const vector<Point2f>& keypointsLast )
{	
	lidarPointClouds.clear(); //将上一帧点云清除
	vector<Point2f> p;
	for(auto kp:keypointsLast){
		p.push_back(kp);
	}

  	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); // 创建点云（指针）

  	if(pcl::io::loadPCDFile<pcl::PointXYZ> (pointClouds_name, *cloud) == -1) //* 读入PCD格式的文件，如果文件不存在，返回-1
 	{
   		PCL_ERROR ("Couldn't read file test_pcd.pcd \n"); //文件不存在时，返回错误，终止程序。
    	exit(0) ;
  	}
	cout<<"point clouds size:"<<cloud->points.size()<<endl;

	vector<Eigen::Vector3d> P_cam;
  	for(size_t i = 0; i < cloud->points.size (); ++i) {
		
		Eigen::Vector4d laserPoint(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z,1);
		Eigen::Vector3d P_uv = transformProject(laserPoint);  //把点云投影到图像上

		P_cam.push_back(P_uv);
		lidarPointClouds.push_back(laserPoint);
	}  
	return P_cam;
}

//返回特征点相机坐标
tree_model* buildKDTree(const vector<Eigen::Vector3d>& P_cam ,size_t n)
{	
	
	float datas[n*2];
	float labels[n*2];
	for(size_t i = 0; i < n ; i++){
		datas[i*2] = P_cam[i][0];  //u
		datas[i*2+1] = P_cam[i][1];  //v
		labels[i*2] =(float)(i*2);
		labels[i*2+1] = (float)(i*2+1);
	}

	tree_model *model = build_kdtree(datas,labels,n,2,2);  //构建kdtree
	return model;

}

Eigen::Vector4d findKlabor(const Point2f& keypointsLast_one, tree_model* model,size_t n,const vector<Eigen::Vector3d>& P_cam)
{
	float test[2];
	for(size_t j=0 ; j < 2; j++){
		test[0] = keypointsLast_one.x;
		test[1] = keypointsLast_one.y;
	}
	size_t args[n];
	float dists[n];

	find_k_nearests(model, test, 1, args, dists);  //寻找最近邻
	//cout<<"args:"<<args[0]<<endl;
	//cout<<"dis:"<<dists[0]<<endl;
	//cout<<"Puv u:"<<P_cam[args[0]][0]<<endl;
	//cout<<"Puv v:"<<P_cam[args[0]][1]<<endl;
	Eigen::Vector3d p = P_rect_00*R_rect_00*T_velToCam*lidarPointClouds[args[0]];
	//cout<<"pre u:"<<p[0]/p[2]<<endl;
	//cout<<"pre v:"<<p[1]/p[2]<<endl;

	return T_velToCam*lidarPointClouds[args[0]];

}
*/


/*
//输入图像平面坐标，返回对应相机坐标系下坐标（未校正）4X1
Eigen::Vector4d findPxyz(const Point2f &p,const vector<Eigen::Vector4d>& lidarPointClouds )
{
	
	vector<float> dis;
	Eigen::Vector3d P_uv;
	map< float , int> maplive ;

	int j = 0;
	for(int i=0;i<lidarPointClouds.size();i++){

		P_uv = transformProject(lidarPointClouds[i]);

		dis.push_back(pow(P_uv[0]-p.x,2)+pow(P_uv[1]-p.y,2));
		maplive[dis[j]] = j;
		j++;


	}

	auto iter = maplive.begin();
	int index = iter->second;
	Eigen::Vector4d tmp_pc = T_velToCam*lidarPointClouds[index];
	
	
	return tmp_pc;
	
}
*/



/*
void triangle()
{
	vector<Point2f> p;
	vector<Point2f> q;
	for(auto kp:keypoints){
		p.push_back(kp);
	}
	for(auto kp:keypointsLast){
		q.push_back(kp);
	}
	vector<Point2f> x1;
	vector<Point2f> x2;
	//求利用QR分解求相机归一化坐标

	for(int i = 0;i<p.size();i++){
		Eigen::Vector4d Pxyz1 = P_rect.colPivHouseholderQr().solve(Eigen::Vector4d(q[i].x,q[i].y,1,1));
		Eigen::Vector4d Pxyz2 = P_rect.colPivHouseholderQr().solve(Eigen::Vector4d(p[i].x,p[i].y,1,1));

		x1[i].x = Pxyz1[0];
		x1[i].y = Pxyz1[1];
		x2[i].x = Pxyz2[0];
		x2[i].y = Pxyz2[1];

	}
	Mat pts_4d;
	vector<Point3f> P_xyz;

}

//去除跟踪失败的点，并重新resize
void reduceVector(list<Point2f> &v, vector<unsigned char> status)
{   int j = 0;
    for(auto iter = v.begin();iter!=v.end();iter++){
        if(status[j]==0){
            iter = v.erase(iter);
            continue;
        }
        j++;
    }
}
//重载reduceVector


//用F矩阵去除outliers
void rejectWithF()
{  
    //if(keypoints.size()>=30){
        vector<Point2f> p;
        vector<Point2f> q;
        for(auto kp:keypoints){
            p.push_back(kp);
        }
        for(auto kp:keypointsLast){
            q.push_back(kp);
        }

        vector<unsigned char> status;
        findFundamentalMat( q, p, FM_RANSAC, 1.0, 0.99, status );
        reduceVector(keypointsLast,status);
        reduceVector(keypoints,status);
   // }

}
	*/
/*
void readPointClouds( string pointClouds_name)
{	
	lidarPointClouds.clear();

	// allocate 4 MB buffer
	int32_t num = 1000000;
  	float *data = (float*)malloc(num*sizeof(float));

	// pointers
	float *px = data+0;
 	float *py = data+1;
  	float *pz = data+2;
  

	// load point cloud
	FILE *stream=nullptr;
	stream = fopen(pointClouds_name.c_str(),"rb");
	num = fread(data,sizeof(float),num,stream)/4;
	for (int32_t i=0; i<num; i++) {
    	Eigen::Vector4d tmp_point;
		tmp_point[0] = *px;
		tmp_point[1] = *py;
		tmp_point[2] = *pz;
		tmp_point[3] = 1;
    	px+=4; py+=4; pz+=4; 
		
		Eigen::Vector3d tmp_point_uv =  transformProject(tmp_point);
		if(tmp_point_uv[0] >= 0 && tmp_point_uv[1] >= 0 && tmp_point_uv[0]<=1242 && tmp_point_uv[1]<=375){
			lidarPointClouds.push_back(tmp_point);
			}
  	}
	free(data);
  	fclose(stream);
	
	
	cout<<"PointCloud size:"<<lidarPointClouds.size()<<endl;
	

}
double dataProcess(const Eigen::Vector3d& x1t,const Eigen::Vector3d& Rx2)
{
	double s1 = x1t[0]/Rx2[0];
	double s2 = x1t[1]/Rx2[1];
	double s3 = x1t[2]/Rx2[2];
	double aver = (s1+s2+s3)/3;
	//double var = sqrt(((s1-aver)*(s1-aver)+(s2-aver)*(s2-aver)+(s3-aver)*(s3-aver))/3);
	cout<<"s1:"<<s1<<endl;
	cout<<"s2:"<<s2<<endl;
	cout<<"s3:"<<s3<<endl;
	double max,min;
	if(s1 > s2){
		max = s1;
		min = s2;
	}
	else{
		max = s2;
		min = s1;
	}

	if(max < s3){
		max = s3;
	}
	else if(min>s3){
		min = s3;
	}

	cout<<"max:"<<max<<endl;
	cout<<"min"<<min<<endl;
	
	cout<<"s1:"<<s1<<endl;
	cout<<"s2:"<<s2<<endl;
	cout<<"s3:"<<s3<<endl;
	double s1_normalization = (s1-min)/(max-min);
	double s2_normalization = (s2-min)/(max-min);
	double s3_normalization = (s3-min)/(max-min);
	cout<<"s1_normalization:"<<s1_normalization<<endl;
	cout<<"s2_normalization:"<<s2_normalization<<endl;
	cout<<"s3_normalization:"<<s3_normalization<<endl;
	return s1_normalization,s2_normalization,s3_normalization;
}
//返回特征点对应的相机坐标系下坐标（未校正）
void makeCameraCoordinate(  list<Point2f>& keypointsLast , list<Point2f>& keypoints ,const Eigen::Matrix4d& Tck_ck1 )
{	
	
	vector<Point2f> tmp_keypointsLast;
	vector<Point2f> tmp_keypoints;

	for(auto kp:keypointsLast){
		tmp_keypointsLast.push_back(kp);
	}
	for(auto kp:keypoints){
		tmp_keypoints.push_back(kp);
	}

	vector<Eigen::Vector3d> x1;   //归一化坐标
	vector<Eigen::Vector3d> x2;

	for(int i = 0;i<tmp_keypointsLast.size();i++){
		Eigen::Vector3d tmp_p1(tmp_keypointsLast[i].x,tmp_keypointsLast[i].y,1);
		x1.push_back( K.colPivHouseholderQr().solve(tmp_p1) );
	}  

	for(int j = 0;j<tmp_keypoints.size();j++){
		Eigen::Vector3d tmp_p2(tmp_keypoints[j].x,tmp_keypoints[j].y,1);
		x2.push_back( K.colPivHouseholderQr().solve(tmp_p2) );
		//cout<<"x2:"<<x2[j]<<endl;
	} 

	R<<Tck_ck1(0,0),Tck_ck1(0,1),Tck_ck1(0,2),
		Tck_ck1(1,0),Tck_ck1(1,1),Tck_ck1(1,2),
		Tck_ck1(2,0),Tck_ck1(2,1),Tck_ck1(2,2);
	
	Tr<<Tck_ck1(0,3),Tck_ck1(1,3),Tck_ck1(2,3);

	auto iter2 = keypoints.begin();
	auto iter = keypointsLast.begin();

	for(int k= 0;k < x1.size();k++){
		Eigen::Vector3d x1t(vecToMat(x1[k])*Tr*(-1));
		Eigen::Vector3d Rx2( vecToMat(x1[k])*R*x2[k]);
		double s1 = x1t[0]/Rx2[0];
		double s2 = x1t[1]/Rx2[1];
		double s3 = x1t[2]/Rx2[2];
		//cout<<"s1:"<<s1/s3<<endl;
		//cout<<"s2:"<<s2/s3<<endl;
		//cout<<"s3:"<<s3/s3<<endl<<endl;
		//double s1_normalization,s2_normalization,s3_normalization = dataProcess(x1t,Rx2);

		//cout<<"s1_normalization:"<<s1_normalization<<endl;
		//cout<<"s2_normalization:"<<s2_normalization<<endl;
		//cout<<"s3_normalization:"<<s3_normalization<<endl<<endl;


		//剔除误差较大的点
		
		if(iter!=keypointsLast.end()){
			if(fabs(s1/s3)>10||fabs(s2/s3)>10||fabs(s1/s3)<0.1||fabs(s2/s3)<0.1){
				iter = keypointsLast.erase(iter);
				iter2 = keypoints.erase(iter2);
				k++;
				continue;
			}
			iter++;
			iter2++;
		}
		
	}
}
*/





//对当前特征点进行追踪，函数执行后，keypointsLast是这一帧特征点，keypoints是跟踪到的特征点
/*单向
void opticalTrack()
{
    vector<Point2f> next_keypoints; 
    vector<Point2f> prev_keypoints;	

	keypointsLast.clear();

	for(auto kp:keypoints){
		prev_keypoints.push_back(kp);
		keypointsLast.push_back(kp);
	}


	vector<unsigned char> status;
	vector<float> error;
	
	calcOpticalFlowPyrLK( image_gray_last ,  image_gray , prev_keypoints , next_keypoints , status ,error ,Size(21, 21), 3);

	int j = 0;
	for(auto iter = keypoints.begin();iter!=keypoints.end();j++){
		if( status[j] == 0){
			iter = keypoints.erase(iter);
			continue;
		}
		*iter = next_keypoints[j];
		iter++;
	}

	int j2 = 0;
	for(auto iter = keypointsLast.begin();iter!=keypointsLast.end();j2++){
		if( status[j2] == 0){
			iter = keypointsLast.erase(iter);
			continue;
		}
		iter++;
	}


}
*/
