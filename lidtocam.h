#pragma once
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace cv;
using namespace std;

unsigned int currentFrame = 0;

Eigen::Matrix3d K;  //相机内参

Mat K_Mat;

bool initial_Flag = false; 
bool addPoint = false;
int num = 0;

Mat	image ;
Mat image_gray;
Mat image_gray_last;

vector<Point2f> keypoints;
vector<Point2f> keypointsLast;
vector<Point3d> pnp3dPoints;
vector<Eigen::Matrix3d> PoseR;
vector<Eigen::Vector3d> Poset;
vector<Eigen::Matrix4d> PoseAll;

//光流跟踪参数
double qualityLevel = 0.01;
double minDistance = 10;
int blockSize = 3;
double k = 0.04;
int maxConerNumber = 300;

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
				
	K<< 7.215377e+02, 0.000000e+00, 6.095593e+02,
				0.000000e+00, 7.215377e+02 ,1.728540e+02 ,
				0.000000e+00, 0.000000e+00, 1.000000e+00 ;

	
			 
	
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
