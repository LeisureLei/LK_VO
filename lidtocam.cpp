#include "lidtocam.h"


int main(int argc,char** argv)
{	
	for(;currentFrame < 600 ;){
		//string pointClouds_name = "/home/leibing/Downloads/2011_09_26_drive_0096_sync/velodyne_points/data_new/" + getFrameStr(currentFrame) + ".pcd";
		string image_name = "/home/leibing/Downloads/2011_09_26_drive_0095_sync/image_00/data/" + getFrameStr(currentFrame) + ".png";
		//string imu_name = "/home/leibing/Downloads/2011_09_26_drive_0061_sync/oxts/data/" + getFrameStr(currentFrame) + ".txt";
		//string imu_namekplus1 = "/home/leibing/Downloads/2011_09_26_drive_0061_sync/oxts/data/" + getFrameStr(currentFrame+1) + ".txt";

		
		vector<Point2f> corners;
		image = imread(image_name);//原图
		image_gray = imread(image_name,0);//灰度图
		cout << "currentFrame:"<< currentFrame << endl;
		//readIMUDatak(imu_name);  
		//readIMUDatakplus1(imu_namekplus1);

		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
		//Eigen::Isometry3d IMU_Pose = makeIMUPose(imu_data_k,imu_data_kplus1);
		//cout<<"IMU_T : \n"<<IMU_T.matrix()<<endl;

		vector<Point2f> keypointsCopy;  
		loadMatrix();
		//光流跟踪
		if(currentFrame==0){
			goodFeaturesToTrack(image_gray,corners,maxConerNumber,qualityLevel,minDistance);
			cout<<"first corner size:"<<corners.size()<<endl;
			currentFrame++;
			for(auto kp:corners){
				keypoints.push_back(kp);
				keypointsCopy.push_back(kp);
			}			
			image_gray_last = image_gray;
			continue ;
		}

		opticalTrack();
		keypointsCopy.clear();
		for(auto kp:keypoints){
			keypointsCopy.push_back(kp);
		}

		if(keypoints.size()<50){  //	当跟踪点<50，加入新的特征点,新特征点不能和原特征点太近，dis>9
		    addPoint =true;
			//pnp求R,t
			Mat r,t;
			solvePnP(pnp3dPoints, keypoints, K_Mat, Mat(), r, t, false, SOLVEPNP_EPNP); 
			Mat R;
			Rodrigues(r, R);
			//cout<<"R is:"<<R<<endl;
			//cout<<"t is:"<<t<<endl;
							
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

			//加入新的特征点
			corners.clear();
			goodFeaturesToTrack(image_gray,corners,maxConerNumber,qualityLevel,minDistance);
			rejectWithF(keypoints,keypointsLast);

			for(int i=0;i<corners.size();i++){
				float x1 = corners[i].x;
				float y1 = corners[i].y;

				vector<float> dis;

				for(auto kp:keypoints){
					float x2 = kp.x;
					float y2 = kp.y;
					dis.push_back((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
				}

				sort(dis.begin(),dis.end());
				if(dis[0]>225){
					keypointsCopy.push_back(corners[i]);
				}			
			}
			keypoints.clear();
			for(auto kp:keypointsCopy){
				keypoints.push_back(kp);
			}

			//对加入后的所有特征点进行光流跟踪
			opticalTrack();

			//三角化新加入后所有特征点，更新pnp3dPoints
				
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
			
			keypointsCopy.clear();

			//rejectWithF();
			for(auto kp:keypoints){
				keypointsCopy.push_back(kp);
			}
		}

		keypointsCopy.clear();

		//rejectWithF();
		for(auto kp:keypoints){
			keypointsCopy.push_back(kp);
		}
		
		
		/*
		//计算Tck_ck+1
		Eigen::Matrix4d Tck1_ck = T_velToCam*T_imuToVel*IMU_Pose*T_imuToVel.inverse()*T_velToCam.inverse();
									
		//cout<<"Tck_ck1"<<Tck_ck1<<endl;
		
		//求出相机坐标系下坐标
		//makeCameraCoordinate(keypointsLast,keypoints,Tck_ck1);
		vector<Eigen::Vector3d> P_cam = readPointClouds(pointClouds_name,keypointsLast);
	
		vector<Point2f> keypointsLastVec;
		for(auto kp:keypointsLast){
			keypointsLastVec.push_back(kp);
		}
		size_t n = P_cam.size();
		tree_model* model = buildKDTree(P_cam , n );

		predictPoint.clear();
		for(size_t i = 0;i<keypointsLastVec.size();i++){
			Eigen::Vector4d Pck = findKlabor( keypointsLastVec[i],model,n,P_cam);
			Eigen::Vector3d Puvk1 = P_rect_00*R_rect_00*Tck1_ck*Pck;
			Point2f tmp_point;
			tmp_point.x = Puvk1[0]/Puvk1[2];
			tmp_point.y = Puvk1[1]/Puvk1[2];
			predictPoint.push_back(tmp_point);
			//cout<<"u       :"<<keypointsCopy[i].x<<endl;
			//cout<<"predict u:"<< Puvk1[0]/Puvk1[2]<<endl;   //u的误差比较大
			//cout<<"v        :"<<keypointsCopy[i].y<<endl;
			//cout<<"predict v:"<< Puvk1[1]/Puvk1[2]<<endl<<endl;

		}
		//去除错误光流跟踪点
		//rejectWithGPS();
		*/
		cout<<"Tracked next points:"<<keypoints.size()<<endl;
		cout<<"keypointsLast:"<<keypointsLast.size()<<endl;

		if(!initial_Flag){
			initialRT(keypointsLast,keypoints);  //对极几何+三角测量
		}
		else if(!addPoint){
			//pnp
			solvedPnP(pnp3dPoints,keypointsLast,keypoints);
			
			//可视化位姿
		}
		addPoint = false;
		visualization();
		visualOdometry(PoseR, Poset, PoseAll);
		//testRT( PoseR,  Poset, keypoints, keypointsLast);
		image_gray_last = image_gray;
		//cout<<"pose num is:"<<PoseR.size()<<endl;


		//debug
		testdebug();

		currentFrame++;	

	}
	return 0;
}
	
	
	
	
	

	



