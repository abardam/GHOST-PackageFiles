#include <opencv2\opencv.hpp>
#include <gh_common.h>
#include <gh_search.h>
#include <cv_skeleton.h>
#include <recons_voxel.h>
#include <recons_voxel_integration.h>
#include <recons_marchingcubes.h>
#include <recons_cylinder.h>

std::string video_directory = "";
std::string voxel_recons_path = "";
std::string extension = ".xml.gz";
int numframes = 10;
bool skip_side = false;

BodyPartDefinitionVector bpdv;
std::vector<SkeletonNodeHardMap> snhmaps;
std::vector<Cylinder> cylinders;
std::vector<VoxelMatrix> voxels;
float voxel_size;
float tsdf_offset = 0;

std::vector<FrameDataProcessed> frame_datas;
BodypartFrameCluster bodypart_frame_cluster;

std::vector<std::vector<float>> triangle_vertices;
std::vector<std::vector<unsigned int>> triangle_indices;
std::vector<std::vector<unsigned char>> triangle_colors;


void load_packaged_file(std::string filename,
	BodyPartDefinitionVector& bpdv,
	std::vector<FrameDataProcessed>& frame_datas,
	BodypartFrameCluster& bodypart_frame_cluster,
	std::vector<std::vector<float>>& triangle_vertices,
	std::vector<std::vector<unsigned int>>& triangle_indices,
	std::vector<VoxelMatrix>& voxels, float& voxel_size){

	cv::FileStorage savefile;
	savefile.open(filename, cv::FileStorage::READ);

	cv::FileNode bpdNode = savefile["bodypartdefinitions"];
	bpdv.clear();
	for (auto it = bpdNode.begin(); it != bpdNode.end(); ++it)
	{
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}

	cv::FileNode frameNode = savefile["frame_datas"];
	frame_datas.clear();
	for (auto it = frameNode.begin(); it != frameNode.end(); ++it){
		cv::Mat camera_pose, camera_matrix;
		SkeletonNodeHard root;
		int facing;
		(*it)["camera_extrinsic"] >> camera_pose;
		(*it)["camera_intrinsic_mat"] >> camera_matrix;
		(*it)["skeleton"] >> root;
		(*it)["facing"] >> facing;
		FrameDataProcessed frame_data(bpdv.size(), 0, 0, camera_matrix, camera_pose, root);
		frame_data.mnFacing = facing;
		frame_datas.push_back(frame_data);
	}

	cv::FileNode clusterNode = savefile["bodypart_frame_cluster"];
	bodypart_frame_cluster.clear();
	bodypart_frame_cluster.resize(bpdv.size());
	for (auto it = clusterNode.begin(); it != clusterNode.end(); ++it){
		int bodypart;
		(*it)["bodypart"] >> bodypart;
		cv::FileNode clusterClusterNode = (*it)["clusters"];
		for (auto it2 = clusterClusterNode.begin(); it2 != clusterClusterNode.end(); ++it2){
			int main_frame;
			(*it2)["main_frame"] >> main_frame;
			std::vector<int> cluster;
			cluster.push_back(main_frame);
			bodypart_frame_cluster[bodypart].push_back(cluster);

			CroppedMat image;
			if ((*it2)["image"].empty()){
				std::string image_path;
				(*it2)["image_path"] >> image_path;
				image.mMat = cv::imread(image_path);
				(*it2)["image_offset"] >> image.mOffset;
				(*it2)["image_size"] >> image.mSize;
			}
			else{
				(*it2)["image"] >> image;
			}

			frame_datas[main_frame].mBodyPartImages.resize(bpdv.size());
			frame_datas[main_frame].mBodyPartImages[bodypart] = image;
		}
	}

	cv::FileNode vertNode = savefile["triangle_vertices"];
	triangle_vertices.clear();
	for (auto it = vertNode.begin(); it != vertNode.end(); ++it){
		triangle_vertices.push_back(std::vector<float>());
		for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
			float vert;
			(*it2) >> vert;
			triangle_vertices.back().push_back(vert);
		}
	}


	cv::FileNode indNode = savefile["triangle_indices"];
	triangle_indices.clear();
	for (auto it = indNode.begin(); it != indNode.end(); ++it){
		triangle_indices.push_back(std::vector<unsigned int>());
		for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
			int ind;
			(*it2) >> ind;
			triangle_indices.back().push_back(ind);
		}
	}

	cv::FileNode voxNode = savefile["voxels"];
	voxels.clear();
	for (auto it = voxNode.begin(); it != voxNode.end(); ++it){
		int width, height, depth;
		(*it)["width"] >> width;
		(*it)["height"] >> height;
		(*it)["depth"] >> depth;
		voxels.push_back(VoxelMatrix(width,height,depth));
	}

	savefile["voxel_size"] >> voxel_size;

	savefile.release();
}


int main(int argc, char ** argv){

	for (int i = 1; i < argc; ++i){
		if (strcmp(argv[i], "-d") == 0){
			video_directory = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-v") == 0){
			voxel_recons_path = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-n") == 0){
			numframes = atoi(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-s") == 0){
			skip_side = true;
			++i;
		}
		else if (strcmp(argv[i], "-t") == 0){
			tsdf_offset = atof(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-e") == 0){
			extension = std::string(argv[i + 1]);
			++i;
		}
		else{
			std::cout << "Options: -d [video directory] -v [voxel path] -n [num frames] -t [tsdf offset] -e [extension]\n"
				<< "-s: skip non-front and back frames\n";
			return 0;
		}
	}

	if (video_directory == ""){
		std::cout << "Specify video directory!\n";
		return 0;
	}

	if (voxel_recons_path == ""){
		std::cout << "Specify voxel path!\n";
		return 0;
	}

	std::string target = video_directory + "/packaged.yml";

	std::vector<VoxelMatrix> voxels;
	float voxel_size;

	load_packaged_file(target, bpdv, frame_datas, bodypart_frame_cluster, triangle_vertices, triangle_indices, voxels, voxel_size);
}