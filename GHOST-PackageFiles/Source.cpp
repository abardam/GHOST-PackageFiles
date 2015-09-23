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


void save_packaged_file(std::string filename,
	const std::string& image_path,
	const BodyPartDefinitionVector& bpdv, 
	const std::vector<FrameDataProcessed>& frame_datas, 
	const BodypartFrameCluster& bodypart_frame_cluster, 
	const std::vector<std::vector<float>>& triangle_vertices, 
	const std::vector<std::vector<unsigned int>>& triangle_indices,
	const std::vector<VoxelMatrix>& voxels, float voxel_size,
	const std::vector<Cylinder>& cylinders){

	cv::FileStorage savefile;
	savefile.open(filename, cv::FileStorage::WRITE);

	savefile << "bodypartdefinitions" << "[";
	for (int i = 0; i < bpdv.size(); ++i)
	{
		savefile << (bpdv[i]);
	}
	savefile << "]";

	
	std::stringstream ss;

	//std::vector<std::vector<int>> bodypart_frame_saved(bpdv.size(), std::vector<int>(snhmaps.size, 0));

	savefile << "bodypart_frame_cluster" << "[";
	for (int i = 0; i < bodypart_frame_cluster.size(); ++i){
		savefile << "{";
		savefile << "bodypart" << i << "clusters" << "[";
		for (int j = 0; j < bodypart_frame_cluster[i].size(); ++j){

			if (bodypart_frame_cluster[i][j].empty()) continue;

			savefile << "{";

			int target_frame = bodypart_frame_cluster[i][j][0];

			savefile << "main_frame" << target_frame;

			ss.str("");
			ss << "bp" << i << "-frame" << target_frame << ".png";

			std::string full_path = image_path + "/" + ss.str();

			//bodypart_frame_saved[i][target_frame] = 1;

			cv::imwrite(full_path, frame_datas[target_frame].mBodyPartImages[i].mMat);
			savefile << "image_path" << ss.str();
			savefile << "image_offset" << frame_datas[target_frame].mBodyPartImages[i].mOffset;
			savefile << "image_size" << frame_datas[target_frame].mBodyPartImages[i].mSize;
			savefile << "}";
		}
		savefile << "]" << "}";
	}
	savefile << "]";

	savefile << "frame_datas" << "[";
	for (int i = 0; i < frame_datas.size(); ++i){
		savefile << "{";
		savefile << "camera_extrinsic" << frame_datas[i].mCameraPose;
		savefile << "camera_intrinsic_mat" << frame_datas[i].mCameraMatrix;
		savefile << "skeleton" << frame_datas[i].mRoot;
		savefile << "facing" << frame_datas[i].mnFacing;

		ss.str("");
		ss << "frame" << i << ".png";

		std::string full_path = image_path + "/" + ss.str();

		cv::imwrite(full_path, frame_datas[i].mBodyImage.mMat);

		savefile << "body_image_path" << ss.str();
		savefile << "body_image_offset" << frame_datas[i].mBodyImage.mOffset;
		savefile << "body_image_size" << frame_datas[i].mBodyImage.mSize;

		savefile << "}";
	}
	savefile << "]";

	savefile << "triangle_vertices" << "[";
	for (int i = 0; i < triangle_vertices.size(); ++i){
		savefile << "[";
		for (int j = 0; j < triangle_vertices[i].size(); ++j){
			savefile << triangle_vertices[i][j];
		}
		savefile << "]";
	}
	savefile << "]";

	savefile << "triangle_indices" << "[";
	for (int i = 0; i < triangle_indices.size(); ++i){
		savefile << "[";
		for (int j = 0; j < triangle_indices[i].size(); ++j){
			savefile << (int)(triangle_indices[i][j]);
		}
		savefile << "]";
	}
	savefile << "]";

	savefile << "voxels" << "[";
	for (int i = 0; i < voxels.size(); ++i){
		savefile << "{";
		savefile << "width" << voxels[i].width
			<<"height" << voxels[i].height
			<<"depth" << voxels[i].depth
			<<"}";
	}
	savefile << "]";

	savefile << "voxel_size" << voxel_size;

	savefile << "cylinders" << "[";
	for (int i = 0; i < cylinders.size(); ++i){
		savefile << "{"
			<< "width" << cylinders[i].width
			<< "height" << cylinders[i].height
			<< "}";
	}
	savefile << "]";

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


	std::stringstream filenameSS;
	int startframe = 0;

	cv::FileStorage fs;

	filenameSS << video_directory << "/bodypartdefinitions" << extension;

	fs.open(filenameSS.str(), cv::FileStorage::READ);
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}
	fs.release();
	std::vector<std::string> filenames;

	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << extension;

		filenames.push_back(filenameSS.str());

	}

	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;

	load_processed_frames(filenames, extension, bpdv.size(), frame_datas, false);

	for (int i = 0; i < frame_datas.size(); ++i){
		snhmaps.push_back(SkeletonNodeHardMap());
		cv_draw_and_build_skeleton(&frame_datas[i].mRoot, cv::Mat::eye(4, 4, CV_32F), frame_datas[i].mCameraMatrix, frame_datas[i].mCameraPose, &snhmaps[i]);
	}

	bodypart_frame_cluster = cluster_frames(64, bpdv, snhmaps, frame_datas, 1000);


	load_voxels(voxel_recons_path, cylinders, voxels, TSDF_array, weight_array, voxel_size);

	triangle_vertices.resize(bpdv.size());
	triangle_indices.resize(bpdv.size());
	triangle_colors.resize(bpdv.size());

	double num_vertices = 0;

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<TRIANGLE> tri_add;

		cv::add(tsdf_offset * cv::Mat::ones(TSDF_array[i].rows, TSDF_array[i].cols, CV_32F), TSDF_array[i], TSDF_array[i]);

		if (TSDF_array[i].empty()){
			tri_add = marchingcubes_bodypart(voxels[i], voxel_size);
		}
		else{
			tri_add = marchingcubes_bodypart(voxels[i], TSDF_array[i], voxel_size);
		}
		std::vector<cv::Vec4f> vertices;
		std::vector<unsigned int> vertex_indices;
		for (int j = 0; j < tri_add.size(); ++j){
			for (int k = 0; k < 3; ++k){
				cv::Vec4f candidate_vertex = tri_add[j].p[k];

				bool vertices_contains_vertex = false;
				int vertices_index;
				for (int l = 0; l < vertices.size(); ++l){
					if (vertices[l] == candidate_vertex){
						vertices_contains_vertex = true;
						vertices_index = l;
						break;
					}
				}
				if (!vertices_contains_vertex){
					vertices.push_back(candidate_vertex);
					vertices_index = vertices.size() - 1;
				}
				vertex_indices.push_back(vertices_index);
			}
		}
		triangle_vertices[i].reserve(vertices.size() * 3);
		triangle_colors[i].reserve(vertices.size() * 3);
		triangle_indices[i].reserve(vertex_indices.size());
		for (int j = 0; j < vertices.size(); ++j){
			triangle_vertices[i].push_back(vertices[j](0));
			triangle_vertices[i].push_back(vertices[j](1));
			triangle_vertices[i].push_back(vertices[j](2));
			triangle_colors[i].push_back(bpdv[i].mColor[0] * 255);
			triangle_colors[i].push_back(bpdv[i].mColor[1] * 255);
			triangle_colors[i].push_back(bpdv[i].mColor[2] * 255);
		}
		num_vertices += vertices.size();
		for (int j = 0; j < vertex_indices.size(); ++j){
			triangle_indices[i].push_back(vertex_indices[j]);
		}
	}

	save_packaged_file(video_directory + "/packaged.yml", video_directory, bpdv, frame_datas, bodypart_frame_cluster, triangle_vertices, triangle_indices, voxels, voxel_size, cylinders);
}