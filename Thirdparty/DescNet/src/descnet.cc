#include "descnet.h"

using namespace std;

namespace descnet
{
	DescNet::DescNet(const string &model_path) : model(model_path) {}

	void DescNet::compute(std::vector<cv::KeyPoint> &kpts) {

	}

	void DescNet::getPatchParams(std::vector<cv::KeyPoint> &kpts, const cv::Mat &img, cv::Mat &patch_param) {
		patch_param = cv::Mat(kpts.size(), 6, CV_32F, 0.0);
		float patch_scale = 6.f;
		float m_cos, m_sin, angle, x, y, size, octave, short_side;
		short_side = (float) (min(img.rows, img.cols));

		for (int i=0; i<kpts.size(); i++) {
			x = (kpts[i].pt.x - img.cols/2) / (img.cols/2);
			y = (kpts[i].pt.y - img.rows/2) / (img.rows/2);
			size = kpts[i].size;
			angle = (360 - kpts[i].angle) * (M_PI / 180);
			octave =  kpts[i].octave & 0xFF;
			m_cos = cos(angle) * patch_scale * size;
			m_sin = sin(angle) * patch_scale * size;

			patch_param.at<float>(i,0) = m_cos / short_side;
			patch_param.at<float>(i,1) = m_sin / short_side;
			patch_param.at<float>(i,2) = x;
			patch_param.at<float>(i,3) = -m_sin / short_side;
			patch_param.at<float>(i,4) = m_cos / short_side;
			patch_param.at<float>(i,5) = y;
		}
	}

	void DescNet::getPatches(std::vector<cv::KeyPoint> &kpts, const cv::Mat &img, cv::Mat &all_patches) {
		int patch_size = 32;
		all_patches = cv::Mat(); 
		int npixel = patch_size*patch_size;
		cv::Mat output_grid = cv::Mat(npixel, 3, CV_32F, 0.0);
		for (int i=0; i<npixel; i++) {
			output_grid.at<float>(i,0) = (i % patch_size) * 1. / patch_size * 2 - 1;
			output_grid.at<float>(i,1) = floor(i / patch_size) * 1. / patch_size * 2 - 1;
			output_grid.at<float>(i,2) = 1;
		}

		float size, ori, radius, m_cos, m_sin;
		std::vector<float> ptf = {0,0};
		int bs = 30; // limited by OpenCV remap implementation using int16 for coordinates
		// int bs = 4000; // limited by OpenCV remap implementation using int16 for coordinates
		float sift_descr_scl_fctr = 3;
		float sift_descr_width = 4;
		float max_rad = sqrt(img.cols*img.cols + img.rows*img.rows);
		cv::Mat affine_mat = cv::Mat(3, 2, CV_32F, 0.0);
		cv::Mat input_grid;
		std::vector<cv::Mat> batch_input_grid;
		cv::Mat batch_input_grid_;
		cv::Mat patches, patch;
		cv::Scalar mean, stddev;
		for (int i=0; i<kpts.size(); i++) {
			size = kpts[i].size * 0.5;
			ptf = {kpts[i].pt.x, kpts[i].pt.y};
			ori = (360. - kpts[i].angle) * (M_PI / 180.);
			radius = round(sift_descr_scl_fctr * size * sqrt(2) * (sift_descr_width + 1) * 0.5);
			radius = min(radius, max_rad);
			// construct affine transformation matrix
			m_cos = cos(ori) * radius;
			m_sin = sin(ori) * radius;
			affine_mat.at<float>(0,0) = m_cos;
			affine_mat.at<float>(1,0) = m_sin;
			affine_mat.at<float>(2,0) = ptf[0];
			affine_mat.at<float>(0,1) = -m_sin;
			affine_mat.at<float>(1,1) = m_cos;
			affine_mat.at<float>(2,1) = ptf[1];
			// get input grid
			input_grid = output_grid * affine_mat;
			batch_input_grid.push_back(input_grid.clone());
			cv::Mat batch_points_, batch_chans_[2];
			if (batch_input_grid.size() != 0 and batch_input_grid.size() % bs == 0 or i == kpts.size() - 1) {
				cv::vconcat(batch_input_grid, batch_input_grid_);
				batch_input_grid_ = batch_input_grid_.reshape(2, {batch_input_grid_.rows, 1});
				cv::remap(img, patches, batch_input_grid_, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(0.0, 0.0, 0.0));

				// standardize patches
				if (true) {
					for (int i=0; i < batch_input_grid.size(); i++) {
						int idx = i*32*32;
						patch = patches({cv::Range(idx, idx+32*32), cv::Range::all()});
						cv::meanStdDev(patch, mean, stddev);
						patch = (patch - mean) / stddev;
						// std::cout << "Patch mean: " << mean << std::endl;
						// std::cout << "Patch stddev: " << stddev << std::endl;
					}

				// 	// subtract patch means and divide by standard deviation
				// 	cv::reduce(patches,patch_mean, 2, cv::REDUCE_AVG);
				// 	cv::reduce(patch_mean,patch_mean, 1, cv::REDUCE_AVG);
				// 	// cv::reduce(patches,patch_std, 2, cv::REDUCE_STD);
				// 	// cv::reduce(patch_std,patch_std, 1, cv::REDUCE_AVG);
				// 	// patches = (patches - patch_mean) / patch_std;
				// 	// cv::reduce(patches,patch_mean, 2, cv::CV_REDUCE_AVG);
				// 	// cv::reduce(patch_mean,patch_mean, 1, cv::CV_REDUCE_AVG);
				// 	// std::cout << "Patch mean: " << patch_mean << std::endl;
				// 	// cv::reduce(patch_mean,patch_mean, 1, cv::CV_REDUCE_AVG);
				// 	std::cout << "Patch mean: " << patches.size() << std::endl;
				}

				if (all_patches.empty()) {
					all_patches = patches.clone();
					// cv::Mat patch_tmp = all_patches.reshape(1, {batch_input_grid.size(), patch_size, patch_size});
					// cout << "Patch: " << patch_tmp({cv::Range(0,1), cv::Range::all(), cv::Range::all()}).reshape(1, {patch_size, patch_size});
			  		// cv::namedWindow("Patch", cv::WINDOW_NORMAL);
			  		// cv::imshow("Patch", patch_tmp({cv::Range(0,1), cv::Range::all(), cv::Range::all()}).reshape(1, {patch_size, patch_size}));
			  		// cv::waitKey(0);
			  	}
				else
					cv::vconcat(all_patches, patches, all_patches);

				batch_input_grid.clear();
			}
		}
		if (all_patches.size[0] < model.getBatchSize()*patch_size*patch_size) {
			// append zeros to to make the batch size
			const int size[3] = {model.getBatchSize()*patch_size*patch_size - all_patches.size[0], 1};
			cv::Mat zeros = cv::Mat::zeros(2, size, all_patches.type());
			cv::vconcat(all_patches, zeros, all_patches);
		}
		// all_patches = all_patches.reshape(1, {kpts.size(), patch_size, patch_size});
		all_patches = all_patches.reshape(1, {model.getBatchSize(), patch_size, patch_size});
	}

	void DescNet::getPatchesCuda(std::vector<cv::KeyPoint> &kpts, const cv::Mat &img, cv::Mat &all_patches) {
		cv::cuda::GpuMat img_cuda(img);

		int patch_size = 32;
		all_patches = cv::Mat(); 
		int npixel = patch_size*patch_size;
		cv::Mat output_grid = cv::Mat(npixel, 3, CV_32F, 0.0);
		for (int i=0; i<npixel; i++) {
			output_grid.at<float>(i,0) = (i % patch_size) * 1. / patch_size * 2 - 1;
			output_grid.at<float>(i,1) = floor(i / patch_size) * 1. / patch_size * 2 - 1;
			output_grid.at<float>(i,2) = 1;
		}

		float size, ori, radius, m_cos, m_sin;
		std::vector<float> ptf = {0,0};
		int bs = 64; // limited by OpenCV remap implementation using int16 for coordinates
		// int bs = 4000; // limited by OpenCV remap implementation using int16 for coordinates
		float sift_descr_scl_fctr = 3;
		float sift_descr_width = 4;
		float max_rad = sqrt(img.cols*img.cols + img.rows*img.rows);
		cv::Mat affine_mat = cv::Mat(3, 2, CV_32F, 0.0);
		cv::Mat input_grid;
		std::vector<cv::Mat> batch_input_grid;
		cv::Mat batch_input_grid_;
		cv::cuda::GpuMat batch_input_x_cuda_, batch_input_y_cuda_;
		cv::cuda::GpuMat patches, patch;
		cv::Mat patches_cpu;
		cv::Scalar mean, stddev;
		for (int i=0; i<kpts.size(); i++) {
			size = kpts[i].size * 0.5;
			ptf = {kpts[i].pt.x, kpts[i].pt.y};
			ori = (360. - kpts[i].angle) * (M_PI / 180.);
			radius = round(sift_descr_scl_fctr * size * sqrt(2) * (sift_descr_width + 1) * 0.5);
			radius = min(radius, max_rad);
			// construct affine transformation matrix
			m_cos = cos(ori) * radius;
			m_sin = sin(ori) * radius;
			affine_mat.at<float>(0,0) = m_cos;
			affine_mat.at<float>(1,0) = m_sin;
			affine_mat.at<float>(2,0) = ptf[0];
			affine_mat.at<float>(0,1) = -m_sin;
			affine_mat.at<float>(1,1) = m_cos;
			affine_mat.at<float>(2,1) = ptf[1];
			// get input grid
			input_grid = output_grid * affine_mat;
			batch_input_grid.push_back(input_grid.clone());
			if (batch_input_grid.size() != 0 and batch_input_grid.size() % bs == 0 or i == kpts.size() - 1) {
				cv::vconcat(batch_input_grid, batch_input_grid_);
				batch_input_x_cuda_.upload(batch_input_grid_(cv::Range::all(), cv::Range(0,1)));
				batch_input_y_cuda_.upload(batch_input_grid_(cv::Range::all(), cv::Range(1,2)));
				cv::cuda::remap(img_cuda, patches, batch_input_y_cuda_, batch_input_y_cuda_, cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(0.0, 0.0, 0.0));

				// standardize patches
				if (true) {
					for (int i=0; i < batch_input_grid.size(); i++) {
						int idx = i*32*32;
						patch = patches(cv::Range(idx, idx+32*32), cv::Range::all());
						cv::cuda::meanStdDev(patch, mean, stddev);
						cv::cuda::subtract(patch, mean, patch);
						cv::cuda::divide(patch, stddev, patch);
						// patch = (patch - mean) / stddev;
					}

				// 	// subtract patch means and divide by standard deviation
				// 	cv::reduce(patches,patch_mean, 2, cv::REDUCE_AVG);
				// 	cv::reduce(patch_mean,patch_mean, 1, cv::REDUCE_AVG);
				// 	// cv::reduce(patches,patch_std, 2, cv::REDUCE_STD);
				// 	// cv::reduce(patch_std,patch_std, 1, cv::REDUCE_AVG);
				// 	// patches = (patches - patch_mean) / patch_std;
				// 	// cv::reduce(patches,patch_mean, 2, cv::CV_REDUCE_AVG);
				// 	// cv::reduce(patch_mean,patch_mean, 1, cv::CV_REDUCE_AVG);
				// 	// std::cout << "Patch mean: " << patch_mean << std::endl;
				// 	// cv::reduce(patch_mean,patch_mean, 1, cv::CV_REDUCE_AVG);
				// 	std::cout << "Patch mean: " << patches.size() << std::endl;
				}

				if (all_patches.empty()) {
					patches.download(all_patches);
					// cv::Mat patch_tmp = all_patches.reshape(1, {batch_input_grid.size(), patch_size, patch_size});
					// cout << "Patch: " << patch_tmp({cv::Range(0,1), cv::Range::all(), cv::Range::all()}).reshape(1, {patch_size, patch_size});
			  		// cv::namedWindow("Patch", cv::WINDOW_NORMAL);
			  		// cv::imshow("Patch", patch_tmp({cv::Range(0,1), cv::Range::all(), cv::Range::all()}).reshape(1, {patch_size, patch_size}));
			  		// cv::waitKey(0);
			  	}
				else {
					patches.download(patches_cpu);
					cv::vconcat(all_patches, patches_cpu, all_patches);
				}

				batch_input_grid.clear();
			}
		}
		if (all_patches.size[0] < model.getBatchSize()*patch_size*patch_size) {
			// append zeros to to make the batch size
			const int size[3] = {model.getBatchSize()*patch_size*patch_size - all_patches.size[0], 1};
			cv::Mat zeros = cv::Mat::zeros(2, size, all_patches.type());
			cv::vconcat(all_patches, zeros, all_patches);
		}
		// all_patches = all_patches.reshape(1, {kpts.size(), patch_size, patch_size});
		all_patches = all_patches.reshape(1, {model.getBatchSize(), patch_size, patch_size});
	}

	void DescNet::convertKpts(std::vector<cv::KeyPoint> &kpts, cv::Mat &npy_kpts) {
		npy_kpts = cv::Mat(kpts.size(), 5, CV_32F, 0.0);
		for (int i=0; i<kpts.size(); i++) {
			npy_kpts.at<float>(i, 0) = kpts[i].pt.x;
			npy_kpts.at<float>(i, 1) = kpts[i].pt.y;
			npy_kpts.at<float>(i, 2) = kpts[i].size;
			npy_kpts.at<float>(i, 3) = (360 - kpts[i].angle) * (M_PI / 180);
			npy_kpts.at<float>(i, 4) = kpts[i].octave & 0xFF;
		}	
	}

	void DescNet::unpackOctave(cv::KeyPoint &kpt, int &octave, int &layer, float &scale) {
		octave = kpt.octave & 255;
		layer  = (kpt.octave >> 8) & 255;
		if (octave >= 128) octave = -128 | octave;
		if (octave >= 0)
			scale = 1 / (1 << octave);
		else
			scale = (float)(1 << -octave);
	}

	void DescNet::computeFeatures(const cv::Mat& img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc)
    {
	    cv::Mat gimg;
	    bool isRGB = false;

		// float max_dim = 1024;	// for loc dense model, should be 1280, but not using dense model
		// float downsample_ratio = 1;
        float dim = max(img.rows, img.cols);
        int down_rows = img.rows;
        int down_cols = img.cols;
        cv::Mat img_ds, gimg_ds;

        // if (dim > max_dim) {
        	// downsample_ratio = max_dim/dim;
        	// down_rows = down_rows*downsample_ratio;
        	// down_cols = down_cols*downsample_ratio;
        // }

	    if(img.channels()==3)
	    {
	        if(isRGB)
	        {
	            cvtColor(img,gimg,cv::COLOR_RGB2GRAY);
	        }
	        else
	        {
	            cvtColor(img,gimg,cv::COLOR_BGR2GRAY);
	        }
	    }
	    else if(img.channels()==4)
	    {
	        if(isRGB)
	        {
	            cvtColor(img,gimg,cv::COLOR_RGBA2GRAY);
	        }
	        else
	        {
	            cvtColor(img,gimg,cv::COLOR_BGRA2GRAY);
	        }
	    }

	    gimg.convertTo(gimg, CV_32FC1);
	    // gimg.convertTo(gimg, CV_8UC1);

        time_point t1, t2;
        double tf;

        // cv::resize(img, img_ds, cv::Size(down_cols, down_rows), cv::INTER_LINEAR);

    	cv::Mat all_patches;
        // t1 = std::chrono::steady_clock::now();
    	getPatches(kpts, gimg, all_patches);
    	// cout << "All patches size: " << all_patches.size() << endl;
    	// getPatchesCuda(kpts, gimg, all_patches);
    	// all_patches.convertTo(all_patches, CV_32F);
        // t2 = std::chrono::steady_clock::now();
        // tf = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        // cout << "Patch extraction time: " << tf << endl;

    	// if (all_patches.size[0] < model.getBatchSize()) {
    	// 	// append zeros to to make the batch size
    	// 	const int size[3] = {model.getBatchSize() - all_patches.size[0], all_patches.size[1], all_patches.size[2]};
    	// 	cv::Mat zeros = cv::Mat::zeros(3, size, all_patches.type());
    	// 	cv::vconcat(all_patches, zeros, all_patches);
    	// }

        // t1 = std::chrono::steady_clock::now();
    	model.run(all_patches, desc);
        // t2 = std::chrono::steady_clock::now();
        // tf = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        // cout << "Feature description time: " << tf << endl;
	}

	void DescNet::computeFeatures(cv::Mat &all_patches, cv::Mat &desc)
    {
        // time_point t1, t2;
        // double tf;

        // t1 = std::chrono::steady_clock::now();
    	model.run(all_patches, desc);
        // t2 = std::chrono::steady_clock::now();
        // tf = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        // cout << "Feature description time: " << tf << endl;
	}

	void DescNet::computeFeaturesCUDA(const int batch_size, float *cuda_patches, cv::Mat &desc)
    {
        // time_point t1, t2;
        // double tf;

        // t1 = std::chrono::steady_clock::now();
    	model.runCUDA(batch_size, cuda_patches, desc);
        // t2 = std::chrono::steady_clock::now();
        // tf = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        // cout << "Feature description time: " << tf << endl;
	}

} // namespace descnet