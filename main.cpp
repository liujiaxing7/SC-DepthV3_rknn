#include <iostream>
//#include "npu_rk/model_npu_rk.h"
#include <opencv2/highgui.hpp>
#include<sys/time.h>
#include <ctime>
#include <unistd.h>
#include <rknn_api.h>
#include <opencv2/imgproc.hpp>
#include "fstream"

void ReadFile(std::string srcFile, std::vector<std::string> &image_files) {

    if (not access(srcFile.c_str(), 0) == 0) {
        return;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open()) {
        exit(0);
    }

    std::string s;
    while (getline(fin, s)) {
        image_files.push_back(s);
    }

    fin.close();
}
bool Read(const char *filename, unsigned char **data, int &size) {
    *data = nullptr;
    ::FILE *fp;
    const int offset = 0;
    int ret = 0;
    unsigned char *dataTemp;

    fp = fopen(filename, "rb");

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);

    ret = fseek(fp, offset, SEEK_SET);

    dataTemp = (unsigned char *) malloc(size);
    ret = fread(dataTemp, 1, size, fp);

    *data = dataTemp;
    fclose(fp);

    return true;

    exit:
    return false;
}
int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << std::endl;
        return -1;
    }

    int img_w = 640;
    int img_h = 192;

    cv::Mat dst;
//    auto model = new ModelNpuRK();
//    model->InitContext("/root/ljdong/ljx/depth_for_rknn/monodepth.rknn");
    rknn_context ctx;

    //load_time
    float load_time_use = 0;
    struct timeval load_start;
    struct timeval load_end;
    gettimeofday(&load_start,NULL);

    std::string modelFile =  argv[1];
    int modelSize = 0;
    unsigned char *data;
    int ret = Read(modelFile.c_str(), &data, modelSize);
    ret = rknn_init(&ctx, data, modelSize, 0, NULL);

    gettimeofday(&load_end,NULL);
    load_time_use=(load_end.tv_sec-load_start.tv_sec)*1000000+(load_end.tv_usec-load_start.tv_usec);
    std::cout<<"load model time : "<<load_time_use/1000.0<<"ms"<<std::endl;

    //加载图像
    std::string imagesTxt = argv[2];
    std::vector<std::string> imageNameList;
    ReadFile(imagesTxt, imageNameList);
    const size_t size = imageNameList.size();

    for (size_t imgid = 0; imgid < size; ++imgid)
    {
        auto imageName = imageNameList.at(imgid);
        cv::Mat image_in = cv::imread(imageName);

        cv::resize(image_in, dst, cv::Size(img_w
                , img_h));

        using TYPE = uint8_t;
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
        TYPE *ptr = (TYPE *) malloc(dst.rows * dst.cols * dst.channels() * sizeof(TYPE));
        memcpy(ptr, dst.data , dst.rows * dst.cols * dst.channels() * sizeof(TYPE));

        //    bool flag = false;
        //    flag = model->UploadTensor(ptr);
        //    flag = model->Forward();
        //    flag = model->DownloadTensor(output);
        //
        //    model->GetOutputSize(outputSize);

        //data_to gpu
        float data_to_gpu = 0;
        struct timeval data_gpu_start;
        struct timeval data_gpu_end;
        gettimeofday(&data_gpu_start,NULL);

        rknn_input inputs[1];
        rknn_output outputs[1];
        inputs[0].buf = ptr;
        inputs[0].index = 0;
        inputs[0].size = dst.cols * dst.rows * dst.channels();
        inputs[0].pass_through = false;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;

        ret = rknn_inputs_set(ctx, 1, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            if(ctx > 0)         rknn_destroy(ctx);
            return ret;
        }
        gettimeofday(&data_gpu_end,NULL);
        data_to_gpu=(data_gpu_end.tv_sec-data_gpu_start.tv_sec)*1000000+(data_gpu_end.tv_usec-data_gpu_start.tv_usec);
        std::cout<<"data_to_gpu time : "<<data_to_gpu/1000.0<<"ms"<<std::endl;


        //forward time
        float forward_time_use = 0;
        struct timeval forward_start;
        struct timeval forward_end;
        gettimeofday(&forward_start,NULL);
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            if(ctx > 0)         rknn_destroy(ctx);
            return ret;
        }
        gettimeofday(&forward_end,NULL);
        forward_time_use=(forward_end.tv_sec-forward_start.tv_sec)*1000000+(forward_end.tv_usec-forward_start.tv_usec);
        std::cout<<"forward time : "<<forward_time_use/1000.0<<"ms"<<std::endl;


        //data to cpu time
        float data_to_cpu = 0;
        struct timeval data_cpu_start;
        struct timeval data_cpu_end;
        gettimeofday(&data_cpu_start,NULL);


        outputs[0].want_float = true;
        outputs[0].is_prealloc = false;
        ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            if(ctx > 0)         rknn_destroy(ctx);
            return ret;
        }
        std::vector<float*> network_outputs {
                (float*)outputs[0].buf,
        };

        gettimeofday(&data_cpu_end,NULL);
        data_to_cpu=(data_cpu_end.tv_sec-data_cpu_start.tv_sec)*1000000+(data_cpu_end.tv_usec-data_cpu_start.tv_usec);
        std::cout<<"data_to_cpu time : "<<data_to_cpu/1000.0<<"ms"<<std::endl;

        //可视化图像
        int outSize_w = dst.cols;
        int outSize_h = dst.rows;
        cv::Mat outimg;
        outimg.create(cv::Size(outSize_w, outSize_h), CV_32FC1);

        cv::Mat showImg;

        for (int i=0; i<outSize_h; ++i) {
            {
                for (int j=0; j<outSize_w; ++j)
                {
                    //                std::cout<<"index: " << i*outSize_w+j <<std::endl;
                    //                std::cout<<"value:" <<network_outputs[0][i*outSize_w+j]<<std::endl;
                    outimg.at<float>(i,j) = network_outputs[0][i*outSize_w+j];
                }
            }
        }

        //可视化
        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(outimg,minp,maxp);
        float minvalue = (float)minv;
        float maxvalue = (float)maxv;

        for (int i=0; i<outSize_h; ++i) {
            {
                for (int j=0; j<outSize_w; ++j)
                {

                    outimg.at<float>(i,j) = 255* (outimg.at<float>(i,j) - minvalue)/(maxvalue-minvalue);
                }
            }
        }

        outimg.convertTo(showImg,CV_8U);
        cv::Mat colorimg;
        cv::Mat colorimgfinal;
        //        cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)
        cv::convertScaleAbs(showImg,colorimg);
        cv::applyColorMap(colorimg,colorimgfinal,cv::COLORMAP_PARULA);
//        cv::applyColorMap(colorimg,colorimgfinal,cv::COLORMAP_HOT);
        //    namedWindow("image", cv::WINDOW_AUTOSIZE);
        //    imshow("image", colorimgfinal);
        cv::imwrite("../result_"+std::to_string(imgid)+".png",colorimgfinal);
    }

    return 0;
}
