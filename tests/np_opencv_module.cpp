// Author: Sudeep Pillai (spillai@csail.mit.edu)
// License: BSD
// Secound Author Chakkrit Termritthikun
// Last modified: Dec 1, 2015

// Wrapper for most external modules
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <exception>

// Opencv includes
#include <opencv2/opencv.hpp>

// np_opencv_converter
#include "np_opencv_converter.hpp"


#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <vector>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::gpu;

namespace py = boost::python;


template<class T>
void convertAndResize(const T& src, T& gray, T& resized, double scale){
    if (src.channels() == 3){
        cvtColor( src, gray, CV_BGR2GRAY );
    }
    else{
        gray = src;
    }
    
    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));
    
    if (scale != 1){
        resize(gray, resized, sz);
    }
    else{
        resized = gray;
    }
}

template <class T>
py::list toPythonList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}


py::list face_detection(const cv::Mat& in, const string cascadeName, const string cascadeNameGPU,const bool useGPU) {
    
    if (getCudaEnabledDeviceCount() == 0){
        cerr << "No GPU found or the library is compiled without GPU support" << endl;
    }
    
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
    
    
    CascadeClassifier_GPU cascade_gpu;
    if (!cascade_gpu.load(cascadeNameGPU)){
        cerr << "ERROR: Could not load cascade classifier "" << cascadeNameGPU << """ << endl;
    }
    
    
    CascadeClassifier cascade_cpu;
    if (!cascade_cpu.load(cascadeName)){
        cerr << "ERROR: Could not load cascade classifier "" << cascadeName << """ << endl;
    }
    
    
    Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, frameDisp;
    vector<Rect> facesBuf_cpu;
    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;
    
    vector<Rect> face_rects ;
    
    /* parameters */
    //bool useGPU = false;
    double scaleFactor = 1.0;
    bool findLargestObject = false;
    bool filterRects = true;
    bool helpScreen = false;
    int detections_num;
    
    in.copyTo(frame_cpu);
    frame_gpu.upload(in);
    convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
    convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
    
    TickMeter tm;
    tm.start();
    
    if (useGPU){
        //cascade_gpu.visualizeInPlace = true;
        cascade_gpu.findLargestObject = findLargestObject;
        
        detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu, 1.2,
        (filterRects || findLargestObject) ? 4 : 0);
        facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
    }
    else
    {
        Size minSize = cascade_gpu.getClassifierSize();
        cascade_cpu.detectMultiScale(resized_cpu, facesBuf_cpu, 1.2,
        (filterRects || findLargestObject) ? 4 : 0,
        (findLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0)
        | CV_HAAR_SCALE_IMAGE,
        minSize);
        detections_num = (int)facesBuf_cpu.size();
    }
    
    tm.stop();
    double detectionTime = tm.getTimeMilli();
    double fps = 1000 / detectionTime;
    //print detections to console
    cout << setfill(' ') << setprecision(2);
    cout << setw(6) << fixed << fps << " FPS, " << detections_num << " det";
    
    
    py::list facereturn;
    Rect *faces = useGPU ? faces_downloaded.ptr<Rect>() : &facesBuf_cpu[0];
    //Rect* faces = faces_downloaded.ptr<cv::Rect>();
    for (int i = 0; i < detections_num; ++i)
    {
        py::list points;
        points.append(faces[i].x);
        points.append(faces[i].y);
        points.append(faces[i].width);
        points.append(faces[i].height);
        facereturn.append(points);
        face_rects.push_back(faces[i]);
    }
    
    
    std::cerr << "detections_num: " << detections_num << std::endl;
    //std::cerr << "sz: " << in.size() << std::endl;
    return facereturn;
}

cv::Mat test_np_mat(const cv::Mat& in) {
    std::cerr << "in: " << in << std::endl;
    std::cerr << "sz: " << in.size() << std::endl;
    return in.clone();
}

cv::Mat test_with_args(const cv::Mat_<float>& in, const int& var1 = 1,
const double& var2 = 10.0, const std::string& name=std::string("test_name")) {
    std::cerr << "in: " << in << std::endl;
    std::cerr << "sz: " << in.size() << std::endl;
    std::cerr << "Returning transpose" << std::endl;
    return in.t();
}

class GenericWrapper {
    public:
    GenericWrapper(const int& _var_int = 1, const float& _var_float = 1.f,
    const double& _var_double = 1.d, const std::string& _var_string = std::string("test_string"))
    : var_int(_var_int), var_float(_var_float), var_double(_var_double), var_string(_var_string)
    {
        
    }
    
    cv::Mat process(const cv::Mat& in) {
        std::cerr << "in: " << in << std::endl;
        std::cerr << "sz: " << in.size() << std::endl;
        std::cerr << "Returning transpose" << std::endl;
        return in.t();
    }
    
    private:
    int var_int;
    float var_float;
    double var_double;
    std::string var_string;
};

// Wrap a few functions and classes for testing purposes
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
namespace fs { namespace python {
        
        BOOST_PYTHON_MODULE(np_opencv_module)
        {
            // Main types export
            fs::python::init_and_export_converters();
            py::scope scope = py::scope();
            
            // Basic test
            py::def("test_np_mat", &test_np_mat);
            
            py::def("face_detection", &face_detection);
            
            
            // With arguments
            py::def("test_with_args", &test_with_args,
            (py::arg("src"), py::arg("var1")=1, py::arg("var2")=10.0, py::arg("name")="test_name"));
            
            // Class
            py::class_<GenericWrapper>("GenericWrapper")
            .def(py::init<py::optional<int, float, double, std::string> >(
            (py::arg("var_int")=1, py::arg("var_float")=1.f, py::arg("var_double")=1.d,
            py::arg("var_string")=std::string("test"))))
            .def("process", &GenericWrapper::process)
            ;
        }
        
    } // namespace fs
} // namespace python
