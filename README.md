# PP-HumanSeg-opencv-onnxrun
分别使用OpenCV、ONNXRuntime部署PP-HumanSeg肖像分割模型，包含C++和Python两个版本的程序

百度PaddlePaddle团队发布的PP-HumanSeg人像分割模型，功能挺强的。
原始程序需要依赖PaddlePaddle框架才能运行，于是我编写了一套分别使用OpenCV、ONNXRuntime部署PP-HumanSeg肖像分割
的程序，彻底摆脱对任何深度学习框架的依赖，.onnx模型文件很小，只有5.9M，可以直接上传到github仓库里。
