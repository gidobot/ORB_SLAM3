echo "Configuring and building Thirdparty/DBoW2 ..."
cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../g2o
echo "Configuring and building Thirdparty/g2o ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../CudaSift
echo "Configuring and building Thirdparty/CudaSift ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../viso2
echo "Configuring and building Thirdparty/viso2 ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../Sophus
echo "Configuring and building Thirdparty/Sophus ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../DescNet
echo "Configuring and building Thirdparty/DescNet ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../tensorrt-cpp-api
echo "Configuring and building Thirdparty/tensorrt-cpp-api ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../../

echo "Configuring and building SIFT_SLAM3 ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
