# Euler-SYCL

## sycl for cuda based on intel/llvm/clang++

1. [intel oneapi 2023.0.0](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline) && [codeplay Solutions for Nvidia and AMD](https://codeplay.com/solutions/oneapi/)
2. use clang++ in llvm-bin as compiler && some compiling options in /CMakeList.txt and /cmake/init_cuda/hip/host/intel.cmake, environment set below:

    ````bash
    source /opt/intel/oneapi/setvars.sh  --force --include-intel-llvm
    ````

3. device discovery:

    ````cmd
    $sycl-ls
    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2022.15.12.0.01_081451]
    [opencl:cpu:1] Intel(R) OpenCL, AMD Ryzen 7 5800X 8-Core Processor              3.0 [2022.15.12.0.01_081451]
    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA T600 0.0 [CUDA 11.5]
    ````

    ````C++
    auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
    sycl::queue q(device);
    ````

    set device_id=1 for selecting host as device and throwing mission to AMD Ryzen 7 5800X 8-Core Processor
    set device_id=2 for selecting nvidia GPU as device and throwing mission to NVIDIA T600

4. MPI libs

    ````cmd
    export MPI_PATH=/path/to/mpi
    ````

    libmpi.so located in $MPI_PATH/lib/libmpi.so

## .ini file arguments

- ### [run] parameters

    |name of parameters|function|type|default value|
    |:-----------------|:------:|:--:|:------------|
    |StartTime|begin time of the caculation|float|0.0f|
    |OutputDir|where to output result file|string|"./"|
    |OutBoundary|if output boundary piont|bool|flase|
    |nStepMax|max number of steps for evolution loop|int|10|
    |nOutMax|max number of files outputted for evolution loop|int|0|
    |OutInterval|interval number of steps for once output|int|nStepMax|
    |OutTimeInterval|time interval of once ouptut|float|0.0|
    |nOutTimeStamps|number of time interval|int|1|
    |DtBlockSize|1D local_ndrange parameter used in GetDt|int|4|
    |blockSize_x|X direction local_ndrange parameter used in SYCL lambda function|int|BlSz.BlockSize|
    |blockSize_y|Y direction local_ndrange parameter used in SYCL lambda function|int|BlSz.BlockSize|
    |blockSize_z|Z direction local_ndrange parameter used in SYCL lambda function|int|BlSz.BlockSize|

- ### [mpi] parameters

    |name of parameters|function|type|default value|
    |:-----------------|:------:|:--:|:------------|
    |mx|number of MPI threads at X direction in MPI Cartesian space|int|1|
    |my|number of MPI threads at Y direction in MPI Cartesian space|int|1|
    |mz|number of MPI threads at Z direction in MPI Cartesian space|int|1|

- ### [mesh] parameters

    |name of parameters|function|type|default value|
    |:-----------------|:------:|:--:|:------------|
    |DOMAIN_length|size of the longest edge of the domain|float|1.0|
    |xmin|starting coordinate at X direction of the domain|float|0.0|
    |ymin|starting coordinate at Y direction of the domain|float|0.0|
    |zmin|starting coordinate at Z direction of the domain|float|0.0|
    |X_inner|resolution setting at X direction of the domain|int|1|
    |Y_inner|resolution setting at Y direction of the domain|int|1|
    |Z_inner|resolution setting at Z direction of the domain|int|1|
    |Bwidth_X|number of ghost cells at X direction's edge of the domain|int|4|
    |Bwidth_Y|number of ghost cells at Y direction's edge of the domain|int|4|
    |Bwidth_Z|number of ghost cells at Z direction's edge of the domain|int|4|
    |CFLnumber|CFL number for advancing in time|float|0.6|
    |boundary_xmin|type of Boundary at xmin edge of the domain,influce values of ghost cells|int|2|
    |boundary_xmax|type of Boundary at xmax edge of the domain,influce values of ghost cells|int|2|
    |boundary_ymin|type of Boundary at ymin edge of the domain,influce values of ghost cells|int|2|
    |boundary_ymax|type of Boundary at ymax edge of the domain,influce values of ghost cells|int|2|
    |boundary_zmin|type of Boundary at zmin edge of the domain,influce values of ghost cells|int|2|
    |boundary_zmax|type of Boundary at zmax edge of the domain,influce values of ghost cells|int|2|

- ### [init] parameters

    |name of parameters|function|type|default value|
    |:-----------------|:------:|:--:|:------------|
    |blast_type|type of blast in domain|int|0|
    |blast_center_x|position of the blast at X direction in domain|float|0.0|
    |blast_center_y|position of the blast at Y direction in domain|float|0.0|
    |blast_center_z|position of the blast at Z direction in domain|float|0.0|
    |blast_radius|radius ratio of shortest edge of domain of blast|float|0.0|
    |blast_density_in|rho of the fluid upstream the blast|float|0.0|
    |blast_density_out|rho of the fluid downstream the blast|float|0.0|
    |blast_pressure_in|P of the fluid upstream the blast|float|0.0|
    |blast_pressure_out|P of the fluid downstream the blast|float|0.0|
    |blast_u_in|u of the fluid upstream the blast|float|0.0|
    |blast_v_in|v of the fluid upstream the blast|float|0.0|
    |blast_w_in|v of the fluid upstream the blast|float|0.0|
    |blast_u_out|u of the fluid downstream the blast|float|0.0|
    |blast_v_out|v of the fluid downstream the blast|float|0.0|
    |blast_w_out|w of the fluid downstream the blast|float|0.0|
    |cop_type|type of compoent area in domain|int|0|
    |cop_center_x|position of compoent area at X direction in domain|float|0.0|
    |cop_center_y|position of compoent area at X direction in domain|float|0.0|
    |cop_center_z|position of compoent area at X direction in domain|float|0.0|
    |cop_radius|radius ratio of shortest edge of domain of compoent area|float|0.0|
    |cop_density_in|rho of the fluid in compoent area|float|blast_density_out|
    |cop_pressure_in|P of the fluid in compoent area|float|blast_pressure_out|
