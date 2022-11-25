#include "block_sycl.h"
#include "fun.h"

void InitializeFluidStates(sycl::queue &q, array<int, 3> WG, array<int, 3> WI, MaterialProperty *material, FlowData &fdata, Real* U, Real* U1, Real* LU,
                           Real* FluxF, Real* FluxG, Real* FluxH, Real* FluxFw, Real* FluxGw, Real* FluxHw, 
                           Real const dx, Real const dy, Real const dz)
{
	auto local_ndrange = range<3>(WG.at(0), WG.at(1), WG.at(2));	// size of workgroup
	auto global_ndrange = range<3>(Xmax, Ymax, Zmax);

	// auto local_ndrange = range<3>(WGSize.at(0), WGSize.at(1), WGSize.at(2));	// size of workgroup
	// auto global_ndrange = range<3>(WISize.at(0), WISize.at(1), WISize.at(2));

    // dim3 dim_grid_max(dim_grid.x + DIM_X, dim_grid.y + DIM_Y, dim_grid.z + DIM_Z);
    // InitialStatesKernel<<<dim_grid_max, dim_blk>>>(material, U, U1, LU, CnsrvU, CnsrvU1, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, 
    //                                                                                 fdata.u, fdata.v, fdata.w, fdata.rho, fdata.p, fdata.H, fdata.c, vof, dx, dy, dz);

	Real *rho = fdata.rho;
	Real *p = fdata.p;
	Real *H = fdata.H;
	Real *c = fdata.c;
	Real *u = fdata.u;
	Real *v = fdata.v;
	Real *w = fdata.w;

	q.submit([&](sycl::handler &h){
		// auto out = sycl::stream(1024, 256, h);
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    		// 利用get_global_id获得全局指标
    		int i = index.get_global_id(0) + Bwidth_X - 1;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			// int ii = index.get_group(0)*index.get_local_range(0) + index.get_local_id(0) + Bwidth_X - 1;
			// int jj = index.get_group(1)*index.get_local_range(1) + index.get_local_id(1) + Bwidth_Y;

			// SYCL不允许在lambda函数里出现结构体成员
			InitialStatesKernel(i, j, k, material, U, U1, LU, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, u, v, w, rho, p, H, c, dx, dy, dz);

			// testkernel(i,j,k, d_U, F, Fw, eigen, u,v,w, rho, p, H,c, 0.1, 0.1, 0.1);
		});
	});
}

Real GetDt(sycl::queue &q, FlowData &fdata, Real* uvw_c_max, Real const dx, Real const dy, Real const dz)
{
	Real *rho = fdata.rho;
	Real *c = fdata.c;
	Real *u = fdata.u;
	Real *v = fdata.v;
	Real *w = fdata.w;

	auto local_ndrange = range<1>(4);	// size of workgroup
	auto global_ndrange = range<1>(Xmax*Ymax*Zmax);
    
	for(int n=0; n<3; n++)
		uvw_c_max[n] = 0;

    q.submit([&](sycl::handler& h) {
      	// define reduction objects for sum, min, max reduction
		// auto reduction_sum = reduction(sum, sycl::plus<>());
    	auto reduction_max_x = reduction(&(uvw_c_max[0]), sycl::maximum<>());
		auto reduction_max_y = reduction(&(uvw_c_max[1]), sycl::maximum<>());
		auto reduction_max_z = reduction(&(uvw_c_max[2]), sycl::maximum<>());
      
		h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_x, reduction_max_y, reduction_max_z, 
	  	[=](nd_item<1> index, auto& temp_max_x, auto& temp_max_y, auto& temp_max_z)
		{
        	auto id = index.get_global_id();
			// if(id < Xmax*Ymax*Zmax)
        	temp_max_x.combine(u[id]+c[id]);
        	temp_max_y.combine(v[id]+c[id]);
			temp_max_z.combine(w[id]+c[id]);
      	});
    }).wait();

    // printf("maxuc = %f, %f, %f \n", h_uvw_c_max[0], h_uvw_c_max[1], h_uvw_c_max[2]);

	Real dtref = 0.0;
	#if DIM_X
	dtref += uvw_c_max[0]/dx;
	#endif
	#if DIM_Y
	dtref += uvw_c_max[1]/dy;
	#endif
	#if DIM_Z
	dtref += uvw_c_max[2]/dz;
	#endif
	return CFLnumber/dtref;
}

void UpdateFluidStateFlux(sycl::queue &q, Real*  UI, FlowData &fdata, Real*  FluxF, Real*  FluxG, Real*  FluxH, Real const Gamma)
{
	auto local_ndrange = range<3>(dim_block_x, dim_block_y, dim_block_z);	// size of workgroup
	auto global_ndrange = range<3>(Xmax, Ymax, Zmax);

	Real *rho = fdata.rho;
	Real *p = fdata.p;
	Real *H = fdata.H;
	Real *c = fdata.c;
	Real *u = fdata.u;
	Real *v = fdata.v;
	Real *w = fdata.w;

	q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);

			UpdateFuidStatesKernel(i, j, k, UI, FluxF, FluxG, FluxH, rho, p, c, H, u, v, w, Gamma);
		});
	});

	q.wait();
}

void UpdateURK3rd(sycl::queue &q, Real* U, Real* U1, Real* LU, Real const dt, int flag)
{
	auto local_ndrange = range<3>(dim_block_x, dim_block_y, dim_block_z);	// size of workgroup
	auto global_ndrange = range<3>(X_inner, Y_inner, Z_inner);

	q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0) + Bwidth_X;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			UpdateURK3rdKernel(i, j, k, U, U1, LU, dt, flag);
		});
	});

	q.wait();
}

void GetLU(sycl::queue &q, Real* UI, Real* LU, Real* FluxF, Real* FluxG, Real* FluxH, 
            Real* FluxFw, Real* FluxGw, Real* FluxHw, Real const Gamma, int const Mtrl_ind, 
            FlowData &fdata, Real* eigen_local, Real const dx, Real const dy, Real const dz)
{
	Real *rho = fdata.rho;
	Real *p = fdata.p;
	Real *H = fdata.H;
	Real *c = fdata.c;
	Real *u = fdata.u;
	Real *v = fdata.v;
	Real *w = fdata.w;

    bool is_3d = DIM_X*DIM_Y*DIM_Z ? true : false;

	auto local_ndrange = range<3>(dim_block_x, dim_block_y, dim_block_z);
	auto global_ndrange_max = range<3>(Xmax, Ymax, Zmax);
	auto global_ndrange_inner = range<3>(X_inner, Y_inner, Z_inner);

	#if DIM_X
	//proceed at x directiom and get F-flux terms at node wall
  	auto global_ndrange_x = range<3>(X_inner+local_ndrange[0], Y_inner, Z_inner);
	
	auto e = q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, 1.0, 0.0, 0.0, eigen_local, u, v, w, c);
		});
	});

	q.submit([&](sycl::handler &h){
		h.depends_on(e);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0) + Bwidth_X - 1;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;
			ReconstructFluxX(i, j, k, UI, FluxF, FluxFw, eigen_local, rho, u, v, w, H, dx);
		});
	}).wait();
	#endif

	// #if DIM_Y
	// //proceed at y directiom and get G-flux terms at node wall
    // GetLocalEigen<<<dim_grid_max, dim_blk>>>(0.0, 1.0, 0.0, eigen_local, fdata.u, fdata.v, fdata.w, fdata.c);
    // CheckCUDAErrors(cudaGetLastError());
    // // dim3 dim_grid_y(dim_grid.x, dim_grid.y + 1, dim_grid.z);
    // // ReconstructFluxY<<<dim_grid_y, dim_blk>>>(UI, FluxG, FluxGw, eigen_local, fdata.rho, fdata.u, fdata.v, fdata.w, fdata.H, dy);
    // blocksize = is_3d ?  dim3(WarpSize,WarpSize/2, 1)  : dim_blk;
    // gridsize = is_3d ? dim3(X_inner/blocksize.x, Y_inner/blocksize.y+1, Z_inner/blocksize.z) : dim3(dim_grid.x, dim_grid.y + 1, dim_grid.z);
    // ReconstructFluxY<<<gridsize, blocksize>>>(UI, FluxG, FluxGw, phi, tag_bnd_xt, Rgn_ind, Gamma, Mtrl_ind, eigen_local, fdata.rho, fdata.u, fdata.v, fdata.w, fdata.H, fdata.c, dy);
	// #endif

	// #if DIM_Z
	// //proceed at y directiom and get G-flux terms at node wall
    // GetLocalEigen<<<dim_grid_max, dim_blk>>>(0.0, 0.0, 1.0, eigen_local, fdata.u, fdata.v, fdata.w, fdata.c);
    // CheckCUDAErrors(cudaGetLastError());
    // // dim3 dim_grid_z(dim_grid.x, dim_grid.y, dim_grid.z + 1);
    // // ReconstructFluxZ<<<dim_grid_z, dim_blk>>>(UI, FluxH, FluxHw, eigen_local, fdata.rho, fdata.u, fdata.v, fdata.w, fdata.H, dz);
    // blocksize = is_3d ?  dim3(WarpSize/2, 1, WarpSize)  : dim_blk;
    // gridsize = is_3d ? dim3(X_inner/blocksize.x, Y_inner/blocksize.y, Z_inner/blocksize.z+1) : dim3(dim_grid.x, dim_grid.y, dim_grid.z + 1);
    // ReconstructFluxZ<<<gridsize, blocksize>>>(UI, FluxH, FluxHw, phi, tag_bnd_xt, Rgn_ind, Gamma, eigen_local, fdata.rho, fdata.u, fdata.v, fdata.w, fdata.H, dz);
	// #endif

	//update LU from cell-face fluxes
    #if NumFluid == 2
    // UpdateFluidMPLU<<<dim_grid, dim_blk>>>(LU, FluxFw, FluxGw, FluxHw, phi, tag_bnd_hlf, Rgn_ind, x_swth, y_swth, z_swth, exchange, dx, dy, dz);
    #else
	q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0) + Bwidth_X;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			UpdateFluidLU(i, j, k, LU, FluxFw, FluxGw, FluxHw, dx, dy, dz);
		});
	}).wait();
    #endif
}

void FluidBoundaryCondition(sycl::queue &q, BConditions BCs[6], Real*  d_UI)
{
	// //x direction
    // constexpr int dim_block_x = DIM_X ? WarpSize : 1;
    // constexpr int dim_block_y = DIM_Y ? WarpSize : 1;
	// constexpr int dim_block_z = 1;//DIM_Z ? WarpSize : 1;

    #if DIM_X
	auto local_ndrange_x = range<3>(Bwidth_X, dim_block_y, dim_block_z);	// size of workgroup
	auto global_ndrange_x = range<3>(Bwidth_X, Ymax, Zmax);

	BConditions BC0 = BCs[0], BC1 = BCs[1];

	q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0) + Bwidth_X - 1;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			FluidBCKernelX(i, j, k, BC0, d_UI, 0, 0, Bwidth_X, 1);
			FluidBCKernelX(i, j, k, BC1, d_UI, Xmax-Bwidth_X, X_inner, Xmax-Bwidth_X-1, -1);
		});
	});
    #endif

    // #if DIM_Y
	// dim3 dim_blk_y(dim_block_x, Bwidth_Y, dim_block_z);
    // dim3 dim_grid_y(X_inner/dim_block_x + DIM_X, 1, Z_inner/dim_block_z + DIM_Z);
    // FluidBCKernelY<<<dim_grid_y, dim_blk_y>>>(BCs[2], d_UI, 0, 0, Bwidth_Y, 1);
    // FluidBCKernelY<<<dim_grid_y, dim_blk_y>>>(BCs[3], d_UI, Ymax-Bwidth_Y, Y_inner, Ymax-Bwidth_Y-1, -1);
    // #endif

    // #if DIM_Z
	// dim3 dim_blk_z(1, dim_block_y, Bwidth_Z);
    // dim3 dim_grid_z(X_inner/1 + DIM_X, Y_inner/dim_block_y + DIM_Y, 1);
    // FluidBCKernelZ<<<dim_grid_z, dim_blk_z>>>(BCs[4], d_UI, 0, 0, Bwidth_Z, 1);
    // FluidBCKernelZ<<<dim_grid_z, dim_blk_z>>>(BCs[5], d_UI, Zmax-Bwidth_Z, Z_inner, Zmax-Bwidth_Z-1, -1);
    // #endif
}