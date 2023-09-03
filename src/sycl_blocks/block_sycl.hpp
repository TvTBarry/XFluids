#include "global_class.h"
#include "sycl_kernels.hpp"

using namespace sycl;

void InitializeFluidStates(sycl::queue &q, Block bl, IniShape ini, MaterialProperty material, Thermal thermal, FlowData &fdata, real_t *U, real_t *U1, real_t *LU,
						   real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
{
	auto local_ndrange = range<3>((bl.dim_block_x + 1) / 2, (bl.dim_block_y + 1) / 2, (bl.dim_block_z + 1) / 2); // size of workgroup
	auto global_ndrange = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	event ei = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			InitialStatesKernel(i, j, k, bl, ini, material, thermal, u, v, w, rho, p, fdata.y, T); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ei);
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			InitialUFKernel(i, j, k, bl, material, thermal, U, U1, LU, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, u, v, w, rho, p, fdata.y, T, H, c); }); })
		.wait();
}

real_t GetDt(sycl::queue &q, Block bl, Thermal &thermal, FlowData &fdata, real_t *uvw_c_max)
{
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;
	real_t *yi = fdata.y;

	int meshSize = bl.Xmax * bl.Ymax * bl.Zmax;
	auto local_ndrange = range<1>(bl.BlockSize); // size of workgroup
	auto global_ndrange = range<1>(meshSize);

	real_t dtref = _DF(0.0);

	// NOTE: dt of inviscous flow
	// add uvw and c individually if need more resources
	for (int n = 0; n < 6; n++)
		uvw_c_max[n] = _DF(0.0);

		// define reduction objects for sum, min, max reduction
		// auto reduction_sum = reduction(sum, sycl::plus<>());
#if DIM_X
	uvw_c_max[5] = sycl::max<real_t>(uvw_c_max[5], bl._dx * bl._dx);
	q.submit([&](sycl::handler &h)
			 {
    	auto reduction_max_x = reduction(&(uvw_c_max[0]), sycl::maximum<>());
		h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_x, [=](nd_item<1> index, auto &temp_max_x)
					   {
			auto id = index.get_global_id();
			temp_max_x.combine(sycl::fabs<real_t>(u[id]) + c[id]); }); });
#endif // end DIM_X
#if DIM_Y
	uvw_c_max[5] = sycl::max<real_t>(uvw_c_max[5], bl._dy * bl._dy);
	q.submit([&](sycl::handler &h)
			 {	
		auto reduction_max_y = reduction(&(uvw_c_max[1]), sycl::maximum<>());
		h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_y, [=](nd_item<1> index, auto &temp_max_y)
					   {
			auto id = index.get_global_id();
			temp_max_y.combine(sycl::fabs<real_t>(v[id]) + c[id]); }); });
#endif // end DIM_Y
#if DIM_Z
	uvw_c_max[5] = sycl::max<real_t>(uvw_c_max[5], bl._dz * bl._dz);
	q.submit([&](sycl::handler &h)
			 {	
		auto reduction_max_z = reduction(&(uvw_c_max[2]), sycl::maximum<>());
		h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_z, [=](nd_item<1> index, auto &temp_max_z)
					   {
			auto id = index.get_global_id();
			temp_max_z.combine(sycl::fabs<real_t>(w[id]) + c[id]); }); });
#endif // end DIM_Z
	q.wait();

	dtref = uvw_c_max[0] * bl._dx + uvw_c_max[1] * bl._dy + uvw_c_max[2] * bl._dz;

	// NOTE: dt of viscous flow
#ifdef Visc
	real_t *va = fdata.viscosity_aver;
	real_t *tca = fdata.thermal_conduct_aver;
	real_t *Da = fdata.Dkm_aver;
	real_t *hi = fdata.hi;

	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	auto local_ndrange_max = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange_max), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			Gettransport_coeff_aver(i, j, k, bl, thermal, va, tca, Da, fdata.y, hi, fdata.rho, fdata.p, fdata.T, fdata.Ertemp1, fdata.Ertemp2); }); })
		.wait();

	// max viscosity
	uvw_c_max[3] = _DF(0.0);
	auto reduction_max_miu = reduction(&(uvw_c_max[3]), sycl::maximum<>());
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_miu, [=](nd_item<1> index, auto &temp_max_miu)
							  {
			auto id = index.get_global_id();
			temp_max_miu.combine(va[id]); }); });
	// max rho
	uvw_c_max[4] = _DF(100.0);
	auto reduction_max_rho = reduction(&(uvw_c_max[4]), sycl::minimum<>());
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_rho, [=](nd_item<1> index, auto &temp_max_rho)
							  {
			auto id = index.get_global_id();
			temp_max_rho.combine(fdata.rho[id]); }); });
	q.wait();

	real_t temp_visc = _DF(14.0 / 3.0) * uvw_c_max[3] * uvw_c_max[5] / uvw_c_max[4];
	dtref = sycl::max<real_t>(dtref, temp_visc);
#endif // end get viscity

	return bl.CFLnumber / dtref;
}

bool UpdateFluidStateFlux(sycl::queue &q, Block bl, Thermal thermal, real_t *UI, FlowData &fdata, real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t const Gamma, int &error_patched_times)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	// // update rho and yi
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					Updaterhoyi(i, j, k, bl, UI, rho, fdata.y); }); })
		.wait();

#ifdef ESTIM_NAN
	int *error_posyi;
	bool *error_org, *error_nan;
	error_posyi = middle::MallocShared<int>(error_posyi, 4 + NUM_SPECIES, q);
	error_org = middle::MallocShared<bool>(error_org, 1, q), error_nan = middle::MallocShared<bool>(error_nan, 1, q);
	*error_nan = false, *error_org = false;
	for (size_t i = 0; i < NUM_SPECIES + 3; i++)
		error_posyi[i] = _DF(0.0);
	auto Sum_Epts = sycl::reduction(&(error_posyi[NUM_SPECIES + 3]), sycl::plus<>()); // error_patch_times

	// // update estimate negative or nan yi and patch
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), Sum_Epts, [=](sycl::nd_item<3> index, auto &tEpts)
							  {
									int i = index.get_global_id(0) + bl.Bwidth_X;
									int j = index.get_global_id(1) + bl.Bwidth_Y;
									int k = index.get_global_id(2) + bl.Bwidth_Z;
									EstimateYiKernel(i, j, k, bl, error_posyi, error_org, error_nan, UI, rho, fdata.y); }); }) //, tEpts
		.wait();

	int offsetx = bl.OutBC ? 0 : bl.Bwidth_X;
	int offsety = bl.OutBC ? 0 : bl.Bwidth_Y;
	int offsetz = bl.OutBC ? 0 : bl.Bwidth_Z;

	if (*error_org)
		error_patched_times += 1; // error_posyi[NUM_SPECIES + 3];
	if (*error_nan)
	{
		error_patched_times++;
		std::cout << "Errors of Yi[";
		for (size_t ii = 0; ii < NUM_COP; ii++)
			std::cout << error_posyi[ii] << ", ";
		std::cout << error_posyi[NUM_COP] << "] located at (i, j, k)= (";
		std::cout << error_posyi[NUM_SPECIES] - offsetx << ", " << error_posyi[NUM_SPECIES + 1] - offsety << ", " << error_posyi[NUM_SPECIES + 2] - offsetz;
#ifdef ERROR_PATCH_YI
			std::cout << ") patched.\n";
#else
			std::cout << ") captured.\n";
			return true;
#endif // end ERROR_PATCH_YI
	}
#endif // end ESTIM_NAN

	q.submit([&](sycl::handler &h)
			 {         
				// sycl::stream stream_ct1(64 * 1024, 80, h);// for output error: sycl::stream decline running efficiency
				h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					UpdateFuidStatesKernel(i, j, k, bl, thermal, UI, FluxF, FluxG, FluxH, rho, p, c, H, u, v, w, fdata.y, fdata.gamma, T, fdata.e, Gamma); }); }) //, stream_ct1
		.wait();

#ifdef ESTIM_NAN
	int *error_pos;
	bool *error_yi, *error_nga;
	error_pos = middle::MallocShared<int>(error_pos, 6 + NUM_SPECIES, q);
	error_yi = middle::MallocShared<bool>(error_yi, 1, q), error_nga = middle::MallocShared<bool>(error_nga, 1, q);
	*error_nga = false, *error_yi = false;
	for (size_t n = 0; n < 6 + NUM_SPECIES; n++)
		error_pos[n] = 0;

	auto global_in_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_in_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
								int i = index.get_global_id(0) + bl.Bwidth_X;
								int j = index.get_global_id(1) + bl.Bwidth_Y;
								int k = index.get_global_id(2) + bl.Bwidth_Z;
								EstimatePrimitiveVarKernel(i, j, k, bl, thermal, error_pos, error_nga, error_yi,
														   UI, rho, u, v, w, p, T, fdata.y, H, fdata.e, fdata.gamma, c); }); })
		.wait();

	if (*error_nga)
	{
		std::cout << "Errors of Primitive variables[rho, T, P][";
		for (size_t ii = 0; ii < 2 + NUM_SPECIES; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[2 + NUM_SPECIES] << "] located at (i, j, k)= (";
		std::cout << error_pos[3 + NUM_SPECIES] - offsetx << ", " << error_pos[4 + NUM_SPECIES] - offsety << ", " << error_pos[5 + NUM_SPECIES] - offsetz;
#ifdef ERROR_PATCH
		std::cout << ") patched.\n";
#else
		std::cout << ") captured.\n";
		return true;
#endif // end ERROR_PATCH
	}

	// free
	{
		middle::Free(error_posyi, q);
		middle::Free(error_org, q);
		middle::Free(error_nan, q);
		middle::Free(error_pos, q);
		middle::Free(error_yi, q);
		middle::Free(error_nga, q);
	}
#endif // end ESTIM_NAN

	return false;
}

void UpdateURK3rd(sycl::queue &q, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			UpdateURK3rdKernel(i, j, k, bl, U, U1, LU, dt, flag); }); })
		.wait();
}

void GetCellCenterDerivative(sycl::queue &q, Block bl, FlowData &fdata, BConditions BC[6])
{
	int range_x = DIM_X ? bl.X_inner + 4 : 1; // NOTE：这里的4是由求微分的算法确定的,内点网格向两边各延伸两个点
	int range_y = DIM_Y ? bl.Y_inner + 4 : 1;
	int range_z = DIM_Z ? bl.Z_inner + 4 : 1;
	int offset_x = DIM_X ? bl.Bwidth_X - 2 : 0; // NOTE: 这是计算第i(j/k)个点右边的那个半点，所以从=(+Bwidth-2) 开始到<(+inner+Bwidth+2)结束
	int offset_y = DIM_Y ? bl.Bwidth_Y - 2 : 0;
	int offset_z = DIM_Z ? bl.Bwidth_Z - 2 : 0;
	auto local_ndrange_ck = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_ck = range<3>(((range_x - 1) / bl.dim_block_x + 1) * bl.dim_block_x, ((range_y - 1) / bl.dim_block_y + 1) * bl.dim_block_y, ((range_z - 1) / bl.dim_block_z + 1) * bl.dim_block_z);

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_ck, local_ndrange_ck), [=](sycl::nd_item<3> index)
							  {
			int i = index.get_global_id(0) + offset_x;
			int j = index.get_global_id(1) + offset_y;
			int k = index.get_global_id(2) + offset_z;
			GetInnerCellCenterDerivativeKernel(i, j, k, bl, fdata.u, fdata.v, fdata.w, fdata.Vde, fdata.vxs, fdata.vx); }); })
		.wait();

#if DIM_X
	auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

	BConditions BC0 = BC[0], BC1 = BC[1];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
    		int i0 = index.get_global_id(0) + 0;
			int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			CenterDerivativeBCKernelX(i0, j, k, bl, BC0, fdata.Vde, 0, bl.Bwidth_X, 1);
			CenterDerivativeBCKernelX(i1, j, k, bl, BC1, fdata.Vde, bl.X_inner, bl.Xmax-bl.Bwidth_X-1, -1); }); }); //.wait()
#endif // DIM_X

#if DIM_Y
	auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

	BConditions BC2 = BC[2], BC3 = BC[3];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j0 = index.get_global_id(1) + 0;
			int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
			int k = index.get_global_id(2);

			CenterDerivativeBCKernelY(i, j0, k, bl, BC2, fdata.Vde, 0, bl.Bwidth_Y, 1);
			CenterDerivativeBCKernelY(i, j1, k, bl, BC3, fdata.Vde, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); }); }); //.wait()
#endif // DIM_Y

#if DIM_Z
	auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
	auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

	BConditions BC4 = BC[4], BC5 = BC[5];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
    		int k0 = index.get_global_id(2) + 0;
			int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

			CenterDerivativeBCKernelZ(i, j, k0, bl, BC4, fdata.Vde, 0, bl.Bwidth_Z, 1);
			CenterDerivativeBCKernelZ(i, j, k1, bl, BC5, fdata.Vde, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); }); }); //.wait()
#endif // DIM_Z
	q.wait();
}

void GetLU(sycl::queue &q, Block bl, BConditions BCs[6], Thermal thermal, real_t *UI, real_t *LU,
		   real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
		   real_t const Gamma, int const Mtrl_ind, FlowData &fdata, real_t *eigen_local_x, real_t *eigen_local_y, real_t *eigen_local_z,
		   real_t *eigen_l, real_t *eigen_r, real_t *uvw_c_max)
{
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	bool is_3d = DIM_X * DIM_Y * DIM_Z ? true : false;

	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

#if DIM_X
#ifdef DEBUG
	// std::cout << "  sleep before ReconstructFluxX\n";
	// sleep(5);
#endif // end DEBUG
	// proceed at x directiom and get F-flux terms at node wall
	auto global_ndrange_x = range<3>(bl.X_inner + local_ndrange[0], bl.Y_inner, bl.Z_inner);

	event ex = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 1.0, 0.0, 0.0, eigen_local_x, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ex);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			ReconstructFluxX(i, j, k, bl, thermal, UI, FluxF, FluxFw, eigen_local_x, eigen_l, eigen_r, fdata.b1x, fdata.b3x, fdata.c2x, fdata.zix, p, rho, u, v, w, fdata.y, T, H);
			 }); }); // real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,

#ifdef DEBUG
	// std::cout << "  sleep after ReconstructFluxX\n";
	// sleep(5);
#endif // end DEBUG
#endif

#if DIM_Y
#ifdef DEBUG
	// std::cout << "  sleep before ReconstructFluxY\n";
	// sleep(5);
#endif // end DEBUG
	// proceed at y directiom and get G-flux terms at node wall
	auto global_ndrange_y = range<3>(bl.X_inner, bl.Y_inner + local_ndrange[1], bl.Z_inner);

	event ey = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 0.0, 1.0, 0.0, eigen_local_y, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ey);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			ReconstructFluxY(i, j, k, bl, thermal, UI, FluxG, FluxGw, eigen_local_y, eigen_l, eigen_r, fdata.b1y, fdata.b3y, fdata.c2y, fdata.ziy, p, rho, u, v, w, fdata.y, T, H); 
			}); });
	//.wait()
#ifdef DEBUG
	// std::cout << "  sleep after ReconstructFluxY\n";
	// sleep(5);
#endif // end DEBUG

#endif

#if DIM_Z
#ifdef DEBUG
	// std::cout << "  sleep before ReconstructFluxZ\n";
	// sleep(5);
#endif // end DEBUG
	   // proceed at y directiom and get G-flux terms at node wall
	auto global_ndrange_z = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner + local_ndrange[2]);

	event ez = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 0.0, 0.0, 1.0, eigen_local_z, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ez);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
			ReconstructFluxZ(i, j, k, bl, thermal, UI, FluxH, FluxHw, eigen_local_z, eigen_l, eigen_r, fdata.b1z, fdata.b3z, fdata.c2z, fdata.ziz, p, rho, u, v, w, fdata.y, T, H); 
			}); }); //.wait()
#ifdef DEBUG
	// std::cout << "  sleep after ReconstructFluxZ\n";
	// sleep(5);
#endif // end DEBUG

#endif

	q.wait();

	// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
	// #if DIM_X
	// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
	// #endif
	// #if DIM_Y
	// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
	// #endif
	// #if DIM_Z
	// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
	// #endif
	// 	q.wait();

	// NOTE: positive preserving
	auto global_ndrange_inner = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	real_t lambda_x0 = uvw_c_max[0], lambda_y0 = uvw_c_max[1], lambda_z0 = uvw_c_max[2];
	real_t lambda_x = bl.CFLnumber / lambda_x0, lambda_y = bl.CFLnumber / lambda_y0, lambda_z = bl.CFLnumber / lambda_z0;
	real_t *epsilon = static_cast<real_t *>(sycl::malloc_shared((NUM_SPECIES + 2) * sizeof(real_t), q));
	epsilon[0] = _DF(1.0e-13), epsilon[1] = _DF(1.0e-13); // 0 for rho and 1 for T and P
	// real_t epsilon[NUM_SPECIES + 2] = {_DF(1.0e-13), _DF(1.0e-13)};
	for (size_t ii = 2; ii < NUM_SPECIES + 2; ii++) // for Yi
		epsilon[ii] = _DF(0.0);						// Ini epsilon for y1-yN(N species)

#ifdef PositivityPreserving
#if DIM_X // sycl::stream error_out(1024 * 1024, 1024, h);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
							  {
		    		int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
					int id_r = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1);
					PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxF, FluxFw, T, lambda_x0, lambda_x, epsilon); }); });
#endif	  // DIM_X
#if DIM_Y // sycl::stream error_out(1024 * 1024, 1024, h);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
							  {
		    		int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
					int id_r = (bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i);
					PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxG, FluxGw, T, lambda_y0, lambda_y, epsilon); }); });
#endif // end DIM_Y
#if DIM_Z
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
							  {
		    		int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
					int id_r = (bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i);
					PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxH, FluxHw, T, lambda_z0, lambda_z, epsilon); }); });
#endif // end DIM_Z
#endif // end posti

// 	q.wait();

// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
// #if DIM_X
// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
// #endif
// #if DIM_Y
// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
// #endif
// #if DIM_Z
// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
// #endif
// 	q.wait();

	GetCellCenterDerivative(q, bl, fdata, BCs); // get Vortex

#ifdef Visc // NOTE: calculate and add viscous wall Flux to physical convection Flux
	/* Viscous LU including physical visc(切应力),Heat transfer(传热), mass Diffusion(质量扩散)
	 * Physical Visc must be included, Heat is alternative, Diffu depends on compent
	 */
	real_t *va = fdata.viscosity_aver;
	real_t *tca = fdata.thermal_conduct_aver;
	real_t *Da = fdata.Dkm_aver;
	real_t *hi = fdata.hi;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
							  {
			int i = index.get_global_id(0);
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			Gettransport_coeff_aver(i, j, k, bl, thermal, va, tca, Da, fdata.y, hi, rho, p, T, fdata.Ertemp1, fdata.Ertemp2); }); })
		.wait();

#if DIM_X
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			GetWallViscousFluxX(i, j, k, bl, FluxFw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, fdata.visFwx, fdata.Dim_wallx, fdata.hi_wallx, fdata.Yi_wallx, fdata.Yil_wallx); }); }); //.wait()
#endif																												  // end DIM_X
#if DIM_Y
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			GetWallViscousFluxY(i, j, k, bl, FluxGw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, fdata.visFwy, fdata.Dim_wally, fdata.hi_wally, fdata.Yi_wally, fdata.Yil_wally); }); }); //.wait()
#endif																												  // end DIM_Y
#if DIM_Z
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
			GetWallViscousFluxZ(i, j, k, bl, FluxHw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, fdata.visFwz, fdata.Dim_wallz, fdata.hi_wallz, fdata.Yi_wallz, fdata.Yil_wallz); }); }); //.wait()
#endif																												  // end DIM_Z

#endif // end Visc
	q.wait();

	// NOTE: update LU from cell-face fluxes
#if NumFluid == 2
#else
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			
			UpdateFluidLU(i, j, k, bl, LU, FluxFw, FluxGw, FluxHw); }); })
		.wait();
#endif

	// free
	{
		middle::Free(epsilon, q);
	}
}

float FluidBoundaryCondition(sycl::queue &q, Setup setup, BConditions BCs[6], real_t *d_UI)
{
	Block bl = setup.BlSz;
#if USE_MPI
	MpiTrans Trans = *(setup.mpiTrans);
// =======================================================
#ifdef EXPLICIT_ALLOC
	// =======================================================
#if DIM_X
	real_t *ptr_TransBufSend_xmin = Trans.d_mpiData.TransBufSend_xmin;
	real_t *ptr_TransBufSend_xmax = Trans.d_mpiData.TransBufSend_xmax;
	real_t *ptr_TransBufRecv_xmin = Trans.d_mpiData.TransBufRecv_xmin;
	real_t *ptr_TransBufRecv_xmax = Trans.d_mpiData.TransBufRecv_xmax;
#endif // end DIM_X
#if DIM_Y
	real_t *ptr_TransBufSend_ymin = Trans.d_mpiData.TransBufSend_ymin;
	real_t *ptr_TransBufSend_ymax = Trans.d_mpiData.TransBufSend_ymax;
	real_t *ptr_TransBufRecv_ymin = Trans.d_mpiData.TransBufRecv_ymin;
	real_t *ptr_TransBufRecv_ymax = Trans.d_mpiData.TransBufRecv_ymax;
#endif // end DIM_Y
#if DIM_Z
	real_t *ptr_TransBufSend_zmin = Trans.d_mpiData.TransBufSend_zmin;
	real_t *ptr_TransBufSend_zmax = Trans.d_mpiData.TransBufSend_zmax;
	real_t *ptr_TransBufRecv_zmin = Trans.d_mpiData.TransBufRecv_zmin;
	real_t *ptr_TransBufRecv_zmax = Trans.d_mpiData.TransBufRecv_zmax;
#endif // end DIM_Z
	   // =======================================================
#else
	// =======================================================
#if DIM_X
#define ptr_TransBufSend_xmin Trans.d_mpiData->TransBufSend_xmin
#define ptr_TransBufSend_xmax Trans.d_mpiData->TransBufSend_xmax
#define ptr_TransBufRecv_xmin Trans.d_mpiData->TransBufRecv_xmin
#define ptr_TransBufRecv_xmax Trans.d_mpiData->TransBufRecv_xmax
#endif // end DIM_X
#if DIM_Y
#define ptr_TransBufSend_ymin Trans.d_mpiData->TransBufSend_ymin
#define ptr_TransBufSend_ymax Trans.d_mpiData->TransBufSend_ymax
#define ptr_TransBufRecv_ymin Trans.d_mpiData->TransBufRecv_ymin
#define ptr_TransBufRecv_ymax Trans.d_mpiData->TransBufRecv_ymax
#endif // end DIM_Y
#if DIM_Z
#define ptr_TransBufSend_zmin Trans.d_mpiData->TransBufSend_zmin
#define ptr_TransBufSend_zmax Trans.d_mpiData->TransBufSend_zmax
#define ptr_TransBufRecv_zmin Trans.d_mpiData->TransBufRecv_zmin
#define ptr_TransBufRecv_zmax Trans.d_mpiData->TransBufRecv_zmax
#endif // end DIM_Z
	// =======================================================
#endif // end EXPLICIT_ALLOC
// =======================================================
#endif // end USE_MPI

	float duration_x = 0.0f, duration_y = 0.0f, duration_z = 0.0f;
#if DIM_X
	auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

#if USE_MPI

	std::chrono::high_resolution_clock::time_point start_time_x = std::chrono::high_resolution_clock::now();

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
								  int i0 = index.get_global_id(0) + 0;
								  int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
								  int j = index.get_global_id(1);
								  int k = index.get_global_id(2);

								  FluidMpiCopyKernelX(i0, j, k, bl, ptr_TransBufSend_xmin, d_UI, 0, -bl.Bwidth_X, BorToBuf);					// X_MIN
								  FluidMpiCopyKernelX(i1, j, k, bl, ptr_TransBufSend_xmax, d_UI, bl.Xmax - bl.Bwidth_X, bl.Bwidth_X, BorToBuf); // X_MAX
							  }); })
		.wait();

	Trans.MpiTransBuf(q, XDIR);

	std::chrono::high_resolution_clock::time_point end_time_x = std::chrono::high_resolution_clock::now();
	duration_x = std::chrono::duration<float, std::milli>(end_time_x - start_time_x).count();

#endif // USE_MPI
	BConditions BC0 = BCs[0], BC1 = BCs[1];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
				   {
					   int i0 = index.get_global_id(0) + 0;
					   int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
					   int j = index.get_global_id(1);
					   int k = index.get_global_id(2);
#if USE_MPI
					   if (Trans.neighborsBC[XMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
						   FluidMpiCopyKernelX(i0, j, k, bl, ptr_TransBufRecv_xmin, d_UI, 0, -bl.Bwidth_X, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelX(i0, j, k, bl, BC0, d_UI, 0, bl.Bwidth_X, 1);
#ifdef USE_MPI
					   if (Trans.neighborsBC[XMAX] == BC_COPY)
						   FluidMpiCopyKernelX(i1, j, k, bl, ptr_TransBufRecv_xmax, d_UI, bl.Xmax - bl.Bwidth_X, bl.Bwidth_X, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelX(i1, j, k, bl, BC1, d_UI, bl.X_inner, bl.Xmax - bl.Bwidth_X - 1, -1); }); })
		.wait();
#endif // end DIM_X

#if DIM_Y
	auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

#if USE_MPI

	std::chrono::high_resolution_clock::time_point start_time_y = std::chrono::high_resolution_clock::now();

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j0 = index.get_global_id(1) + 0;
								  int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
								  int k = index.get_global_id(2);

								  FluidMpiCopyKernelY(i, j0, k, bl, ptr_TransBufSend_ymin, d_UI, 0, -bl.Bwidth_Y, BorToBuf);					// X_MIN
								  FluidMpiCopyKernelY(i, j1, k, bl, ptr_TransBufSend_ymax, d_UI, bl.Ymax - bl.Bwidth_Y, bl.Bwidth_Y, BorToBuf); // X_MAX
							  }); })
		.wait();

	Trans.MpiTransBuf(q, YDIR);

	std::chrono::high_resolution_clock::time_point end_time_y = std::chrono::high_resolution_clock::now();
	duration_y = std::chrono::duration<float, std::milli>(end_time_y - start_time_y).count();

#endif // USE_MPI
	BConditions BC2 = BCs[2], BC3 = BCs[3];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j0 = index.get_global_id(1) + 0;
								  int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
								  int k = index.get_global_id(2);
#if USE_MPI
								  if (Trans.neighborsBC[YMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
									  FluidMpiCopyKernelY(i, j0, k, bl, ptr_TransBufRecv_ymin, d_UI, 0, -bl.Bwidth_Y, BufToBC);
								  else
#endif // USE_MPI
									  FluidBCKernelY(i, j0, k, bl, BC2, d_UI, 0, bl.Bwidth_Y, 1);
#ifdef USE_MPI
								  if (Trans.neighborsBC[YMAX] == BC_COPY)
									  FluidMpiCopyKernelY(i, j1, k, bl, ptr_TransBufRecv_ymax, d_UI, bl.Ymax - bl.Bwidth_Y, bl.Bwidth_Y, BufToBC);
								  else
#endif																													  // USE_MPI
									  FluidBCKernelY(i, j1, k, bl, BC3, d_UI, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); //
							  }); })
		.wait();
#endif // end DIM_Y

#if DIM_Z
	auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
	auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

#if USE_MPI

	std::chrono::high_resolution_clock::time_point start_time_z = std::chrono::high_resolution_clock::now();

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j = index.get_global_id(1);
								  int k0 = index.get_global_id(2) + 0;
								  int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

								  FluidMpiCopyKernelZ(i, j, k0, bl, ptr_TransBufSend_zmin, d_UI, 0, -bl.Bwidth_Z, BorToBuf);					// X_MIN
								  FluidMpiCopyKernelZ(i, j, k1, bl, ptr_TransBufSend_zmax, d_UI, bl.Zmax - bl.Bwidth_Z, bl.Bwidth_Z, BorToBuf); // X_MAX
							  }); })
		.wait();

	Trans.MpiTransBuf(q, ZDIR);

	std::chrono::high_resolution_clock::time_point end_time_z = std::chrono::high_resolution_clock::now();
	duration_z = std::chrono::duration<float, std::milli>(end_time_z - start_time_z).count();

#endif // USE_MPI
	BConditions BC4 = BCs[4], BC5 = BCs[5];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j = index.get_global_id(1);
								  int k0 = index.get_global_id(2) + 0;
								  int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;
#if USE_MPI
								  if (Trans.neighborsBC[ZMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
									  FluidMpiCopyKernelZ(i, j, k0, bl, ptr_TransBufRecv_zmin, d_UI, 0, -bl.Bwidth_Z, BufToBC);
								  else
#endif // USE_MPI
									  FluidBCKernelZ(i, j, k0, bl, BC4, d_UI, 0, bl.Bwidth_Z, 1);
#ifdef USE_MPI
								  if (Trans.neighborsBC[ZMAX] == BC_COPY)
									  FluidMpiCopyKernelZ(i, j, k1, bl, ptr_TransBufRecv_zmax, d_UI, bl.Zmax - bl.Bwidth_Z, bl.Bwidth_Z, BufToBC);
								  else
#endif
									  FluidBCKernelZ(i, j, k1, bl, BC5, d_UI, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); //
							  }); })
		.wait();
#endif // end DIM_Z
	q.wait();

	return (duration_x + duration_y + duration_z) * 1.0e-3f;
}

void ZeroDimensionalFreelyFlameBlock(Setup &sep)
{
	ZeroDimensionalFreelyFlameKernel(sep, 0);
}

#ifdef COP_CHEME
void ChemeODEQ2Solver(sycl::queue &q, Block bl, Thermal thermal, FlowData &fdata, real_t *UI, Reaction react, const real_t dt)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0) + bl.Bwidth_X;
								  int j = index.get_global_id(1) + bl.Bwidth_Y;
								  int k = index.get_global_id(2) + bl.Bwidth_Z;
								  ChemeODEQ2SolverKernel( i, j, k, bl, thermal, react, UI, fdata.y, rho, T, fdata.e, dt); }); })
		.wait();
}
#endif // end COP_CHEME