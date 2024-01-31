#include "../timer/timer.h"
#include "Update_kernels.hpp"
#include "Estimate_kernels.hpp"
#include "UpdateStates_block.h"
#include "../utils/atttribute/attribute.h"

std::pair<bool, std::vector<float>> UpdateFluidStateFlux(sycl::queue &q, Setup Ss, Thermal thermal, real_t *UI,
														 FlowData &fdata, real_t *FluxF, real_t *FluxG, real_t *FluxH,
														 real_t const Gamma, int &error_patched_times, const int rank)
{
	Block bl = Ss.BlSz;
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *e = fdata.e;
	real_t *g = fdata.gamma;
	real_t *T = fdata.T;
	// real_t *Ri = fdata.Ri;
	// real_t *Cp = fdata.Cp;

	std::vector<float> timer_UD;
	float runtime_emyi = 0.0f, runtime_empv = 0.0f;
	float runtime_rhoyi = 0.0f, runtime_states = 0.0f;
	std::chrono::high_resolution_clock::time_point runtime_ud_start;

	auto global_ndrange = sycl::range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	auto local_ndrange = sycl::range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup

	// // update rho and yi
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index) { //
					   int i = index.get_global_id(0);
					   int j = index.get_global_id(1);
					   int k = index.get_global_id(2);
					   Updaterhoyi(i, j, k, bl, UI, rho, fdata.y);
				   }); })
		.wait();
	runtime_rhoyi = OutThisTime(runtime_ud_start);

#if ESTIM_NAN
	int *error_posyi;
	bool *error_org, *error_nan;
	error_org = middle::MallocShared<bool>(error_org, 1, q);
	error_nan = middle::MallocShared<bool>(error_nan, 1, q);
	error_posyi = middle::MallocShared<int>(error_posyi, 5 + NUM_SPECIES, q);
	*error_nan = false, *error_org = false;
	for (size_t i = 0; i < NUM_SPECIES + 4; i++)
		error_posyi[i] = _DF(0.0);

	// // update estimate negative or nan rho, yi
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index) { //
					   int i = index.get_global_id(0) + bl.Bwidth_X;
					   int j = index.get_global_id(1) + bl.Bwidth_Y;
					   int k = index.get_global_id(2) + bl.Bwidth_Z;
					   EstimateYiKernel(i, j, k, bl, error_posyi, error_org, error_nan, UI, rho, fdata.y);
				   }); })
		.wait();
	runtime_emyi = OutThisTime(runtime_ud_start);

	int offsetx = OutBoundary ? 0 : bl.Bwidth_X;
	int offsety = OutBoundary ? 0 : bl.Bwidth_Y;
	int offsetz = OutBoundary ? 0 : bl.Bwidth_Z;

	if (*error_org)
		error_patched_times += 1;
	if (*error_nan)
	{
		error_patched_times++;
		std::cout << "\nErrors of rho/Yi[";
		std::cout << error_posyi[NUM_SPECIES + 1] << ", ";
		for (size_t ii = 0; ii < NUM_SPECIES - 1; ii++)
			std::cout << error_posyi[ii] << ", ";
		std::cout << error_posyi[NUM_SPECIES] << "] located at (i, j, k)= ("
				  << error_posyi[NUM_SPECIES + 2] - offsetx << ", "
				  << error_posyi[NUM_SPECIES + 3] - offsety << ", "
				  << error_posyi[NUM_SPECIES + 4] - offsetz << ") of rank: " << rank;
#ifdef ERROR_PATCH_YI
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
		timer_UD.push_back(runtime_emyi);
		timer_UD.push_back(runtime_empv);
		timer_UD.push_back(runtime_rhoyi);
		timer_UD.push_back(runtime_states);
		return std::make_pair(true, timer_UD);
#endif // end ERROR_PATCH_YI
	}
#endif // end ESTIM_NAN

	// sycl::stream stream_ct1(64 * 1024, 80, h);// for output error: sycl::stream decline running
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index) { //
					   int i = index.get_global_id(0);
					   int j = index.get_global_id(1);
					   int k = index.get_global_id(2);
					   UpdateFuidStatesKernel(i, j, k, bl, thermal, UI, FluxF, FluxG, FluxH, rho, p, c, H, u, v, w, fdata.y, fdata.gamma, T, fdata.e, Gamma);
				   }); }) //, stream_ct1
		.wait();
	runtime_states = OutThisTime(runtime_ud_start);

#if ESTIM_NAN
	int *error_pos;
	bool *error_yi, *error_nga;
	error_pos = middle::MallocShared<int>(error_pos, 6 + NUM_SPECIES, q);
	error_yi = middle::MallocShared<bool>(error_yi, 1, q), error_nga = middle::MallocShared<bool>(error_nga, 1, q);
	*error_nga = false, *error_yi = false;
	for (size_t n = 0; n < 6 + NUM_SPECIES; n++)
		error_pos[n] = 0;

	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(sycl::range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner), local_ndrange), [=](sycl::nd_item<3> index) { //
					   int i = index.get_global_id(0) + bl.Bwidth_X;
					   int j = index.get_global_id(1) + bl.Bwidth_Y;
					   int k = index.get_global_id(2) + bl.Bwidth_Z;
					   EstimatePrimitiveVarKernel(i, j, k, bl, thermal, error_pos, error_nga, error_yi,
												  UI, rho, u, v, w, p, T, fdata.y, H, fdata.e, fdata.gamma, c);
				   }); })
		.wait();
	runtime_empv = OutThisTime(runtime_ud_start);

	if (*error_nga)
	{
		std::cout << "\nErrors of Primitive variables[rho, T, P][";
		for (size_t ii = 0; ii < 2; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[2] << "] located at (i, j, k)= ("
				  << error_pos[3 + NUM_SPECIES] - offsetx << ", "
				  << error_pos[4 + NUM_SPECIES] - offsety << ", "
				  << error_pos[5 + NUM_SPECIES] - offsetz << ") of rank: " << rank;
#ifdef ERROR_PATCH
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
		timer_UD.push_back(runtime_emyi);
		timer_UD.push_back(runtime_empv);
		timer_UD.push_back(runtime_rhoyi);
		timer_UD.push_back(runtime_states);
		return std::make_pair(true, timer_UD);
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

	timer_UD.push_back(runtime_emyi);
	timer_UD.push_back(runtime_empv);
	timer_UD.push_back(runtime_rhoyi);
	timer_UD.push_back(runtime_states);
	return std::make_pair(false, timer_UD);
}

void UpdateURK3rd(sycl::queue &q, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
	auto local_ndrange = sycl::range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = sycl::range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
				   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			UpdateURK3rdKernel(i, j, k, bl, U, U1, LU, dt, flag); }); })
		.wait();
}
