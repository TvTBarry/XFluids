#include "Utils_block.hpp"

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
