#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"

void Getrhoyi(real_t UI[Emax], real_t &rho, real_t yi[NUM_SPECIES])
{
	rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	yi[NUM_COP] = _DF(1.0);
#ifdef COP
	for (size_t ii = 5; ii < Emax; ii++) // calculate yi
		yi[ii - 5] = UI[ii] * rho1, yi[NUM_COP] += -yi[ii - 5];

		// /** ceil(m): get an real_t value >= m and < m+1
		//  * step(a,b): return 1 while  a <= b
		//  */
		// real_t posy = sycl::step(sycl::ceil(yi[NUM_COP]), _DF(1.0e-20)), sum = _DF(0.0);
		// for (size_t ii = 0; ii < NUM_COP; ii++)
		// 	sum += yi[ii];
		// sum = _DF(1.0) / sum;
		// for (size_t ii = 0; ii < NUM_COP; ii++)
		// 	yi[ii] += yi[ii] * sum * yi[NUM_COP] * posy;
		// yi[NUM_COP] = yi[NUM_COP] * (1 - posy);
#endif // end COP
}
/**
 * @brief Obtain state at a grid point
 */
void GetStates(real_t *UI, real_t &rho, real_t &u, real_t &v, real_t &w, real_t &p, real_t &H, real_t &c,
			   real_t &gamma, real_t &T, real_t &e, Thermal thermal, real_t *yi)
{
	// rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	u = UI[1] * rho1;
	v = UI[2] * rho1;
	w = UI[3] * rho1;
	real_t tme = UI[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);

	// yi[NUM_COP] = _DF(1.0);
#ifdef COP
	// for (size_t ii = 5; ii < Emax; ii++) // calculate yi
	// 	yi[ii - 5] = UI[ii] * rho1, yi[NUM_COP] += -yi[ii - 5];
	real_t R = get_CopR(thermal._Wi, yi);
	T = get_T(thermal, yi, tme, T);
	p = rho * R * T; // 对所有气体都适用
	gamma = get_CopGamma(thermal, yi, T);
#else
	gamma = NCOP_Gamma;
	p = (NCOP_Gamma - _DF(1.0)) * rho * tme; //(UI[4] - _DF(0.5) * rho * (u * u + v * v + w * w));
#endif // end COP
	H = (UI[4] + p) * rho1;
	c = sycl::sqrt(gamma * p * rho1);
	e = tme;
}

void ReGetStates(Thermal thermal, real_t *yi, real_t *U, real_t &rho, real_t &u, real_t &v, real_t &w,
				 real_t &p, real_t &T, real_t &H, real_t &c, real_t &e, real_t &gamma)
{
	// real_t h = get_Coph(thermal, yi, T);
	// U[4] = rho * (h + _DF(0.5) * (u * u + v * v + w * w)) - p;

	real_t R = get_CopR(thermal._Wi, yi), rho1 = _DF(1.0) / rho;
	e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
	T = get_T(thermal, yi, e, T);
	p = rho * R * T; // 对所有气体都适用
	gamma = get_CopGamma(thermal, yi, T);
	H = (U[4] + p) * rho1;
	c = sycl::sqrt(gamma * p * rho1);

	// U[1] = rho * u;
	// U[2] = rho * v;
	// U[3] = rho * w;

	for (size_t nn = 0; nn < NUM_COP; nn++)
		U[nn + 5] = U[0] * yi[nn];
}

/**
 * @brief  Obtain fluxes at a grid point
 */
void GetPhysFlux(real_t *UI, real_t const *yi, real_t *FluxF, real_t *FluxG, real_t *FluxH,
				 real_t const rho, real_t const u, real_t const v, real_t const w, real_t const p, real_t const H, real_t const c)
{
	FluxF[0] = UI[1];
	FluxF[1] = UI[1] * u + p;
	FluxF[2] = UI[1] * v;
	FluxF[3] = UI[1] * w;
	FluxF[4] = (UI[4] + p) * u;

	FluxG[0] = UI[2];
	FluxG[1] = UI[2] * u;
	FluxG[2] = UI[2] * v + p;
	FluxG[3] = UI[2] * w;
	FluxG[4] = (UI[4] + p) * v;

	FluxH[0] = UI[3];
	FluxH[1] = UI[3] * u;
	FluxH[2] = UI[3] * v;
	FluxH[3] = UI[3] * w + p;
	FluxH[4] = (UI[4] + p) * w;

#ifdef COP
	for (size_t ii = 5; ii < Emax; ii++)
	{
		FluxF[ii] = UI[1] * yi[ii - 5];
		FluxG[ii] = UI[2] * yi[ii - 5];
		FluxH[ii] = UI[3] * yi[ii - 5];
	}
#endif
}
