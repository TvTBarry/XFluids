#pragma once
#include "include/global_class.h"
#include "device_func.hpp"

extern SYCL_EXTERNAL void InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal *thermal, real_t *U, real_t *U1, real_t *LU,
                                              real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
                                              real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T, real_t *H, real_t *c)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    real_t dx = bl.dx;
    real_t dy = bl.dy;
    real_t dz = bl.dz;

    int id = k * Ymax * Zmax + j * Ymax + i;

#if DIM_X
    if (i >= Xmax)
        return;
#endif
#if DIM_Y
    if (j >= Ymax)
        return;
#endif
#if DIM_Z
    if (k >= Zmax)
        return;
#endif

    real_t x = (i - Bwidth_X + bl.myMpiPos_x * (Xmax - Bwidth_X - Bwidth_X)) * dx + 0.5 * dx;
    real_t y = (j - Bwidth_Y + bl.myMpiPos_y * (Ymax - Bwidth_Y - Bwidth_Y)) * dy + 0.5 * dy;
    real_t z = (k - Bwidth_Z + bl.myMpiPos_z * (Zmax - Bwidth_Z - Bwidth_Z)) * dz + 0.5 * dz;

    real_t d2;
    switch (ini.blast_type)
    {
    case 0:
        d2 = x;
        break;
    case 1:
        d2 = ((x - ini.blast_center_x) * (x - ini.blast_center_x) + (y - ini.blast_center_y) * (y - ini.blast_center_y));
        break;
    }
#ifdef COP
    real_t dy2, copBin = 0.0, copBout = 0.0;
    int n = int(ini.cop_radius / dx) + 1;
    switch (ini.cop_type)
    { // 可以选择组分不同区域，圆形或类shock-wave
    case 1:
        dy2 = ((x - ini.cop_center_x) * (x - ini.cop_center_x) + (y - ini.cop_center_y) * (y - ini.cop_center_y));
        copBin = (n - 1) * (n - 1) * dx * dx;
        copBout = (n + 1) * (n + 1) * dx * dx;
        break;
    case 0:
        dy2 = x;
        copBin = ini.blast_center_x - dx;
        copBout = ini.blast_center_x + dx;
        break;
    }
#endif

#if 1 == NumFluid
    // 1d shock tube case
    if (d2 < ini.blast_center_x)
    {
        rho[id] = ini.blast_density_in;
        u[id] = ini.blast_u_in;
        v[id] = ini.blast_v_in;
        w[id] = ini.blast_w_in;
        p[id] = ini.blast_pressure_in;
#ifdef COP
        _y[id * NUM_COP + 0] = ini.cop_y1_out;
#endif // COP
    }
    else
    {
        rho[id] = ini.blast_density_out;
        p[id] = ini.blast_pressure_out;
        u[id] = ini.blast_u_out;
        v[id] = ini.blast_v_out;
        w[id] = ini.blast_w_out;
#ifdef COP
        if (dy2 < copBin) //|| dy2 == (n - 1) * (n - 1) * dx * dx)
        {
            rho[id] = ini.cop_density_in;         // 气泡内单独赋值密度以和气泡外区分
            p[id] = ini.cop_pressure_in;          // 气泡内单独赋值压力以和气泡外区分
            _y[id * NUM_COP + 0] = ini.cop_y1_in; // 组分气泡必须在激波下游
        }
        else if (dy2 > copBout)
        {
            rho[id] = ini.blast_density_out;
            p[id] = ini.blast_pressure_out;
            _y[id * NUM_COP + 0] = ini.cop_y1_out;
        }
        else
        {
            rho[id] = 0.5 * (ini.cop_density_in + ini.blast_density_out);
            p[id] = 0.5 * (ini.cop_pressure_in + ini.blast_pressure_out);
            _y[id * NUM_COP + 0] = 0.5 * (ini.cop_y1_out + ini.cop_y1_in);
        }
#endif // COP
    }
#endif // MumFluid
#if 2 == NumFluid
    if (material.Rgn_ind > 0.5)
    {
        rho[id] = 0.125;
        u[id] = 0.0;
        v[id] = 0.0;
        w[id] = 0.0;
        if (x < 0.1)
            p[id] = 10;
        else
            p[id] = 0.1;
    }
    else
    {
        rho[id] = 1.0;
        u[id] = 0.0;
        v[id] = 0.0;
        w[id] = 0.0;
        p[id] = 1.0;
    }
#endif // 2==NumFluid
    U[Emax * id + 0] = rho[id];
    U[Emax * id + 1] = rho[id] * u[id];
    U[Emax * id + 2] = rho[id] * v[id];
    U[Emax * id + 3] = rho[id] * w[id];
// EOS was included
#ifdef COP
    real_t yi[NUM_SPECIES];
    get_yi(_y, yi, id);
    real_t R = get_CopR(thermal->species_chara, yi);
    T[id] = p[id] / rho[id] / R; // p[id] / rho[id] / R;
    real_t Gamma_m = get_CopGamma(thermal, yi, T[id]);
    real_t h = get_Coph(thermal, yi, T[id]);
    U[Emax * id + 4] = rho[id] * h - p[id];
    // printf("for %d=%d,%d,%d,R=%lf,T=%lf,yi=%lf,%lf,h of Ini=%lf,U[4]=%lf \n", id, i, j, k, R, T[id], yi[0], yi[1], h, U[Emax * id + 4]);
    for (size_t ii = 5; ii < Emax; ii++)
    {
        U[Emax * id + ii] = rho[id] * _y[id * NUM_COP + ii - 5];
        FluxF[Emax * id + ii] = rho[id] * u[id] * _y[id * NUM_COP + ii - 5];
        FluxG[Emax * id + ii] = rho[id] * v[id] * _y[id * NUM_COP + ii - 5];
        FluxH[Emax * id + ii] = rho[id] * w[id] * _y[id * NUM_COP + ii - 5];
    }
    c[id] = sqrt(p[id] / rho[id] * Gamma_m);
#else
    //  for both singlephase && multiphase
    c[id] = material.Mtrl_ind == 0 ? sqrt(material.Gamma * p[id] / rho[id]) : sqrt(material.Gamma * (p[id] + material.B - material.A) / rho[id]);
    if (material.Mtrl_ind == 0)
        U[Emax * id + 4] = p[id] / (material.Gamma - 1.0) + 0.5 * rho[id] * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id]);
    else
        U[Emax * id + 4] = (p[id] + material.Gamma * (material.B - material.A)) / (material.Gamma - 1.0) + 0.5 * rho[id] * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id]);
#endif // COP
    // printf("U in Ini=%lf,%lf,%lf,%lf,%lf,%lf,T=%lf,h=%lf", U[Emax * id + 0], U[Emax * id + 1], U[Emax * id + 2], U[Emax * id + 3], U[Emax * id + 4], U[Emax * id + 5], T[id], h);
    H[id] = (U[Emax * id + 4] + p[id]) / rho[id];
    // initial flux terms F, G, H
    FluxF[Emax * id + 0] = U[Emax * id + 1];
    FluxF[Emax * id + 1] = U[Emax * id + 1] * u[id] + p[id];
    FluxF[Emax * id + 2] = U[Emax * id + 1] * v[id];
    FluxF[Emax * id + 3] = U[Emax * id + 1] * w[id];
    FluxF[Emax * id + 4] = (U[Emax * id + 4] + p[id]) * u[id];

    FluxG[Emax * id + 0] = U[Emax * id + 2];
    FluxG[Emax * id + 1] = U[Emax * id + 2] * u[id];
    FluxG[Emax * id + 2] = U[Emax * id + 2] * v[id] + p[id];
    FluxG[Emax * id + 3] = U[Emax * id + 2] * w[id];
    FluxG[Emax * id + 4] = (U[Emax * id + 4] + p[id]) * v[id];

    FluxH[Emax * id + 0] = U[Emax * id + 3];
    FluxH[Emax * id + 1] = U[Emax * id + 3] * u[id];
    FluxH[Emax * id + 2] = U[Emax * id + 3] * v[id];
    FluxH[Emax * id + 3] = U[Emax * id + 3] * w[id] + p[id];
    FluxH[Emax * id + 4] = (U[Emax * id + 4] + p[id]) * w[id];

#if NumFluid != 1
    real_t fraction = material.Rgn_ind > 0.5 ? vof[id] : 1.0 - vof[id];
#endif
    // give intial value for the interval matrixes
    for (int n = 0; n < Emax; n++)
    {
        LU[Emax * id + n] = 0.0;              // incremental of one time step
        U1[Emax * id + n] = U[Emax * id + n]; // intermediate conwervatives
        FluxFw[Emax * id + n] = 0.0;          // numerical flux F
        FluxGw[Emax * id + n] = 0.0;          // numerical flux G
        FluxHw[Emax * id + n] = 0.0;          // numerical flux H
#if NumFluid != 1
        CnsrvU[Emax * id + n] = U[Emax * id + n] * fraction;
        CnsrvU1[Emax * id + n] = CnsrvU[Emax * id + n];
#endif // NumFluid
    }
}

/*
// 被SYCL内核调用的函数需要加"extern SYCL_EXTERNAL"
extern SYCL_EXTERNAL void InitialStatesKernel(int i, int j, int k, Block bl, MaterialProperty *material, real_t *U, real_t *U1, real_t *LU,
                                              real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
                                              real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *H, real_t *c)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    real_t dx = bl.dx;
    real_t dy = bl.dy;
    real_t dz = bl.dz;
    real_t dl = bl.dl;

    int id = Xmax*Ymax*k + Xmax*j + i;

    real_t x = (i - Bwidth_X) * dx + half_float * dx;
    real_t y = (j - Bwidth_Y) * dy + half_float * dy;
    real_t z = (k - Bwidth_Z) * dz + half_float * dz;

    // 1d shock tube case
    #if USE_DP
    if (x < 0.5 * 1)
    {
        rho[id] = 1.0;
        u[id] = 0.0;
        v[id] = 0.0;
        w[id] = 0.0;
        p[id] = 1.0;
    }
    else{
        rho[id] = 0.125;
        u[id] = 0.0;
        v[id] = 0.0;
        w[id] = 0.0;
        p[id] = 0.1;
    }
    #else
    if (x < 0.5f * 1)
    {
        rho[id] = 1.0f;
        u[id] = 0.0f;
        v[id] = 0.0f;
        w[id] = 0.0f;
        p[id] = 1.0f;
    }
    else{
        rho[id] = 0.125f;
        u[id] = 0.0f;
        v[id] = 0.0f;
        w[id] = 0.0f;
        p[id] = 0.1f;
    }
    #endif

    // // 2d sod case
    // d_rho[id] = 1.0; d_u[id] = 0;
    // d_v[id] = 0;		d_p[id] = 4.0e-13;
    // if(x<=dx && y<=dy)
    // 	d_p[id] = 9.79264/dx/dy*10000.0;

    // // two-phase
    // if(material.Rgn_ind>0.5){
    //     rho[id] = 0.125;
    //     u[id] = 0.0;
    //     v[id] = 0.0;
    //     w[id] = 0.0;
    //     p[id] = 0.1;
    // }
    // else
    // {
    //     rho[id] = 1.0;
    //     u[id] = 0.0;
    //     v[id] = 0.0;
    //     w[id] = 0.0;
    //     p[id] = 1.0;
    // }

    U[Emax*id+0] = rho[id];
    U[Emax*id+1] = rho[id]*u[id];
    U[Emax*id+2] = rho[id]*v[id];
    U[Emax*id+3] = rho[id]*w[id];
    //EOS was included
    if(material->Mtrl_ind == 0)
        U[Emax*id+4] = p[id] /(material->Gamma-one_float) + half_float*rho[id]*(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
    else
        U[Emax*id+4] = (p[id] + material->Gamma*(material->B-material->A))/(material->Gamma-one_float)
                                            + half_float*rho[id]*(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);

    H[id]		= (U[Emax*id+4] + p[id])/rho[id];
    c[id]		= material->Mtrl_ind == 0 ? sqrt(material->Gamma*p[id]/rho[id]) : sqrt(material->Gamma*(p[id] + material->B - material->A)/rho[id]);

    //initial flux terms F, G, H
    FluxF[Emax*id+0] = U[Emax*id+1];
    FluxF[Emax*id+1] = U[Emax*id+1]*u[id] + p[id];
    FluxF[Emax*id+2] = U[Emax*id+1]*v[id];
    FluxF[Emax*id+3] = U[Emax*id+1]*w[id];
    FluxF[Emax*id+4] = (U[Emax*id+4] + p[id])*u[id];

    FluxG[Emax*id+0] = U[Emax*id+2];
    FluxG[Emax*id+1] = U[Emax*id+2]*u[id];
    FluxG[Emax*id+2] = U[Emax*id+2]*v[id] + p[id];
    FluxG[Emax*id+3] = U[Emax*id+2]*w[id];
    FluxG[Emax*id+4] = (U[Emax*id+4] + p[id])*v[id];

    FluxH[Emax*id+0] = U[Emax*id+3];
    FluxH[Emax*id+1] = U[Emax*id+3]*u[id];
    FluxH[Emax*id+2] = U[Emax*id+3]*v[id];
    FluxH[Emax*id+3] = U[Emax*id+3]*w[id] + p[id];
    FluxH[Emax*id+4] = (U[Emax*id+4] + p[id])*w[id];

    // real_t fraction = material->Rgn_ind > 0.5 ? vof[id] : 1.0 - vof[id];

    //give intial value for the interval matrixes
    for(int n=0; n<Emax; n++){
        LU[Emax*id+n] = 0.0; //incremental of one time step
        U1[Emax*id+n] = U[Emax*id+n]; //intermediate conwervatives

        // CnsrvU[Emax*id+n] = U[Emax*id+n]*fraction;
        // CnsrvU1[Emax*id+n] = CnsrvU[Emax*id+n];

        FluxFw[Emax*id+n] = 0.0; //numerical flux F
        FluxGw[Emax*id+n] = 0.0; //numerical flux G
        FluxHw[Emax*id+n] = 0.0; //numerical flux H
    }
}
*/

/**
 * @brief calculate c^2 of the mixture at given point
 */
real_t get_CopC2(real_t z[NUM_SPECIES], Thermal *thermal, real_t yi[NUM_SPECIES], real_t hi[NUM_SPECIES], const real_t h, const real_t gamma, const real_t T)
{
    real_t Sum_dpdrhoi = 0.0;                      // Sum_dpdrhoi:first of c2,存在累加项
    real_t Ri[NUM_SPECIES], _dpdrhoi[NUM_SPECIES]; // hi[NUM_SPECIES]
    // enthalpy h_i (unit: J/kg) and mass fraction Y_i of each specie
    for (size_t n = 0; n < NUM_SPECIES; n++)
    {
        Ri[n] = Ru / thermal->species_chara[n * SPCH_Sz + 6];
        _dpdrhoi[n] = (gamma - one_float) * (hi[0] - hi[n]) + gamma * (Ri[n] - Ri[0]) * T;
        // printf("_dpdrhoi[n]=%lf \n", _dpdrhoi[n]);
        z[n] = -one_float * _dpdrhoi[n] / (gamma - one_float);
        if (0 != n)
            Sum_dpdrhoi += yi[n] * _dpdrhoi[n];
    }
    real_t _CopC2 = Sum_dpdrhoi + (gamma - 1) * (h - hi[0]) + gamma * Ri[0] * T;
    // printf("gamma=%lf,h=%lf,hi=%lf,%lf,Ri=%lf,%lf,T=%lf \n", gamma, h, hi[0], hi[1], Ri[0], Ri[1], T);
    return _CopC2;
}

// add "sycl::nd_item<3> item" for get_global_id
// add "stream const s" for output
extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Block bl, Thermal *thermal, real_t const Gamma, real_t *UI, real_t *Fx,
                                           real_t *Fxwall, real_t *eigen_local, real_t *rho, real_t *u, real_t *v, real_t *w,
                                           real_t *y, real_t *T, real_t *H)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    real_t dx = bl.dx;
    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*k + Xmax*j + i + 1;

	// cout<<"i,j,k = "<<i<<", "<<j<<", "<<k<<", "<<Emax*(Xmax*j + i-2)+0<<"\n";
	// printf("%f", UI[0]);

    if(i>= X_inner+Bwidth_X)
        return;

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax];

    // preparing some interval value for roe average
    real_t D = sqrt(rho[id_r] / rho[id_l]);
    real_t D1 = one_float / (D + one_float);

    real_t _u = (u[id_l] + D * u[id_r]) * D1;
    real_t _v = (v[id_l] + D * v[id_r]) * D1;
    real_t _w = (w[id_l] + D * w[id_r]) * D1;
    real_t _H = (H[id_l] + D * H[id_r]) * D1;
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

#ifdef COP
    real_t _T = (T[id_l] + D * T[id_r]) * D1;
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_SPECIES];
    // NOTE: _hi Only defined by T , get_T at left && right may be different
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
        real_t Ri = Ru / (thermal->species_chara[i * SPCH_Sz + 6]);
        hi_l[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], Ri, i);
        hi_r[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], Ri, i);
    }
    get_yi(y, yi_l, id_l);
    get_yi(y, yi_r, id_r);
    real_t _h = 0;
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
    {
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;
        _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;
        _h += _hi[ii] * _yi[ii];
    }
    //  _h=_H-half_float*(_u*_u+_v*_v+_w*_w);
    real_t Gamma0 = get_CopGamma(thermal, _yi, _T); // out from RoeAverage_x
    real_t c2 = get_CopC2(z, thermal, _yi, _hi, _h, Gamma0, _T); // z[NUM_SPECIES] 是一个在该函数中同时计算的数组变量
    // printf("argus at [%d],[%d][%d][%d]=yi=%lf, %lf, T=%lf,hi=%lf, %lf, _h=%lf, Gamma=%lf, c2=%lf, zi=%lf,%lf\n", id_l, i, j, k, _yi[0], _yi[1], T, _hi[0], _hi[1], _h, Gamma0, c2, z[0], z[1]); //_Cp
#else
    real_t Gamma0 = Gamma;
    real_t c2 = Gamma0 * (_H - half_float * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
    real_t z[NUM_SPECIES] = {0};
#endif
    //  printf("%lf , %lf , %lf , %lf , %lf , %lf \n", UI[Emax * id_l + 0], UI[Emax * id_l + 1], UI[Emax * id_l + 2], UI[Emax * id_l + 3], UI[Emax * id_l + 4], UI[Emax * id_l + 5]);
    //  printf("%lf , %lf , %lf , %lf , %lf , %lf \n", UI[Emax * id_r + 0], UI[Emax * id_r + 1], UI[Emax * id_r + 2], UI[Emax * id_r + 3], UI[Emax * id_r + 4], UI[Emax * id_r + 5]);

    RoeAverage_x(eigen_l, eigen_r, z, _yi, c2, _rho, _u, _v, _w, _H, D, D1, Gamma0);

    real_t uf[10], ff[10], pp[10], mm[10];
    real_t f_flux, _p[Emax][Emax];

    // construct the right value & the left value scalar equations by characteristic reduction			
	// at i+1/2 in x direction
    // #pragma unroll Emax
	for(int n=0; n<Emax; n++){
        real_t eigen_local_max = zero_float;
        for(int m=-2; m<=3; m++){
            int id_local = Xmax*Ymax*k + Xmax*j + i + m;
            eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax*id_local+n]));//local lax-friedrichs	
        }

		for(int m=i-3; m<=i+4; m++){	// 3rd oder and can be modified
            int id_local = Xmax*Ymax*k + Xmax*j + m;
            #if USE_DP
			uf[m-i+3] = 0.0;
			ff[m-i+3] = 0.0;
            #else
			uf[m-i+3] = 0.0f;
			ff[m-i+3] = 0.0f;
            #endif

			for(int n1=0; n1<Emax; n1++){
				uf[m-i+3] = uf[m-i+3] + UI[Emax*id_local+n1]*eigen_l[n][n1];
				ff[m-i+3] = ff[m-i+3] + Fx[Emax*id_local+n1]*eigen_l[n][n1];
			}
			// for local speed
			pp[m-i+3] = 0.5f*(ff[m-i+3] + eigen_local_max*uf[m-i+3]);
			mm[m-i+3] = 0.5f*(ff[m-i+3] - eigen_local_max*uf[m-i+3]);
        }

		// calculate the scalar numerical flux at x direction
        #if USE_DP
        f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx))/6.0;
        #else
        f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx))/6.0f;
        #endif

		// get Fp
		for(int n1=0; n1<Emax; n1++)
			_p[n][n1] = f_flux*eigen_r[n1][n];
    }

	// reconstruction the F-flux terms
	for(int n=0; n<Emax; n++){
        #if USE_DP
        real_t fluxx = 0.0;
#else
        real_t fluxx = 0.0f;
#endif
		for(int n1=0; n1<Emax; n1++) {
            fluxx += _p[n1][n];
		}
        Fxwall[Emax*id_l+n] = fluxx;
	}
}

extern SYCL_EXTERNAL void ReconstructFluxY(int i, int j, int k, Block bl, Thermal *thermal, real_t const Gamma, real_t *UI, real_t *Fy,
                                           real_t *Fywall, real_t *eigen_local, real_t *rho, real_t *u, real_t *v, real_t *w,
                                           real_t *y, real_t *T, real_t *H)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    real_t dy = bl.dy;
    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*k + Xmax*(j+ 1) + i;

    if(j>= Y_inner+Bwidth_Y)
        return;

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax];

    //preparing some interval value for roe average
    real_t D = sqrt(rho[id_r] / rho[id_l]);
    real_t D1 = one_float / (D + one_float);
    real_t _u = (u[id_l] + D * u[id_r]) * D1;
    real_t _v = (v[id_l] + D * v[id_r]) * D1;
    real_t _w = (w[id_l] + D * w[id_r]) * D1;
    real_t _H = (H[id_l] + D * H[id_r]) * D1;
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

#ifdef COP
    real_t _T = (T[id_l] + D * T[id_r]) * D1;
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_SPECIES];
    // NOTE: _hi Only defined by T , get_T at left && right may be different
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
        real_t Ri = Ru / (thermal->species_chara[i * SPCH_Sz + 6]);
        hi_l[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], Ri, i);
        hi_r[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], Ri, i);
    }
    get_yi(y, yi_l, id_l);
    get_yi(y, yi_r, id_r);
    real_t _h = 0;
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
    {
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;
        _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;
        _h += _hi[ii] * _yi[ii];
    }
    //  _h=_H-half_float*(_u*_u+_v*_v+_w*_w);
    real_t Gamma0 = get_CopGamma(thermal, _yi, _T); // out from RoeAverage_x
    real_t c2 = get_CopC2(z, thermal, _yi, _hi, _h, Gamma0, _T);
    // printf("argus at [%d],[%d][%d][%d]=yi=%lf, %lf, T=%lf,hi=%lf, %lf, _h=%lf, Gamma=%lf, c2=%lf, zi=%lf,%lf\n", id_l, i, j, k, _yi[0], _yi[1], T, _hi[0], _hi[1], _h, Gamma0, c2, z[0], z[1]); //_Cp
#else
    real_t Gamma0 = Gamma;
    real_t c2 = Gamma0 * (_H - half_float * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
    real_t z[NUM_SPECIES] = {0};
#endif

    RoeAverage_y(eigen_l, eigen_r, z, _yi, c2, _rho, _u, _v, _w, _H, D, D1, Gamma0);

    real_t ug[10], gg[10], pp[10], mm[10];
    real_t g_flux, _p[Emax][Emax];

    //construct the right value & the left value scalar equations by characteristic reduction			
	// at j+1/2 in y direction
	for(int n=0; n<Emax; n++){
        #if USE_DP
        real_t eigen_local_max = 0.0;
#else
        real_t eigen_local_max = 0.0f;
#endif

        for(int m=-2; m<=3; m++){
            int id_local = Xmax*Ymax*k + Xmax*(j + m) + i;
            eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax*id_local+n]));//local lax-friedrichs	
        }

		for(int m=j-3; m<=j+4; m++){	// 3rd oder and can be modified
            int id_local = Xmax*Ymax*k + Xmax*m + i;
            #if USE_DP
			ug[m-j+3] = 0.0;
			gg[m-j+3] = 0.0;
            #else
			ug[m-j+3] = 0.0f;
			gg[m-j+3] = 0.0f;
            #endif

			for(int n1=0; n1<Emax; n1++){
				ug[m-j+3] = ug[m-j+3] + UI[Emax*id_local+n1]*eigen_l[n][n1];
				gg[m-j+3] = gg[m-j+3] + Fy[Emax*id_local+n1]*eigen_l[n][n1];
			}
			//for local speed
			pp[m-j+3] = 0.5f*(gg[m-j+3] + eigen_local_max*ug[m-j+3]); 
			mm[m-j+3] = 0.5f*(gg[m-j+3] - eigen_local_max*ug[m-j+3]); 
        }
		// calculate the scalar numerical flux at y direction
        #if USE_DP
        g_flux = (weno5old_P(&pp[3], dy) + weno5old_M(&mm[3], dy))/6.0;
        #else
        g_flux = (weno5old_P(&pp[3], dy) + weno5old_M(&mm[3], dy))/6.0f;
        #endif

		// get Gp
		for(int n1=0; n1<Emax; n1++)
			_p[n][n1] = g_flux*eigen_r[n1][n];
    }
	// reconstruction the G-flux terms
	for(int n=0; n<Emax; n++){
        #if USE_DP
        real_t fluxy = 0.0;
#else
        real_t fluxy = 0.0f;
#endif
		for(int n1=0; n1<Emax; n1++){
            fluxy += _p[n1][n];
		}
        Fywall[Emax*id_l+n] = fluxy;
	}
}

extern SYCL_EXTERNAL void ReconstructFluxZ(int i, int j, int k, Block bl, Thermal *thermal, real_t const Gamma, real_t *UI, real_t *Fz,
                                           real_t *Fzwall, real_t *eigen_local, real_t *rho, real_t *u, real_t *v, real_t *w,
                                           real_t *y, real_t *T, real_t *H)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    real_t dz = bl.dz;
    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*(k+1) + Xmax*j + i;

    if(k>= Z_inner+Bwidth_Z)
        return;

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax];

    //preparing some interval value for roe average
    real_t D = sqrt(rho[id_r] / rho[id_l]);
    real_t D1 = one_float / (D + one_float);
    real_t _u = (u[id_l] + D * u[id_r]) * D1;
    real_t _v = (v[id_l] + D * v[id_r]) * D1;
    real_t _w = (w[id_l] + D * w[id_r]) * D1;
    real_t _H = (H[id_l] + D * H[id_r]) * D1;
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

#ifdef COP
    real_t _T = (T[id_l] + D * T[id_r]) * D1;
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_SPECIES];
    // NOTE: _hi Only defined by T , get_T at left && right may be different
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
        real_t Ri = Ru / (thermal->species_chara[i * SPCH_Sz + 6]);
        hi_l[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], Ri, i);
        hi_r[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], Ri, i);
    }
    get_yi(y, yi_l, id_l);
    get_yi(y, yi_r, id_r);
    real_t _h = 0;
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
    {
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;
        _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;
        _h += _hi[ii] * _yi[ii];
    }
    //  _h=_H-half_float*(_u*_u+_v*_v+_w*_w);
    real_t Gamma0 = get_CopGamma(thermal, _yi, _T); // out from RoeAverage_x
    real_t c2 = get_CopC2(z, thermal, _yi, _hi, _h, Gamma0, _T);
    // printf("argus at [%d],[%d][%d][%d]=yi=%lf, %lf, T=%lf,hi=%lf, %lf, _h=%lf, Gamma=%lf, c2=%lf, zi=%lf,%lf\n", id_l, i, j, k, _yi[0], _yi[1], T, _hi[0], _hi[1], _h, Gamma0, c2, z[0], z[1]); //_Cp
#else
    real_t Gamma0 = Gamma;
    real_t c2 = Gamma0 * (_H - half_float * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
    real_t z[NUM_SPECIES] = {0};
#endif

    RoeAverage_z(eigen_l, eigen_r, z, _yi, c2, _rho, _u, _v, _w, _H, D, D1, Gamma0);

    real_t uh[10], hh[10], pp[10], mm[10];
    real_t h_flux, _p[Emax][Emax];

    //construct the right value & the left value scalar equations by characteristic reduction
	// at k+1/2 in z direction
	for(int n=0; n<Emax; n++){
        #if USE_DP
        real_t eigen_local_max = 0.0;
#else
        real_t eigen_local_max = 0.0f;
#endif

        for(int m=-2; m<=3; m++){
            int id_local = Xmax*Ymax*(k + m) + Xmax*j + i;
            eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax*id_local+n]));//local lax-friedrichs	
        }
		for(int m=k-3; m<=k+4; m++){
            int id_local = Xmax*Ymax*m + Xmax*j + i;
            #if USE_DP
			uh[m-k+3] = 0.0;
			hh[m-k+3] = 0.0;
            #else
			uh[m-k+3] = 0.0f;
			hh[m-k+3] = 0.0f;
            #endif

			for(int n1=0; n1<Emax; n1++){
				uh[m-k+3] = uh[m-k+3] + UI[Emax*id_local+n1]*eigen_l[n][n1];
				hh[m-k+3] = hh[m-k+3] + Fz[Emax*id_local+n1]*eigen_l[n][n1];
			}
			//for local speed
			pp[m-k+3] = 0.5f*(hh[m-k+3] + eigen_local_max*uh[m-k+3]);  
			mm[m-k+3] = 0.5f*(hh[m-k+3] - eigen_local_max*uh[m-k+3]); 
        }
		// calculate the scalar numerical flux at y direction
        #if USE_DOUBLE
        h_flux = (weno5old_P(&pp[3], dz) + weno5old_M(&mm[3], dz))/6.0;
        #else
        h_flux = (weno5old_P(&pp[3], dz) + weno5old_M(&mm[3], dz))/6.0f;
        #endif
		
		// get Gp
		for(int n1=0; n1<Emax; n1++)
            _p[n][n1] = h_flux*eigen_r[n1][n];
    }
	// reconstruction the H-flux terms
	for(int n=0; n<Emax; n++){
        #if USE_DP
        real_t fluxz = 0.0;
#else
        real_t fluxz = 0.0f;
#endif
		for(int n1=0; n1<Emax; n1++){
            fluxz +=  _p[n1][n];
        }
        Fzwall[Emax*id_l+n]  = fluxz;
	}
}

extern SYCL_EXTERNAL void GetLocalEigen(int i, int j, int k, Block bl, real_t AA, real_t BB, real_t CC, real_t *eigen_local, real_t *u, real_t *v, real_t *w, real_t *c)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_X
    if(i >= Xmax)
    return;
    #endif
    #if DIM_Y
    if(j >= Ymax)
    return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
    return;
    #endif

    real_t uu = AA * u[id] + BB * v[id] + CC * w[id];
    real_t uuPc = uu + c[id];
    real_t uuMc = uu - c[id];
    //local eigen values
#ifdef COP
    eigen_local[Emax * id + 0] = uuMc;
    for (size_t ii = 1; ii < Emax - 1; ii++)
    {
    eigen_local[Emax * id + ii] = uu;
    }
    eigen_local[Emax * id + Emax - 1] = uuPc;
#else
    eigen_local[Emax * id + 0] = uuMc;
    eigen_local[Emax * id + 1] = uu;
    eigen_local[Emax * id + 2] = uu;
    eigen_local[Emax * id + 3] = uu;
    eigen_local[Emax * id + 4] = uuPc;
#endif // COP
}

extern SYCL_EXTERNAL void UpdateFluidLU(int i, int j, int k, Block bl, real_t *LU, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    real_t dx = bl.dx;
    real_t dy = bl.dy;
    real_t dz = bl.dz;
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_X
    int id_im = Xmax*Ymax*k + Xmax*j + i - 1;
    #endif
    #if DIM_Y
    int id_jm = Xmax*Ymax*k + Xmax*(j-1) + i;
    #endif
    #if DIM_Z
    int id_km = Xmax*Ymax*(k-1) + Xmax*j + i;
    #endif

    for(int n=0; n<Emax; n++){
        #if USE_DP
    real_t LU0 = 0.0;
#else
        real_t LU0 = 0.0f;
#endif
        
        #if DIM_X
        LU0 += (FluxFw[Emax*id_im+n] - FluxFw[Emax*id+n])/dx;
        #endif
        #if DIM_Y
        LU0 += (FluxGw[Emax*id_jm+n] - FluxGw[Emax*id+n])/dy;
        #endif
        #if DIM_Z
        LU0 += (FluxHw[Emax*id_km+n] - FluxHw[Emax*id+n])/dz;
        #endif
        LU[Emax*id+n] = LU0;
    }
}

extern SYCL_EXTERNAL void UpdateFuidStatesKernel(int i, int j, int k, Block bl, Thermal *thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
                                                 real_t *rho, real_t *p, real_t *c, real_t *H, real_t *u, real_t *v, real_t *w, real_t *_y, real_t *T,
                                                 real_t const Gamma)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_X
    if(i >= Xmax)
        return;
    #endif
    #if DIM_Y
    if(j >= Ymax)
        return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
        return;
    #endif

    real_t U[Emax], yi[NUM_SPECIES];
    yi[0] = 1;
    for (size_t n = 0; n < Emax; n++)
    {
        U[n] = UI[Emax * id + n];
    }
    for (size_t ii = 0; ii < NUM_COP; ii++)
    { // calculate yi
        yi[ii + 1] = real_t(U[Emax - NUM_COP + ii]) / real_t(U[0]);
        _y[id * NUM_COP + ii] = yi[ii + 1];
        yi[0] += -yi[ii + 1];
    }
    // printf("U at [%d],[%d][%d][%d] before upate=%lf,%lf,%lf,%lf,%lf,%lf\n", id, i, j, k, U[0], U[1], U[2], U[3], U[4], U[5]);
    GetStates(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], T[id], thermal, yi, Gamma);

    real_t *Fx = &(FluxF[Emax * id]);
    real_t *Fy = &(FluxG[Emax * id]);
    real_t *Fz = &(FluxH[Emax * id]);

    GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);
}

extern SYCL_EXTERNAL void UpdateURK3rdKernel(int i, int j, int k, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int id = Xmax*Ymax*k + Xmax*j + i;

    switch(flag) {
        case 1:
        for(int n=0; n<Emax; n++)
            U1[Emax*id+n] = U[Emax*id+n] + dt*LU[Emax*id+n];
        break;
        case 2:
        for(int n=0; n<Emax; n++){
            #if USE_DP
            U1[Emax*id+n] = 0.75*U[Emax*id+n] + 0.25*U1[Emax*id+n] + 0.25*dt*LU[Emax*id+n];
            #else
            U1[Emax*id+n] = 0.75f*U[Emax*id+n] + 0.25f*U1[Emax*id+n] + 0.25f*dt*LU[Emax*id+n];
            #endif
        }   
        break;
        case 3:
        for(int n=0; n<Emax; n++){
            #if USE_DP
            U[Emax*id+n] = (U[Emax*id+n] + 2.0*U1[Emax*id+n])/3.0 + 2.0*dt*LU[Emax*id+n]/3.0;
            #else
            U[Emax*id+n] = (U[Emax*id+n] + 2.0f*U1[Emax*id+n])/3.0f + 2.0f*dt*LU[Emax*id+n]/3.0f;
            #endif
        }
        break;
    }
}

extern SYCL_EXTERNAL void FluidBCKernelX(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_Y
    if(j >= Ymax)
        return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
        return;
    #endif

    switch(BC) {
        case Symmetry:
        {
            int offset = 2*(Bwidth_X+mirror_offset)-1;
            int target_id = Xmax*Ymax*k + Xmax*j + (offset-i);
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
            d_UI[Emax*id+1] = -d_UI[Emax*target_id+1];
        }
        break;

        case Periodic:
        {
            int target_id = Xmax*Ymax*k + Xmax*j + (i + sign*X_inner);
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
            int target_id = Xmax*Ymax*k + Xmax*j + index_inner;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
        }
        break;

        case Wall:
        {
            int offset = 2*(Bwidth_X+mirror_offset)-1;
            int target_id = Xmax*Ymax*k + Xmax*j + (offset-i);
            d_UI[Emax*id+0] = d_UI[Emax*target_id+0];
            d_UI[Emax*id+1] = -d_UI[Emax*target_id+1];
            d_UI[Emax*id+2] = -d_UI[Emax*target_id+2];
            d_UI[Emax*id+3] = -d_UI[Emax*target_id+3];
            d_UI[Emax*id+4] = d_UI[Emax*target_id+4];
        }
        break;
    }
}

extern SYCL_EXTERNAL void FluidBCKernelY(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_X
    if(i >= Xmax)
        return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
        return;
    #endif

    switch(BC) {
        case Symmetry:
        {
            int offset = 2*(Bwidth_Y+mirror_offset)-1;
            int target_id = Xmax*Ymax*k + Xmax*(offset-j) + i;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
            d_UI[Emax*id+2] = -d_UI[Emax*target_id+2];
        }
        break;

        case Periodic:
        {
            int target_id = Xmax*Ymax*k + Xmax*(j + sign*Y_inner) + i;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
            int target_id = Xmax*Ymax*k + Xmax*index_inner + i;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
        }
        break;

        case Wall:
        {
            int offset = 2*(Bwidth_Y+mirror_offset)-1;
            int target_id = Xmax*Ymax*k + Xmax*(offset-j) + i;
            d_UI[Emax*id+0] = d_UI[Emax*target_id+0];
            d_UI[Emax*id+1] = -d_UI[Emax*target_id+1];
            d_UI[Emax*id+2] = -d_UI[Emax*target_id+2];
            d_UI[Emax*id+3] = -d_UI[Emax*target_id+3];
            d_UI[Emax*id+4] = d_UI[Emax*target_id+4];
        }
        break;
    }
}

extern SYCL_EXTERNAL void FluidBCKernelZ(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int Bwidth_X = bl.Bwidth_X;
    int Bwidth_Y = bl.Bwidth_Y;
    int Bwidth_Z = bl.Bwidth_Z;
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_X
    if(i >= Xmax)
        return;
    #endif
    #if DIM_Y
    if(j >= Ymax)
        return;
    #endif

    switch(BC) {
        case Symmetry:
        {
            int offset = 2*(Bwidth_Z+mirror_offset)-1;
            int target_id = Xmax*Ymax*(offset-k) + Xmax*j + i;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
            d_UI[Emax*id+3] = -d_UI[Emax*target_id+3];
        }
        break;

        case Periodic:
        {
            int target_id = Xmax*Ymax*(k + sign*Z_inner) + Xmax*j + i;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
            int target_id = Xmax*Ymax*index_inner + Xmax*j + i;
            for(int n=0; n<Emax; n++)	d_UI[Emax*id+n] = d_UI[Emax*target_id+n];
        }
        break;

        case Wall:
        {
            int offset = 2*(Bwidth_Z+mirror_offset)-1;
            int target_id = Xmax*Ymax*(k-offset) + Xmax*j + i;
            d_UI[Emax*id+0] = d_UI[Emax*target_id+0];
            d_UI[Emax*id+1] = -d_UI[Emax*target_id+1];
            d_UI[Emax*id+2] = -d_UI[Emax*target_id+2];
            d_UI[Emax*id+3] = -d_UI[Emax*target_id+3];
            d_UI[Emax*id+4] = d_UI[Emax*target_id+4];
        }
        break;
    }
}