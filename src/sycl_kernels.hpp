#pragma once
#include "include/global_class.h"
#include "device_func.hpp"

extern SYCL_EXTERNAL void InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal *thermal, real_t *U, real_t *U1, real_t *LU,
                                              real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
                                              real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *const *_y, real_t *T, real_t *H, real_t *c)
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

    int id = Xmax * Ymax * k + Xmax * j + i;

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
    // 1d shock tube case： no bubble
    if (d2 < ini.blast_center_x)
    {
        rho[id] = ini.blast_density_in;
        u[id] = ini.blast_u_in;
        v[id] = ini.blast_v_in;
        w[id] = ini.blast_w_in;
        p[id] = ini.blast_pressure_in;
#ifdef COP
#ifdef React
        for (size_t i = 0; i < NUM_SPECIES; i++)
        {
            _y[i][id] = thermal->species_ratio_out[i];
        }
#else
        _y[0][id] = ini.cop_y1_out;
#endif // end React
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
        {                 // in the bubble
            rho[id] = ini.cop_density_in;         // 气泡内单独赋值密度以和气泡外区分
            p[id] = ini.cop_pressure_in;          // 气泡内单独赋值压力以和气泡外区分
#ifdef React
            for (size_t i = 0; i < NUM_SPECIES; i++)
            { // set the last specie in .dat as fliud
                _y[i][id] = thermal->species_ratio_in[i];
            }
#else
            _y[0][id] = ini.cop_y1_in; // 组分气泡必须在激波下游
#endif // end React
        }
        else if (dy2 > copBout)
        { // out of bubble
            rho[id] = ini.blast_density_out;
            p[id] = ini.blast_pressure_out;
#ifdef React
            for (size_t i = 0; i < NUM_SPECIES; i++)
            {
                _y[i][id] = thermal->species_ratio_out[i];
            }
#else
            _y[0][id] = ini.cop_y1_out;
#endif // end React
        }
        else
        { // boundary of bubble && shock
            rho[id] = 0.5 * (ini.cop_density_in + ini.blast_density_out);
            p[id] = 0.5 * (ini.cop_pressure_in + ini.blast_pressure_out);
#ifdef React
            for (size_t i = 0; i < NUM_SPECIES; i++)
            {
                _y[i][id] = _DF(0.5) * (thermal->species_ratio_in[i] + thermal->species_ratio_out[i]);
            }
#else
            _y[0][id] = _DF(0.5) * (ini.cop_y1_out + ini.cop_y1_in);
#endif // end React
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

// EOS was included
#ifdef COP
    real_t yi[NUM_SPECIES];
    get_yi(_y, yi, id);
    real_t R = get_CopR(thermal->species_chara, yi);
    // T[id] = p[id] / rho[id] / R;
    T[id] = x < 0.5 ? _DF(900.0) : _DF(700.0); // TODO: for debug
    p[id] = x < 0.5 ? _DF(72200.0) : _DF(36100.0);
    u[id] = x < 0.5 ? _DF(10.0) : 0.0;
    v[id] = y < 0.5 ? _DF(0.0) : 10.0;
    rho[id] = p[id] / R / T[id];
    real_t Gamma_m = get_CopGamma(thermal, yi, T[id]);
    real_t h = get_Coph(thermal, yi, T[id]);
    // U[4] of mixture differ from pure gas
    U[Emax * id + 4] = rho[id] * (h + _DF(0.5) * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id])) - p[id];
    for (size_t ii = 5; ii < Emax; ii++)
    { // equations of species
        U[Emax * id + ii] = rho[id] * _y[ii - 5][id];
        FluxF[Emax * id + ii] = rho[id] * u[id] * _y[ii - 5][id];
        FluxG[Emax * id + ii] = rho[id] * v[id] * _y[ii - 5][id];
        FluxH[Emax * id + ii] = rho[id] * w[id] * _y[ii - 5][id];
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
    H[id] = (U[Emax * id + 4] + p[id]) / rho[id];
    // initial U[0]-U[3]
    U[Emax * id + 0] = rho[id];
    U[Emax * id + 1] = rho[id] * u[id];
    U[Emax * id + 2] = rho[id] * v[id];
    U[Emax * id + 3] = rho[id] * w[id];
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
        LU[Emax * id + n] = _DF(0.0);     // incremental of one time step
        U1[Emax * id + n] = U[Emax * id + n]; // intermediate conwervatives
        FluxFw[Emax * id + n] = _DF(0.0); // numerical flux F
        FluxGw[Emax * id + n] = _DF(0.0); // numerical flux G
        FluxHw[Emax * id + n] = _DF(0.0); // numerical flux H
#if NumFluid != 1
        CnsrvU[Emax * id + n] = U[Emax * id + n] * fraction;
        CnsrvU1[Emax * id + n] = CnsrvU[Emax * id + n];
#endif // NumFluid
    }
    // real_t de_U[Emax];
    // get_Array(U, de_U, Emax, id);
    // real_t de_UI[Emax];
}

/**
 * @brief calculate c^2 of the mixture at given point
 */
// NOTE: realted with yn=yi[0] or yi[N] : hi[] Ri[]
real_t get_CopC2(real_t z[NUM_SPECIES], Thermal *thermal, const real_t yi[NUM_SPECIES], real_t hi[NUM_SPECIES], const real_t h, const real_t gamma, const real_t T)
{
    real_t Sum_dpdrhoi = _DF(0.0);                 // Sum_dpdrhoi:first of c2,存在累加项
    real_t Ri[NUM_SPECIES], _dpdrhoi[NUM_SPECIES]; // hi[NUM_SPECIES]
    for (size_t n = 0; n < NUM_SPECIES; n++)
    {
        Ri[n] = Ru / thermal->species_chara[n * SPCH_Sz + 6];
        _dpdrhoi[n] = (gamma - _DF(1.0)) * (hi[NUM_COP] - hi[n]) + gamma * (Ri[n] - Ri[NUM_COP]) * T; // related with yi
        z[n] = -_DF(1.0) * _dpdrhoi[n] / (gamma - _DF(1.0));
        if (NUM_COP != n) // related with yi
            Sum_dpdrhoi += yi[n] * _dpdrhoi[n];
    }
    real_t _CopC2 = Sum_dpdrhoi + (gamma - _DF(1.0)) * (h - hi[NUM_COP]) + gamma * Ri[NUM_COP] * T; // related with yi
    return _CopC2;
}

extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Block bl, Thermal *thermal, real_t const Gamma, real_t *UI, real_t *Fx,
                                           real_t *Fxwall, real_t *eigen_local, real_t *rho, real_t *u, real_t *v, real_t *w,
                                           real_t *const *y, real_t *T, real_t *H)
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
    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*k + Xmax*j + i + 1;
    real_t dx = bl.dx;

    if(i>= X_inner+Bwidth_X)
        return;

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax];

    // preparing some interval value for roe average
    real_t D = sqrt(rho[id_r] / rho[id_l]);
    real_t D1 = _DF(1.0) / (D + _DF(1.0));

    real_t _u = (u[id_l] + D * u[id_r]) * D1;
    real_t _v = (v[id_l] + D * v[id_r]) * D1;
    real_t _w = (w[id_l] + D * w[id_r]) * D1;
    real_t _H = (H[id_l] + D * H[id_r]) * D1;
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

#ifdef COP
    real_t _T = (T[id_l] + D * T[id_r]) * D1;
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_SPECIES];
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
        real_t Ri = Ru / (thermal->species_chara[i * SPCH_Sz + 6]);
        hi_l[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], Ri, i);
        hi_r[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], Ri, i);
    }
    get_yi(y, yi_l, id_l);
    get_yi(y, yi_r, id_r);
    real_t _h = _DF(0.0);
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
    {
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;
        _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;
        _h += _hi[ii] * _yi[ii];
    }
    real_t Gamma0 = get_CopGamma(thermal, _yi, _T);              // out from RoeAverage_x , 使用半点的数据计算出半点处的Gamma
    real_t c2 = get_CopC2(z, thermal, _yi, _hi, _h, Gamma0, _T); // z[NUM_SPECIES] 是一个在该函数中同时计算的数组变量
#else
    real_t Gamma0 = Gamma;
    real_t c2 = Gamma0 * (_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
    real_t z[NUM_SPECIES] = {0};
#endif

    RoeAverage_x(eigen_l, eigen_r, z, _yi, c2, _rho, _u, _v, _w, _H, D, D1, Gamma0);

    real_t uf[10], ff[10], pp[10], mm[10];
    real_t f_flux, _p[Emax][Emax];

    // construct the right value & the left value scalar equations by characteristic reduction			
	// at i+1/2 in x direction
    // #pragma unroll Emax
	for(int n=0; n<Emax; n++){
        real_t eigen_local_max = _DF(0.0);
        for(int m=-2; m<=3; m++){
            int id_local = Xmax * Ymax * k + Xmax * j + i + m;
            eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax * id_local + n])); // local lax-friedrichs
        }

		for(int m=i-3; m<=i+4; m++){	// 3rd oder and can be modified
            int id_local = Xmax * Ymax * k + Xmax * j + m;

            uf[m - i + 3] = _DF(0.0);
            ff[m - i + 3] = _DF(0.0);

            for (int n1 = 0; n1 < Emax; n1++)
            {
                uf[m - i + 3] = uf[m - i + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];
                ff[m - i + 3] = ff[m - i + 3] + Fx[Emax * id_local + n1] * eigen_l[n][n1];
            }
            // for local speed
            pp[m - i + 3] = _DF(0.5) * (ff[m - i + 3] + eigen_local_max * uf[m - i + 3]);
            mm[m - i + 3] = _DF(0.5) * (ff[m - i + 3] - eigen_local_max * uf[m - i + 3]);
        }

		// calculate the scalar numerical flux at x direction
        f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx)) / _DF(6.0);

        // get Fp
        for (int n1 = 0; n1 < Emax; n1++)
            _p[n][n1] = f_flux * eigen_r[n1][n];
    }

	// reconstruction the F-flux terms
	for(int n=0; n<Emax; n++){
        real_t fluxx = _DF(0.0);
        for (int n1 = 0; n1 < Emax; n1++)
        {
            fluxx += _p[n1][n];
        }
        Fxwall[Emax*id_l+n] = fluxx;
	}

    // real_t de_fw[Emax];
    // get_Array(Fxwall, de_fw, Emax, id_l);
    // real_t de_fx[Emax];
}

extern SYCL_EXTERNAL void ReconstructFluxY(int i, int j, int k, Block bl, Thermal *thermal, real_t const Gamma, real_t *UI, real_t *Fy,
                                           real_t *Fywall, real_t *eigen_local, real_t *rho, real_t *u, real_t *v, real_t *w,
                                           real_t *const *y, real_t *T, real_t *H)
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
    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*k + Xmax*(j+ 1) + i;
    real_t dy = bl.dy;

    if(j>= Y_inner+Bwidth_Y)
        return;

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax];

    //preparing some interval value for roe average
    real_t D = sqrt(rho[id_r] / rho[id_l]);
    real_t D1 = _DF(1.0) / (D + _DF(1.0));
    real_t _u = (u[id_l] + D * u[id_r]) * D1;
    real_t _v = (v[id_l] + D * v[id_r]) * D1;
    real_t _w = (w[id_l] + D * w[id_r]) * D1;
    real_t _H = (H[id_l] + D * H[id_r]) * D1;
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

#ifdef COP
    real_t _T = (T[id_l] + D * T[id_r]) * D1;
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_SPECIES];
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
    real_t Gamma0 = get_CopGamma(thermal, _yi, _T); // out from RoeAverage_x
    real_t c2 = get_CopC2(z, thermal, _yi, _hi, _h, Gamma0, _T);
#else
    real_t Gamma0 = Gamma;
    real_t c2 = Gamma0 * (_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
    real_t z[NUM_SPECIES] = {0};
#endif

    RoeAverage_y(eigen_l, eigen_r, z, _yi, c2, _rho, _u, _v, _w, _H, D, D1, Gamma0);

    real_t ug[10], gg[10], pp[10], mm[10];
    real_t g_flux, _p[Emax][Emax];

    //construct the right value & the left value scalar equations by characteristic reduction			
	// at j+1/2 in y direction
	for(int n=0; n<Emax; n++){
        real_t eigen_local_max = _DF(0.0);

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
            pp[m - j + 3] = _DF(0.5) * (gg[m - j + 3] + eigen_local_max * ug[m - j + 3]);
            mm[m - j + 3] = _DF(0.5) * (gg[m - j + 3] - eigen_local_max * ug[m - j + 3]);
        }
		// calculate the scalar numerical flux at y direction
        g_flux = (weno5old_P(&pp[3], dy) + weno5old_M(&mm[3], dy)) / _DF(6.0);
        // get Gp
        for (int n1 = 0; n1 < Emax; n1++)
            _p[n][n1] = g_flux * eigen_r[n1][n];
    }
	// reconstruction the G-flux terms
	for(int n=0; n<Emax; n++){
        real_t fluxy = _DF(0.0);
        for (int n1 = 0; n1 < Emax; n1++)
        {
            fluxy += _p[n1][n];
        }
        Fywall[Emax*id_l+n] = fluxy;
	}
}

extern SYCL_EXTERNAL void ReconstructFluxZ(int i, int j, int k, Block bl, Thermal *thermal, real_t const Gamma, real_t *UI, real_t *Fz,
                                           real_t *Fzwall, real_t *eigen_local, real_t *rho, real_t *u, real_t *v, real_t *w,
                                           real_t *const *y, real_t *T, real_t *H)
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
    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*(k+1) + Xmax*j + i;
    real_t dz = bl.dz;

    if(k>= Z_inner+Bwidth_Z)
        return;

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax];

    //preparing some interval value for roe average
    real_t D = sqrt(rho[id_r] / rho[id_l]);
    real_t D1 = _DF(1.0) / (D + _DF(1.0));
    real_t _u = (u[id_l] + D * u[id_r]) * D1;
    real_t _v = (v[id_l] + D * v[id_r]) * D1;
    real_t _w = (w[id_l] + D * w[id_r]) * D1;
    real_t _H = (H[id_l] + D * H[id_r]) * D1;
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

#ifdef COP
    real_t _T = (T[id_l] + D * T[id_r]) * D1;
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_SPECIES];
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
    real_t Gamma0 = get_CopGamma(thermal, _yi, _T); // out from RoeAverage_x
    real_t c2 = get_CopC2(z, thermal, _yi, _hi, _h, Gamma0, _T);
#else
    real_t Gamma0 = Gamma;
    real_t c2 = Gamma0 * (_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
    real_t z[NUM_SPECIES] = {0};
#endif

    RoeAverage_z(eigen_l, eigen_r, z, _yi, c2, _rho, _u, _v, _w, _H, D, D1, Gamma0);

    real_t uh[10], hh[10], pp[10], mm[10];
    real_t h_flux, _p[Emax][Emax];

    //construct the right value & the left value scalar equations by characteristic reduction
	// at k+1/2 in z direction
	for(int n=0; n<Emax; n++){
        real_t eigen_local_max = _DF(0.0);

        for(int m=-2; m<=3; m++){
            int id_local = Xmax*Ymax*(k + m) + Xmax*j + i;
            eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax*id_local+n]));//local lax-friedrichs	
        }
		for(int m=k-3; m<=k+4; m++){
            int id_local = Xmax*Ymax*m + Xmax*j + i;
            uh[m - k + 3] = _DF(0.0);
            hh[m - k + 3] = _DF(0.0);

            for (int n1 = 0; n1 < Emax; n1++)
            {
                uh[m - k + 3] = uh[m - k + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];
                hh[m - k + 3] = hh[m - k + 3] + Fz[Emax * id_local + n1] * eigen_l[n][n1];
            }
            // for local speed
            pp[m - k + 3] = _DF(0.5) * (hh[m - k + 3] + eigen_local_max * uh[m - k + 3]);
            mm[m - k + 3] = _DF(0.5) * (hh[m - k + 3] - eigen_local_max * uh[m - k + 3]);
        }
		// calculate the scalar numerical flux at y direction
        h_flux = (weno5old_P(&pp[3], dz) + weno5old_M(&mm[3], dz)) / _DF(6.0);

        // get Gp
        for (int n1 = 0; n1 < Emax; n1++)
            _p[n][n1] = h_flux*eigen_r[n1][n];
    }
	// reconstruction the H-flux terms
	for(int n=0; n<Emax; n++){
        real_t fluxz = _DF(0.0);
        for (int n1 = 0; n1 < Emax; n1++)
        {
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
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
    int id_im = Xmax * Ymax * k + Xmax * j + i - 1;
#endif
#if DIM_Y
    int id_jm = Xmax * Ymax * k + Xmax * (j - 1) + i;
#endif
#if DIM_Z
    int id_km = Xmax * Ymax * (k - 1) + Xmax * j + i;
#endif

    for(int n=0; n<Emax; n++){
    real_t LU0 = _DF(0.0);
#if DIM_X
    LU0 += (FluxFw[Emax * id_im + n] - FluxFw[Emax * id + n]) / bl.dx;
#endif
#if DIM_Y
    LU0 += (FluxGw[Emax * id_jm + n] - FluxGw[Emax * id + n]) / bl.dy;
#endif
#if DIM_Z
    LU0 += (FluxHw[Emax * id_km + n] - FluxHw[Emax * id + n]) / bl.dz;
#endif
    LU[Emax * id + n] = LU0;
    }
}

extern SYCL_EXTERNAL void UpdateFuidStatesKernel(int i, int j, int k, Block bl, Thermal *thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
                                                 real_t *rho, real_t *p, real_t *c, real_t *H, real_t *u, real_t *v, real_t *w, real_t *const *_y, real_t *T,
                                                 real_t const Gamma)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int id = Xmax * Ymax * k + Xmax * j + i;

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

    real_t U[Emax], yi[NUM_SPECIES];
    for (size_t n = 0; n < Emax; n++)
    {
        U[n] = UI[Emax * id + n];
    }
    yi[NUM_COP] = _DF(1.0);
    for (size_t ii = 5; ii < Emax; ii++)
    { // calculate yi
        yi[ii - 5] = U[ii] / U[0];
        _y[ii - 5][id] = yi[ii - 5];
        yi[NUM_COP] += -yi[ii - 5];
    }
    _y[NUM_COP][id] = yi[NUM_COP];

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
    int id = Xmax * Ymax * k + Xmax * j + i;

    switch(flag) {
        case 1:
        for(int n=0; n<Emax; n++)
            U1[Emax * id + n] = U[Emax * id + n] + dt * LU[Emax * id + n];
        break;
        case 2:
        for(int n=0; n<Emax; n++){
            U1[Emax * id + n] = _DF(0.75) * U[Emax * id + n] + _DF(0.25) * U1[Emax * id + n] + _DF(0.25) * dt * LU[Emax * id + n];
        }   
        break;
        case 3:
        for(int n=0; n<Emax; n++){
            U[Emax * id + n] = (U[Emax * id + n] + _DF(2.0) * U1[Emax * id + n]) / _DF(3.0) + _DF(2.0) * dt * LU[Emax * id + n] / _DF(3.0);
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
        int offset = 2 * (Bwidth_X + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        }
        break;

        case Periodic:
        {
        int target_id = Xmax * Ymax * k + Xmax * j + (i + sign * X_inner);
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
        int target_id = Xmax * Ymax * k + Xmax * j + index_inner;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Wall:
        {
        int offset = 2 * (Bwidth_X + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
        d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
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
        int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        }
        break;

        case Periodic:
        {
        int target_id = Xmax * Ymax * k + Xmax * (j + sign * Y_inner) + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
        int target_id = Xmax * Ymax * k + Xmax * index_inner + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Wall:
        {
        int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
        d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
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
    int id = Xmax * Ymax * k + Xmax * j + i;

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
        int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
        int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        }
        break;

        case Periodic:
        {
        int target_id = Xmax * Ymax * (k + sign * Z_inner) + Xmax * j + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
        int target_id = Xmax * Ymax * index_inner + Xmax * j + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Wall:
        {
        int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
        int target_id = Xmax * Ymax * (k - offset) + Xmax * j + i;
        d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
        }
        break;
    }
}

#ifdef Visc
extern SYCL_EXTERNAL void GetInnerCellCenterDerivativeKernel(int i, int j, int k, Block bl, real_t *u, real_t *v, real_t *w, real_t **Vde)
{
#if DIM_X
    if (i > bl.Xmax - bl.Bwidth_X + 1)
        return;
#endif // DIM_X
#if DIM_Y
    if (i > bl.Ymax - bl.Bwidth_Y + 1)
            return;
#endif // DIM_Y
#if DIM_Z
    if (i > bl.Zmax - bl.Bwidth_Z + 1)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;

#if DIM_X
    real_t dx = bl.dx;
    int id_m1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
    int id_m2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
    int id_p1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
    int id_p2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;

    Vde[ducx][id] = (_DF(8.0) * (u[id_p1_x] - u[id_m1_x]) - (u[id_p2_x] - u[id_m2_x])) / dx / _DF(12.0);
    Vde[dvcx][id] = DIM_Y ? (_DF(8.0) * (v[id_p1_x] - v[id_m1_x]) - (v[id_p2_x] - v[id_m2_x])) / dx / _DF(12.0) : _DF(0.0);
    Vde[dwcx][id] = DIM_Z ? (_DF(8.0) * (w[id_p1_x] - w[id_m1_x]) - (w[id_p2_x] - w[id_m2_x])) / dx / _DF(12.0) : _DF(0.0);
#else
    Vde[ducx][id] = _DF(0.0);
    Vde[dvcx][id] = _DF(0.0);
    Vde[dwcx][id] = _DF(0.0);
#endif // end DIM_X
#if DIM_Y
    real_t dy = bl.dy;
    int id_m1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
    int id_m2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
    int id_p1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
    int id_p2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;

    Vde[ducy][id] = DIM_X ? (_DF(8.0) * (u[id_p1_y] - u[id_m1_y]) - (u[id_p2_y] - u[id_m2_y])) / dy / _DF(12.0) : _DF(0.0);
    Vde[dvcy][id] = (_DF(8.0) * (v[id_p1_y] - v[id_m1_y]) - (v[id_p2_y] - v[id_m2_y])) / dy / _DF(12.0);
    Vde[dwcy][id] = DIM_Z ? (_DF(8.0) * (w[id_p1_y] - w[id_m1_y]) - (w[id_p2_y] - w[id_m2_y])) / dy / _DF(12.0) : _DF(0.0);
#else
    Vde[ducy][id] = _DF(0.0);
    Vde[dvcy][id] = _DF(0.0);
    Vde[dwcy][id] = _DF(0.0);
#endif // end DIM_Y
#if DIM_Z
    real_t dz = bl.dz;
    int id_m1_z = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
    int id_m2_z = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
    int id_p1_z = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
    int id_p2_z = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;

    Vde[ducz][id] = DIM_X ? (_DF(8.0) * (u[id_p1_z] - u[id_m1_z]) - (u[id_p2_z] - u[id_m2_z])) / dz / _DF(12.0) : _DF(0.0);
    Vde[dvcz][id] = DIM_Y ? (_DF(8.0) * (v[id_p1_z] - v[id_m1_z]) - (v[id_p2_z] - v[id_m2_z])) / dz / _DF(12.0) : _DF(0.0);
    Vde[dwcz][id] = (_DF(8.0) * (w[id_p1_z] - w[id_m1_z]) - (w[id_p2_z] - w[id_m2_z])) / dz / _DF(12.0);
#else
    Vde[ducz][id] = _DF(0.0);
    Vde[dvcz][id] = _DF(0.0);
    Vde[dwcz][id] = _DF(0.0);
#endif // end DIM_Z
}

extern SYCL_EXTERNAL void CenterDerivativeBCKernelX(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
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
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_Y
    if (j >= Ymax)
            return;
#endif
#if DIM_Z
    if (k >= Zmax)
            return;
#endif

    switch (BC)
    {
    case Symmetry:
    {
            int offset = 2 * (Bwidth_X + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Periodic:
    {
            int target_id = Xmax * Ymax * k + Xmax * j + (i + sign * X_inner);
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Inflow:
            for (int n = 0; n < 4; n++)
            Vde[n][id] = real_t(0.0f);
            break;

    case Outflow:
    {
            int target_id = Xmax * Ymax * k + Xmax * j + index_inner;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Wall:
    {
            int offset = 2 * (Bwidth_X + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
            for (int n = 0; n < 4; n++)
            Vde[n][id] = -Vde[n][target_id];
    }
    break;
    }
}

extern SYCL_EXTERNAL void CenterDerivativeBCKernelY(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
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
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
    if (i >= Xmax)
            return;
#endif
#if DIM_Z
    if (k >= Zmax)
            return;
#endif

    switch (BC)
    {
    case Symmetry:
    {
            int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Periodic:
    {
            int target_id = Xmax * Ymax * k + Xmax * (j + sign * Y_inner) + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Inflow:
            for (int n = 0; n < 4; n++)
            Vde[n][id] = real_t(0.0f);
            break;

    case Outflow:
    {
            int target_id = Xmax * Ymax * k + Xmax * index_inner + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Wall:
    {
            int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = -Vde[n][target_id];
    }
    break;
    }
}

extern SYCL_EXTERNAL void CenterDerivativeBCKernelZ(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
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
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
    if (i >= Xmax)
            return;
#endif
#if DIM_Y
    if (j >= Ymax)
            return;
#endif

    switch (BC)
    {
    case Symmetry:
    {
            int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
            int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Periodic:
    {
            int target_id = Xmax * Ymax * (k + sign * Z_inner) + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Inflow:
            for (int n = 0; n < 4; n++)
            Vde[n][id] = real_t(0.0f);
            break;

    case Outflow:
    {
            int target_id = Xmax * Ymax * index_inner + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = Vde[n][target_id];
    }
    break;

    case Wall:
    {
            int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
            int target_id = Xmax * Ymax * (k - offset) + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde[n][id] = -Vde[n][target_id];
    }
    break;
    }
}

extern SYCL_EXTERNAL void Gettransport_coeff_aver(int i, int j, int k, Block bl, Thermal *thermal, real_t *viscosity_aver, real_t *thermal_conduct_aver,
                                                  real_t *Dkm_aver, real_t *const *y, real_t *hi, real_t *rho, real_t *p, real_t *T)
{
#ifdef DIM_X
    if (i >= bl.Xmax)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (i >= bl.Ymax)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (i >= bl.Zmax)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    // get mole fraction of each specie
    real_t X[NUM_SPECIES] = {_DF(0.0)}, yi[NUM_SPECIES] = {_DF(0.0)};
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
            real_t Ri = Ru / thermal->species_chara[i * SPCH_Sz + 6];
            hi[i + NUM_SPECIES * id] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id], Ri, i);
    }
    get_yi(y, yi, id);
    real_t C_total = get_xi(X, yi, thermal->species_chara);
    // calculate viscosity_aver via equattion(5-49)//
    Get_transport_coeff_aver(thermal, &(Dkm_aver[NUM_SPECIES * id]), viscosity_aver[id], thermal_conduct_aver[id], X, rho[id], p[id], T[id], C_total);
}

#if DIM_X
extern SYCL_EXTERNAL void GetWallViscousFluxX(int i, int j, int k, Block bl, real_t *FluxFw, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *const *Yi, real_t *const *V, real_t **Vde)
{ // compute Physical、Heat、Diffu viscity in this function
#ifdef DIM_X
    if (i >= bl.X_inner + bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (i >= bl.Y_inner + bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (i >= bl.Z_inner + bl.Bwidth_Z)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    int id_m1 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
    int id_m2 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
    int id_p1 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
    int id_p2 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;
    real_t *Ducy = Vde[ducy];
    real_t *Ducz = Vde[ducz];
    real_t *Dvcy = Vde[dvcy];
    real_t *Dwcz = Vde[dwcz];
    real_t *u = V[0];
    real_t *v = V[1];
    real_t *w = V[2];
    real_t dx = bl.dx;

    real_t F_x_wall_v[Emax];
    real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    real_t lamada = -_DF(2.0) / _DF(3.0) * mue;
    real_t f_x, f_y, f_z;
    real_t u_hlf, v_hlf, w_hlf;
    f_x = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) / dx / _DF(24.0);
    f_y = DIM_Y ? mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) / dx / _DF(24.0) : _DF(0.0);
    f_z = DIM_Z ? mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) / dx / _DF(24.0) : _DF(0.0);

    f_x += lamada * (_DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) / _DF(16.0);
    f_y += DIM_Y ? mue * (_DF(9.0) * (Ducy[id_p1] + Ducy[id]) - (Ducy[id_p2] + Ducy[id_m1])) / _DF(16.0) : _DF(0.0);
    f_z += DIM_Z ? mue * (_DF(9.0) * (Ducz[id_p1] + Ducz[id]) - (Ducz[id_p2] + Ducz[id_m1])) / _DF(16.0) : _DF(0.0);

    u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) / _DF(16.0);
    v_hlf = DIM_Y ? (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) / _DF(16.0) : _DF(0.0);
    w_hlf = DIM_Z ? (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) / _DF(16.0) : _DF(0.0);
    F_x_wall_v[0] = _DF(0.0);
    F_x_wall_v[1] = f_x;
    F_x_wall_v[2] = f_y;
    F_x_wall_v[3] = f_z;
    F_x_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

#ifdef Heat    // Fourier thermal conductivity
    real_t kk; // thermal conductivity at wall
    kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0);
    real_t tempo = _DF(0.0);
    tempo = (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dx / _DF(24.0);
    kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dx / _DF(24.0);                                                // temperature gradient at wall
    F_x_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
#endif                                                                                                                              // end Heat
#ifdef Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
    real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
    real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yix_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
    for (int l = 0; l < NUM_SPECIES; l++)
    {
            hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
            Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
            Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
            Yix_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dx / _DF(24.0); // temperature gradient at wall
    }
#ifdef Heat
    for (int l = 0; l < NUM_SPECIES; l++)
            F_x_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yix_wall[l];
#endif     // end Heat
#ifdef COP // visc flux for cop equations
    real_t CorrectTermX = _DF(0.0);
    for (int l = 0; l < NUM_SPECIES; l++)
            CorrectTermX += Dim_wall[l] * Yix_wall[l];
    CorrectTermX *= rho_wall;
    // ADD Correction Term in X-direction
    for (int p = 5; p < Emax; p++)
            F_x_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yix_wall[p - 5] - Yi_wall[p - 5] * CorrectTermX;
#endif // end COP
#endif // end Diffu
    for (size_t i = 0; i < Emax; i++)
    { // add viscous flux to fluxwall
            FluxFw[i + Emax * id] -= F_x_wall_v[i];
    }
}
#endif // end DIM_X

#if DIM_Y
extern SYCL_EXTERNAL void GetWallViscousFluxY(int i, int j, int k, Block bl, real_t *FluxGw, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *const *Yi, real_t *const *V, real_t **Vde)
{ // compute Physical、Heat、Diffu viscity in this function
#ifdef DIM_X
    if (i >= bl.X_inner + bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (i >= bl.Y_inner + bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (i >= bl.Z_inner + bl.Bwidth_Z)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    int id_m1 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
    int id_m2 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
    int id_p1 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
    int id_p2 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;
    real_t *Dvcx = Vde[dvcx];
    real_t *Dvcz = Vde[dvcz];
    real_t *Ducx = Vde[ducx];
    real_t *Dwcz = Vde[dwcz];
    real_t *u = V[0];
    real_t *v = V[1];
    real_t *w = V[2];
    real_t dy = bl.dy;

    // mue at wall
    real_t F_y_wall_v[Emax];
    real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    real_t lamada = -_DF(2.0) / _DF(3.0) * mue;
    real_t f_x, f_y, f_z;
    real_t u_hlf, v_hlf, w_hlf;
    f_x = DIM_X ? mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) / dy / _DF(24.0) : _DF(0.0);
    f_y = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) / dy / _DF(24.0);
    f_z = DIM_Z ? mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) / dy / _DF(24.0) : _DF(0.0);

    f_x += DIM_X ? mue * (_DF(9.0) * (Dvcx[id_p1] + Dvcx[id]) - (Dvcx[id_p2] + Dvcx[id_m1])) / _DF(16.0) : _DF(0.0);
    f_y += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) / _DF(16.0);
    f_z += DIM_Z ? mue * (_DF(9.0) * (Dvcz[id_p1] + Dvcz[id]) - (Dvcz[id_p2] + Dvcz[id_m1])) / _DF(16.0) : _DF(0.0);

    u_hlf = DIM_X ? (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) / _DF(16.0) : _DF(0.0);
    v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) / _DF(16.0);
    w_hlf = DIM_Z ? (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) / _DF(16.0) : _DF(0.0);
    F_y_wall_v[0] = _DF(0.0);
    F_y_wall_v[1] = f_x;
    F_y_wall_v[2] = f_y;
    F_y_wall_v[3] = f_z;
    F_y_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

#ifdef Heat    // Fourier thermal conductivity
    real_t kk; // thermal conductivity at wall
    kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0);
    kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dy / _DF(24.0);                                                // temperature gradient at wall
    F_y_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
#endif                                                                                                                              // end Heat
#ifdef Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
    real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
    real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yiy_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
    for (int l = 0; l < NUM_SPECIES; l++)
    {
            hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
            Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
            Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
            Yiy_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dy / _DF(24.0); // temperature gradient at wal
    }
#ifdef Heat
    for (int l = 0; l < NUM_SPECIES; l++)
            F_y_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yiy_wall[l];
#endif     // end Heat
#ifdef COP // visc flux for cop equations
    real_t CorrectTermY = _DF(0.0);
    for (int l = 0; l < NUM_SPECIES; l++)
            CorrectTermY += Dim_wall[l] * Yiy_wall[l];
    CorrectTermY *= rho_wall;
    // ADD Correction Term in X-direction
    for (int p = 5; p < Emax; p++)
            F_y_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yiy_wall[p - 5] - Yi_wall[p - 5] * CorrectTermY;
#endif // end COP
#endif // end Diffu
    for (size_t i = 0; i < Emax; i++)
    { // add viscous flux to fluxwall
            FluxGw[i + Emax * id] -= F_y_wall_v[i];
    }
}
#endif // end DIM_Y

#if DIM_Z
extern SYCL_EXTERNAL void GetWallViscousFluxZ(int i, int j, int k, Block bl, real_t *FluxHw, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *const *Yi, real_t *const *V, real_t **Vde)
{ // compute Physical、Heat、Diffu viscity in this function
#ifdef DIM_X
    if (i >= bl.X_inner + bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (i >= bl.Y_inner + bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (i >= bl.Z_inner + bl.Bwidth_Z)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    int id_m1 = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
    int id_m2 = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
    int id_p1 = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
    int id_p2 = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;
    real_t *Dwcx = Vde[dwcx];
    real_t *Dwcy = Vde[dwcy];
    real_t *Ducx = Vde[ducx];
    real_t *Dvcy = Vde[dvcy];
    real_t *u = V[0];
    real_t *v = V[1];
    real_t *w = V[2];
    real_t dz = bl.dz;

    // mue at wall
    real_t F_z_wall_v[Emax];
    real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    real_t lamada = -_DF(2.0) / _DF(3.0) * mue;
    real_t f_x, f_y, f_z;
    real_t u_hlf, v_hlf, w_hlf;
    f_x = DIM_X ? mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) / dz / _DF(24.0) : _DF(0.0);
    f_y = DIM_Y ? mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) / dz / _DF(24.0) : _DF(0.0);
    f_z = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) / dz / _DF(24.0);

    f_x += DIM_X ? mue * (_DF(9.0) * (Dwcx[id_p1] + Dwcx[id]) - (Dwcx[id_p2] + Dwcx[id_m1])) / _DF(16.0) : _DF(0.0);
    f_y += DIM_Y ? mue * (_DF(9.0) * (Dwcy[id_p1] + Dwcy[id]) - (Dwcy[id_p2] + Dwcy[id_m1])) / _DF(16.0) : _DF(0.0);
    f_z += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1])) / _DF(16.0);
    u_hlf = DIM_X ? (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) / _DF(16.0) : _DF(0.0);
    v_hlf = DIM_Y ? (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) / _DF(16.0) : _DF(0.0);
    w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) / _DF(16.0);
    F_z_wall_v[0] = _DF(0.0);
    F_z_wall_v[1] = f_x;
    F_z_wall_v[2] = f_y;
    F_z_wall_v[3] = f_z;
    F_z_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

#ifdef Heat    // Fourier thermal conductivity
    real_t kk; // thermal conductivity at wall
    kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id_m1])) / _DF(16.0);
    kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dz / _DF(24.0);                                                // temperature gradient at wall
    F_z_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
#endif                                                                                                                              // end Heat
#ifdef Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
    real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
    real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yiz_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
    for (int l = 0; l < NUM_SPECIES; l++)
    {
            hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
            Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
            Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
            Yiz_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dz / _DF(24.0); // temperature gradient at wall
    }
#ifdef Heat
    for (int l = 0; l < NUM_SPECIES; l++)
            F_z_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yiz_wall[l];
#endif     // end Heat
#ifdef COP // visc flux for cop equations
    real_t CorrectTermZ = _DF(0.0);
    for (int l = 0; l < NUM_SPECIES; l++)
            CorrectTermZ += Dim_wall[l] * Yiz_wall[l];
    CorrectTermZ *= rho_wall;
    // ADD Correction Term in X-direction
    for (int p = 5; p < Emax; p++)
            F_z_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yiz_wall[p - 5] - Yi_wall[p - 5] * CorrectTermZ;
#endif // end COP
#endif // end Diffu
    for (size_t i = 0; i < Emax; i++)
    { // add viscous flux to fluxwall
            FluxHw[i + Emax * id] -= F_z_wall_v[i];
    }
}
#endif // end DIM_Z
#endif // Visc

#ifdef React
extern SYCL_EXTERNAL void FluidODESolverKernel(int i, int j, int k, Block bl, Thermal *thermal, Reaction *react, real_t *UI, real_t *const *y, real_t *rho, real_t *T, const real_t dt)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int Zmax = bl.Zmax;
    int X_inner = bl.X_inner;
    int Y_inner = bl.Y_inner;
    int Z_inner = bl.Z_inner;
    int id = Xmax * Ymax * k + Xmax * j + i;

    real_t yi[NUM_SPECIES], Kf[NUM_REA], Kb[NUM_REA], U[Emax - NUM_COP];
    get_yi(y, yi, id);
    get_KbKf(Kf, Kb, react->Rargus, thermal->species_chara, thermal->Hia, thermal->Hib, react->Nu_d_, T[id]); // get_e
    for (size_t n = 0; n < Emax - NUM_COP; n++)
    {
            U[n] = UI[Emax * id + n];
    }
    real_t rho1 = _DF(1.0) / U[0];
    real_t u = U[1] * rho1;
    real_t v = U[2] * rho1;
    real_t w = U[3] * rho1;
    real_t e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
    Chemeq2(thermal, Kf, Kb, react->React_ThirdCoef, react->Rargus, react->Nu_b_, react->Nu_f_, react->Nu_d_, react->third_ind,
            react->reaction_list, react->reactant_list, react->product_list, react->rns, react->rts, react->pls, yi, dt, T[id], rho[id], e);
    // update partial density according to C0
    for (int n = 0; n < NUM_COP; n++)
    { // NOTE: related with yi[n]
            UI[Emax * id + n + 5] = yi[n] * rho[id];
    }
}
#endif // React