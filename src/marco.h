#pragma once
// =======================================================
// repeated code definitions
// =======================================================
/**
 * set Domain size
 */
#define MARCO_DOMAIN()        \
    int Xmax = bl.Xmax;       \
    int Ymax = bl.Ymax;       \
    int Zmax = bl.Zmax;       \
    int X_inner = bl.X_inner; \
    int Y_inner = bl.Y_inner; \
    int Z_inner = bl.Z_inner;
/**
 * set Domain size
 */
#define MARCO_DOMAIN_GHOST()    \
    int Xmax = bl.Xmax;         \
    int Ymax = bl.Ymax;         \
    int Zmax = bl.Zmax;         \
    int X_inner = bl.X_inner;   \
    int Y_inner = bl.Y_inner;   \
    int Z_inner = bl.Z_inner;   \
    int Bwidth_X = bl.Bwidth_X; \
    int Bwidth_Y = bl.Bwidth_Y; \
    int Bwidth_Z = bl.Bwidth_Z;
/**
 * get Roe values insde Reconstructflux
 */
#define MARCO_ROE()                           \
    real_t D = sqrt(rho[id_r] / rho[id_l]);   \
    real_t D1 = _DF(1.0) / (D + _DF(1.0));    \
    real_t _u = (u[id_l] + D * u[id_r]) * D1; \
    real_t _v = (v[id_l] + D * v[id_r]) * D1; \
    real_t _w = (w[id_l] + D * w[id_r]) * D1; \
    real_t _H = (H[id_l] + D * H[id_r]) * D1; \
    real_t _P = (p[id_l] + D * p[id_r]) * D1; \
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

/**
 * get c2 #ifdef COP inside Reconstructflux
 */
#define MARCO_COPC2()                                                                                                                                                                                                                        \
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], /*_hi[NUM_SPECIES],*/ hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_COP], b1 = _DF(0.0), b3 = _DF(0.0);                                                                     \
    for (size_t n = 0; n < NUM_SPECIES; n++)                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                        \
        hi_l[n] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], thermal->Ri[n], n);                                                                                                                                                      \
        hi_r[n] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], thermal->Ri[n], n);                                                                                                                                                      \
    }                                                                                                                                                                                                                                        \
    get_yi(y, yi_l, id_l);                                                                                                                                                                                                                   \
    get_yi(y, yi_r, id_r);                                                                                                                                                                                                                   \
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                                                                                                              \
    {                                                                                                                                                                                                                                        \
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                                                                                                                            \
        /*_hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;*/                                                                                                                                                                                        \
    }                                                                                                                                                                                                                                        \
    real_t _dpdrhoi[NUM_COP], drhoi[NUM_COP];                                                                                                                                                                                                \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                                                                                                                   \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                                                                                                                   \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                                                                                                                 \
    real_t q2_l = u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l];                                                                                                                                                                 \
    real_t q2_r = u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r];                                                                                                                                                                 \
    real_t e_l = H[id_l] - _DF(0.5) * q2_l - p[id_l] / rho[id_l];                                                                                                                                                                            \
    real_t e_r = H[id_r] - _DF(0.5) * q2_r - p[id_r] / rho[id_r];                                                                                                                                                                            \
    real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                                                                                                                         \
    real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                                                                                                                         \
    real_t R_l = get_CopR(thermal->species_chara, yi_l);                                                                                                                                                                                     \
    real_t R_r = get_CopR(thermal->species_chara, yi_r);                                                                                                                                                                                     \
    real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], thermal->Ri[NUM_COP], q2_l, Cp_l, R_l, T[id_l], e_l, gamma_l),                                                                                                                 \
                                    get_DpDrho(hi_r[NUM_COP], thermal->Ri[NUM_COP], q2_r, Cp_r, R_r, T[id_r], e_r, gamma_r), D, D1);                                                                                                         \
    for (size_t nn = 0; nn < NUM_COP; nn++)                                                                                                                                                                                                  \
    {                                                                                                                                                                                                                                        \
        _dpdrhoi[nn] = get_RoeAverage(get_DpDrhoi(hi_l[nn], thermal->Ri[nn], hi_l[NUM_COP], thermal->Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),                                                                                              \
                                      get_DpDrhoi(hi_r[nn], thermal->Ri[nn], hi_r[NUM_COP], thermal->Ri[NUM_COP], T[id_r], Cp_r, R_r, gamma_r), D, D1);                                                                                      \
        drhoi[nn] = rho[id_r] * yi_r[nn] - rho[id_l] * yi_l[nn];                                                                                                                                                                             \
    }                                                                                                                                                                                                                                        \
    real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1) + _DF(0.5) * D * D1 * D1 * (sycl::pow<real_t>(u[id_r] - u[id_l], 2) + sycl::pow<real_t>(v[id_r] - v[id_l], 2) + sycl::pow<real_t>(w[id_r] - w[id_l], 2)); \
    real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);                                                                                                                                                            \
    real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);                                                                                                                                \
    real_t c2 = SoundSpeedMultiSpecies(z, b1, b3, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);

/**
 * get c2 #else COP
 */
#define MARCO_NOCOPC2()                                                                                            \
    real_t yi_l[NUM_SPECIES] = {_DF(1.0)}, yi_r[NUM_SPECIES] = {_DF(1.0)}, _yi[] = {1}, b3 = _DF(0.0), z[] = {0};  \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                         \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                         \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                       \
    real_t c2 = Gamma0 * _P / _rho; /*(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x */ \
    real_t b1 = (Gamma0 - _DF(1.0)) / c2;

/**
 * Caculate flux_wall
 */
// WENO 7 // used by MARCO_FLUXWALL_WENO7(i + m, j, k, i + m - stencil_P, j, k); in x
#define MARCO_FLUXWALL_WENO7(_i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                                                             \
    real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux;                                                                                           \
    for (int n = 0; n < Emax; n++)                                                                                                                           \
    {                                                                                                                                                        \
        real_t eigen_local_max = _DF(0.0);                                                                                                                   \
        eigen_local_max = eigen_value[n];                                                                                                                    \
        real_t lambda_l = eigen_local[Emax * id_l + n];                                                                                                      \
        real_t lambda_r = eigen_local[Emax * id_r + n];                                                                                                      \
        if (lambda_l * lambda_r < 0.0)                                                                                                                       \
        {                                                                                                                                                    \
            for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                                      \
            {                                                                                                                                                \
                int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /*Xmax * Ymax * k + Xmax * j + i + m*/ \
                eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/              \
            }                                                                                                                                                \
        }                                                                                                                                                    \
        for (size_t m = 0; m < stencil_size; m++)                                                                                                            \
        {                                                                                                                                                    \
            int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /*Xmax * Ymax * k + Xmax * j + m + i - stencil_P*/                               \
            uf[m] = _DF(0.0);                                                                                                                                \
            ff[m] = _DF(0.0);                                                                                                                                \
            for (int n1 = 0; n1 < Emax; n1++)                                                                                                                \
            {                                                                                                                                                \
                uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_l[n][n1];                                                                                 \
                ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_l[n][n1];                                                                                 \
            }                                                                                                                                                \
            /* for local speed*/                                                                                                                             \
            pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);                                                                                            \
            mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);                                                                                            \
        }                                                                                                                                                    \
        /* calculate the scalar numerical flux at x direction*/                                                                                              \
        f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl);                                                                                  \
        /* get Fp*/                                                                                                                                          \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                    \
        {                                                                                                                                                    \
            _p[n][n1] = f_flux * eigen_r[n1][n];                                                                                                             \
        }                                                                                                                                                    \
    }                                                                                                                                                        \
    /* reconstruction the F-flux terms*/                                                                                                                     \
    for (int n = 0; n < Emax; n++)                                                                                                                           \
    {                                                                                                                                                        \
        real_t temp_flux = _DF(0.0);                                                                                                                         \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                    \
        {                                                                                                                                                    \
            temp_flux += _p[n1][n];                                                                                                                          \
        }                                                                                                                                                    \
        Fwall[Emax * id_l + n] = temp_flux;                                                                                                                  \
    }

// WENO 5 //used by: MARCO_FLUXWALL_WENO5(i + m, j, k, i + m, j, k);
#define MARCO_FLUXWALL_WENO5(_i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                                                                                                      \
    real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax];                                                                                                                                    \
    for (int n = 0; n < Emax; n++)                                                                                                                                                                    \
    {                                                                                                                                                                                                 \
        real_t eigen_local_max = _DF(0.0);                                                                                                                                                            \
        for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                                                                                   \
        {                                                                                                                                                                                             \
            int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /*Xmax * Ymax * k + Xmax * j + i + m*/                                              \
            eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/                                                           \
        }                                                                                                                                                                                             \
        for (int m = -3; m <= 4; m++)                                                                                                                                                                 \
        {                                                                 /* 3rd oder and can be modified*/                                                                                           \
            int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /*Xmax * Ymax * k + Xmax * j + m + i;*/                                                                                     \
            uf[m + 3] = _DF(0.0);                                                                                                                                                                     \
            ff[m + 3] = _DF(0.0);                                                                                                                                                                     \
            for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                         \
            {                                                                                                                                                                                         \
                uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];                                                                                                                    \
                ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_l[n][n1];                                                                                                                    \
            } /*  for local speed*/                                                                                                                                                                   \
            pp[m + 3] = _DF(0.5) * (ff[m + 3] + eigen_local_max * uf[m + 3]);                                                                                                                         \
            mm[m + 3] = _DF(0.5) * (ff[m + 3] - eigen_local_max * uf[m + 3]);                                                                                                                         \
        }                                                                                                                                     /* calculate the scalar numerical flux at x direction*/ \
        f_flux = (weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)); /* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/ /* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/  \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                             \
        { /* get Fp*/                                                                                                                                                                                 \
            _p[n][n1] = f_flux * eigen_r[n1][n];                                                                                                                                                      \
        }                                                                                                                                                                                             \
    } /* reconstruction the F-flux terms*/                                                                                                                                                            \
    for (int n = 0; n < Emax; n++)                                                                                                                                                                    \
    {                                                                                                                                                                                                 \
        Fwall[Emax * id_l + n] = _DF(0.0);                                                                                                                                                            \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                             \
        {                                                                                                                                                                                             \
            Fwall[Emax * id_l + n] += _p[n1][n];                                                                                                                                                      \
        }                                                                                                                                                                                             \
    }

/**
 * prepare for getting viscous flux
 */
#define MARCO_PREVISCFLUX()                                                                                                                               \
    real_t F_wall_v[Emax], f_x, f_y, f_z, u_hlf, v_hlf, w_hlf;                                                                                            \
    real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0); /*mue at wall*/ \
    real_t lamada = -_DF(2.0) / _DF(3.0) * mue;

/**
 * get viscous flux
 */
#ifdef COP
const bool _COP = true;
#else
const bool _COP = false;
#endif
#ifdef Heat
const bool _Heat = true;
#else
const bool _Heat = false;
#endif
#ifdef Diffu
const bool _Diffu = true;
#else
const bool _Diffu = false;
#endif
#define MARCO_VISCFLUX()                                                                                                                                                                               \
    F_wall_v[0] = _DF(0.0);                                                                                                                                                                            \
    F_wall_v[1] = f_x;                                                                                                                                                                                 \
    F_wall_v[2] = f_y;                                                                                                                                                                                 \
    F_wall_v[3] = f_z;                                                                                                                                                                                 \
    F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;                                                                                                                                             \
    if (_Heat) /* Fourier thermal conductivity*/                                                                                                                                                       \
    {                                                                                                                                                                                                  \
        real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0); /* thermal conductivity at wall*/ \
        kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dl / _DF(24.0);                                                                             /* temperature gradient at wall*/ \
        F_wall_v[4] += kk;                                                                                                                                                                             \
    }                                                                                                                                                                                                  \
    if (_Diffu) /* energy fiffusion depends on mass diffusion*/                                                                                                                                        \
    {                                                                                                                                                                                                  \
        real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);                                                                                                 \
        real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];                                                                                               \
        for (int l = 0; l < NUM_SPECIES; l++)                                                                                                                                                          \
        {                                                                                                                                                                                              \
            hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);                              \
            Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);     \
            if (_COP)                                                                                                                                                                                  \
            {                                                                                                                                                                                          \
                Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);                                                                                      \
                Yil_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dl / _DF(24.0); /* temperature gradient at wall*/                                             \
            }                                                                                                                                                                                          \
            else                                                                                                                                                                                       \
            {                                                                                                                                                                                          \
                Yil_wall[l] = _DF(0.0);                                                                                                                                                                \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
        if (_Heat)                                                                                                                                                                                     \
            for (int l = 0; l < NUM_SPECIES; l++)                                                                                                                                                      \
            {                                                                                                                                                                                          \
                F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];                                                                                                                      \
            }                                                                                                                                                                                          \
        if (_COP) /* visc flux for cop equations*/                                                                                                                                                     \
        {                                                                                                                                                                                              \
            real_t CorrectTerm = _DF(0.0);                                                                                                                                                             \
            for (int l = 0; l < NUM_SPECIES; l++)                                                                                                                                                      \
            {                                                                                                                                                                                          \
                CorrectTerm += Dim_wall[l] * Yil_wall[l];                                                                                                                                              \
            }                                                                                                                                                                                          \
            CorrectTerm *= rho_wall;                                                                                                                                                                   \
            for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                                                                                                                     \
            {                                                                                                                                                                                          \
                F_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yil_wall[p - 5] - Yi_wall[p - 5] * CorrectTerm;                                                                                             \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
    }                                                                                                                                                                                                  \
    for (size_t n = 0; n < Emax; n++) /* add viscous flux to fluxwall*/                                                                                                                                \
    {                                                                                                                                                                                                  \
        Flux_wall[n + Emax * id] -= F_wall_v[n];                                                                                                                                                       \
    }

/**
 * Pre get eigen_martix
 */
#define MARCO_PREEIGEN()                      \
    real_t q2 = _u * _u + _v * _v + _w * _w;  \
    real_t _c = sqrt(c2);                     \
    real_t b2 = _DF(1.0) + b1 * q2 - b1 * _H; \
    real_t _c1 = _DF(1.0) / _c;
// =======================================================
// end repeated code definitions
// =======================================================