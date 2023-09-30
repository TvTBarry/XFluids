#pragma once

#include "Utils_kernels.hpp"

extern SYCL_EXTERNAL void GetLocalEigen(int i, int j, int k, Block bl, real_t AA, real_t BB, real_t CC, real_t *eigen_local, real_t *u, real_t *v, real_t *w, real_t *c)
{
	MARCO_DOMAIN();
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
#if SCHEME_ORDER <= 6
	real_t uu = AA * u[id] + BB * v[id] + CC * w[id];
	real_t uuPc = uu + c[id];
	real_t uuMc = uu - c[id];

	// local eigen values
	eigen_local[Emax * id + 0] = uuMc;
	for (size_t ii = 1; ii < Emax - 1; ii++)
	{
		eigen_local[Emax * id + ii] = uu;
	}
	eigen_local[Emax * id + Emax - 1] = uuPc;
#elif SCHEME_ORDER == 7
	for (size_t ii = 0; ii < Emax; ii++)
		eigen_local[Emax * id + ii] = _DF(0.0);
#endif // end FLUX_method

	// real_t de_fw[Emax];
	// get_Array(eigen_local, de_fw, Emax, id);
	// real_t de_fx[Emax];
}