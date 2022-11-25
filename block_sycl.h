#pragma once
#include "setup.h"
#include "sycl_kernels.h"
#include "fun.h"

// SYCL head files
#include <CL/sycl.hpp>
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

void InitializeFluidStates(sycl::queue &q, array<int, 3> WG, array<int, 3> WI, MaterialProperty *material, FlowData &fdata, 
                            Real* U, Real* U1, Real* LU,
                            Real* FluxF, Real* FluxG, Real* FluxH, 
                            Real* FluxFw, Real* FluxGw, Real* FluxHw, 
                            Real const dx, Real const dy, Real const dz);


void FluidBoundaryCondition(sycl::queue &q, BConditions BCs[6], Real*  d_UI);

void UpdateFluidStateFlux(sycl::queue &q, Real*  UI, FlowData &fdata, Real*  FluxF, Real*  FluxG, Real*  FluxH, Real const Gamma);

void UpdateURK3rd(sycl::queue &q, Real* U, Real* U1, Real* LU, Real const dt, int flag);

void GetLU(sycl::queue &q, Real* UI, Real* LU, Real* FluxF, Real* FluxG, Real* FluxH, 
            Real* FluxFw, Real* FluxGw, Real* FluxHw, Real const Gamma, int const Mtrl_ind, 
            FlowData &fdata, Real* eigen_local, Real const dx, Real const dy, Real const dz);

Real GetDt(sycl::queue &q, FlowData &fdata, Real* uvw_c_max, Real const dx, Real const dy, Real const dz);