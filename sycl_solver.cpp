#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <math.h>

#include "setup.h"
#include "global_class.h"

SYCLSolver::SYCLSolver(sycl::queue &q)
{
	// Print device name and version
    std::cout << "Device: "<< q.get_device().get_info<sycl::info::device::name>() 
						<< ",  version = "<<q.get_device().get_info<sycl::info::device::version>() << "\n";
	
	workgroup_size = {dim_block_x, dim_block_y, dim_block_z};
	workitem_size = {X_inner, Y_inner, Z_inner};

	// --------------------------------------------------------------------------------------
	// Geo setup, BC
	// --------------------------------------------------------------------------------------
	int Nmax = std::max(BLOCK_ratio[0],BLOCK_ratio[1]);
	Nmax = std::max(Nmax,BLOCK_ratio[2]);
    //Domain sizes
    domain_length	=	DOMAIN_length*BLOCK_ratio[0]/Nmax;
    domain_width	=	DOMAIN_length*BLOCK_ratio[1]/Nmax;
    domain_height	=	DOMAIN_length*BLOCK_ratio[2]/Nmax;

	dx = DIM_X ? domain_length/(Real)(X_inner) : 0;
	dy = DIM_Y ? domain_width/(Real)(Y_inner) : 0;
	dz = DIM_Z ? domain_height/(Real)(Z_inner) : 0;

	dl = std::max(std::max(domain_length, domain_width), domain_height);

	#if DIM_X
	dl = std::min(dl, dx);
	#endif
	#if DIM_Y
	dl = std::min(dl, dy);
	#endif
	#if DIM_Z
	dl = std::min(dl, dz);
	#endif
    dt = 0;

    //Outer domain boundary conditions
    for (int n=0; n<6; n++)	BCs[n] = Boundarys[n];
	//display domain sizes
	cout<<"\n\n Domain sizes are: "<<domain_length<<"*"<<domain_width<<"*"<<domain_height<<"\n";

	//display the information of fluid materials
	for(int n=0; n<NumFluid; n++){
		fluids[n] = new FluidSYCL(dx, dy, dz, dl, dt,workitem_size,workgroup_size);
		fluids[n]->initialize(n);
	}

	// levelset = new LevelsetGPU(dx,dy,dz,dl, dim_blk, dim_grid);
}

void SYCLSolver::Evolution(sycl::queue &q)
{
	Real physicalTime = 0.0;
	int Iteration = 0;

	double duration = 0.0;
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	// RK3
	while(physicalTime < EndTime) {

		//get minmum dt
		dt = 0.001;//ComputeTimeStep(q);//5.0e-5;//

		if(physicalTime + dt > EndTime) dt = EndTime - physicalTime;
		
    	//solved the fluid with 3rd order Runge-Kutta method
		SinglePhaseSolverRK3rd(q);

		physicalTime = physicalTime + dt;
		Iteration++;

		//screen output
		// if(Iteration%10 == 0)
	        cout<<"N="<<std::setw(6)<<Iteration<<" physicalTime: "<<std::setw(10)<<std::setprecision(8)<<physicalTime<<"	dt: "<<dt<<"\n";

		if(Iteration%100 == 0){
			CopyDataFromDevice(q);
			Output(physicalTime);
		}
	}

	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<float, std::milli>(end_time - start_time).count();
	printf("GPU runtime : %12.8f s\n", duration/1000.0f);
	
    Output(physicalTime);
}

void SYCLSolver::SinglePhaseSolverRK3rd(sycl::queue &q)
{
	RungeKuttaSP3rd(q, 1);
    RungeKuttaSP3rd(q, 2);
    RungeKuttaSP3rd(q, 3);
}

void SYCLSolver::RungeKuttaSP3rd(sycl::queue &q, int flag)
{
	switch(flag) {
        case 1:
            //the fisrt step
            BoundaryCondition(q, 0);
            UpdateStates(q, 0);
            ComputeLU(q, 0);
            UpdateU(q, 1);
        break;
        case 2:
            //the second step
            BoundaryCondition(q, 1);
            UpdateStates(q, 1);
            ComputeLU(q, 1);
            UpdateU(q, 2);
        break;
        case 3:
            //the third step
            BoundaryCondition(q, 1);
            UpdateStates(q, 1);
            ComputeLU(q, 1);
            UpdateU(q, 3);
        break;
    }
}

Real SYCLSolver::ComputeTimeStep(sycl::queue &q)
{
	Real dt_ref = 10e10;
	#if NumFluid == 1
	dt_ref = fluids[0]->GetFluidDt(q);
	#elif NumFluid == 2
	// dt_ref = fluids[0]->GetFluidDt(levelset);
	// dt_ref = min(dt_ref, fluids[1]->GetFluidDt(levelset));
	#endif

	return dt_ref;
}

void SYCLSolver::ComputeLU(sycl::queue &q, int flag)
{
	fluids[0]->ComputeFluidLU(q, flag);
}

void SYCLSolver::UpdateU(sycl::queue &q,int flag)
{
	for(int n=0; n<NumFluid; n++)
		fluids[n]->UpdateFluidURK3(q, flag, dt);
}

void SYCLSolver::BoundaryCondition(sycl::queue &q, int flag)
{
	for(int n=0; n<NumFluid; n++)
		fluids[n]->BoundaryCondition(q, BCs, flag);
}

void SYCLSolver::UpdateStates(sycl::queue &q, int flag)
{
	for(int n=0; n<NumFluid; n++)
		fluids[n]->UpdateFluidStates(q, flag);
}

void SYCLSolver::AllocateMemory(sycl::queue &q)
{
	d_BCs = static_cast<BConditions *>(malloc_device(6*sizeof(BConditions), q));

	q.memcpy(d_BCs, BCs, 6*sizeof(BConditions)).wait();

	//host arrays for each fluid
	for(int n=0; n<NumFluid; n++)
		fluids[n]->AllocateFluidMemory(q);
	
	// levelset->AllocateLSMemory();
}

void SYCLSolver::InitialCondition(sycl::queue &q)
{
	// #if NumFluid == 2
	// levelset->InitPhi();
	// #endif

	for(int n=0; n<NumFluid; n++)
		fluids[n]->InitialU(q, dx, dy, dz);
}

void SYCLSolver::CopyDataFromDevice(sycl::queue &q)
{
	// copy mem from device to host
	for(int n=0; n<NumFluid; n++){
		q.memcpy(fluids[n]->h_fstate.rho, fluids[n]->d_fstate.rho, bytes);
		q.memcpy(fluids[n]->h_fstate.p, fluids[n]->d_fstate.p, bytes);
		q.memcpy(fluids[n]->h_fstate.c, fluids[n]->d_fstate.c, bytes);
		q.memcpy(fluids[n]->h_fstate.H, fluids[n]->d_fstate.H, bytes);
		q.memcpy(fluids[n]->h_fstate.u, fluids[n]->d_fstate.u, bytes);
		q.memcpy(fluids[n]->h_fstate.v, fluids[n]->d_fstate.v, bytes);
		q.memcpy(fluids[n]->h_fstate.w, fluids[n]->d_fstate.w, bytes);
	}
	q.wait();
}

void SYCLSolver::Output(Real Time)
{
	Real Itime;
	char file_name[50], file_list[50];
	
	//produce output file name
	Itime = Time*1.0e6;
	strcpy(file_name,"./outdata/flowfield_");
	sprintf(file_list, "%d", (int)Itime);
	strcat(file_name, file_list);
	strcat(file_name, ".plt");

	ofstream out(file_name);
	//defining header for tecplot(plot software)
	out<<"title='View'"<<"\n";
	int LEN = 0;
	#if (DIM_X+DIM_Y+DIM_Z==1)
	LEN = 2;
	out<<"variables=x, u, p, rho"<<"\n";
	#elif (DIM_X+DIM_Y+DIM_Z==2)
	LEN = 2;
	out<<"variables=x, y, u, v, p, rho"<<"\n";
	#elif (DIM_X+DIM_Y+DIM_Z==3)
	LEN = 2;
	out<<"variables=x, y, z, u, v, w, p, rho"<<"\n";
	#endif
	out<<"zone t='filed', i="<<X_inner+DIM_X<<", j="<<Y_inner+DIM_Y<<", k="<<Z_inner+DIM_Z<<"  DATAPACKING=BLOCK, VARLOCATION=([";
	int pos_s = DIM_X+DIM_Y+DIM_Z+1;
	out<<pos_s<<"-";
	out<<2*pos_s -1 + LEN-1<<"]=CELLCENTERED) SOLUTIONTIME="<<Time<<"\n";

	int ii = Xmax-Bwidth_X + DIM_X - 1;
	int jj = Ymax-Bwidth_Y + DIM_Y - 1;
	int kk = Zmax-Bwidth_Z + DIM_Z - 1;

	#if DIM_X
	for(int k=Bwidth_Z; k<=kk; k++){
		for(int j=Bwidth_Y; j<=jj; j++){
			for(int i=Bwidth_X; i<=ii; i++)
			{
				out<<(i-Bwidth_X)*dx<<" ";
			}
			out<<"\n";
		}
	}
	#endif
	#if DIM_Y
	for(int k=Bwidth_Z; k<=kk; k++){
		for(int j=Bwidth_Y; j<=jj; j++){
			for(int i=Bwidth_X; i<=ii; i++)
			{
				out<<(j-Bwidth_Y)*dy<<" ";
			}
			out<<"\n";
		}
	}
	#endif
	#if DIM_Z
	for(int k=Bwidth_Z; k<=kk; k++){
		for(int j=Bwidth_Y; j<=jj; j++){
			for(int i=Bwidth_X; i<=ii; i++)
			{
				out<<(k-Bwidth_Z)*dz<<" ";
			}
			out<<"\n";
		}
	}
	#endif

	#if DIM_X
	//u
	for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
		for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
			for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
			{
				int id = Xmax*Ymax*k + Xmax*j + i;
				// if(levelset->h_phi[id]>= 0.0)
					out<<fluids[0]->h_fstate.u[id]<<" ";
				// else
				// 	out<<fluids[1]->h_fstate.u[id]<<" ";
			}
			out<<"\n";
		}
	}
	#endif

	#if DIM_Y
	//v
	for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
		for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
			for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
			{
				int id = Xmax*Ymax*k + Xmax*j + i;
				// if(levelset->h_phi[id]>= 0.0)
					out<<fluids[0]->h_fstate.v[id]<<" ";
				// else
				// 	out<<fluids[1]->h_fstate.v[id]<<" ";
			}
			out<<"\n";
		}
	}
	#endif

	#if DIM_Z
	//w
	for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
		for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
			for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
			{
				int id = Xmax*Ymax*k + Xmax*j + i;
				// if(levelset->h_phi[id]>= 0.0)
					out<<fluids[0]->h_fstate.w[id]<<" ";
				// else
				// 	out<<fluids[1]->h_fstate.w[id]<<" ";
			}
			out<<"\n";
		}
	}
	#endif

	//P
	for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
		for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
			for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
			{
				int id = Xmax*Ymax*k + Xmax*j + i;
				// if(levelset->h_phi[id]>= 0.0)
					out<<fluids[0]->h_fstate.p[id]<<" ";
				// else
				// 	out<<fluids[1]->h_fstate.p[id]<<" ";
			}
			out<<"\n";
		}
	}

	// rho
	for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
		for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
			for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
			{
				int id = Xmax*Ymax*k + Xmax*j + i;
				// if(levelset->h_phi[id]>= 0.0)
					out<<fluids[0]->h_fstate.rho[id]<<" ";
				// else
				// 	out<<fluids[1]->h_fstate.rho[id]<<" ";
			}
			out<<"\n";
		}
	}

	// //phi
	// for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
	// 	for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
	// 		for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
	// 		{
	// 			int id = Xmax*Ymax*k + Xmax*j + i;
	// 			out<<levelset->h_phi[id]<<" ";
	// 		}
	// 		out<<"\n";
	// 	}
	// }

	out.close();
}