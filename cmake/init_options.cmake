add_compile_options(-DNumFluid=1)
IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)#将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)
IF(Comp_Vis)
  add_compile_options(-DVisCOMP)
ENDIF(Comp_Vis)
IF(DIM_X)
  add_compile_options(-DDIM_X=1)
ELSE(DIM_X)
  add_compile_options(-DDIM_X=0)
ENDIF(DIM_X)
IF(DIM_Y)
  add_compile_options(-DDIM_Y=1)
ELSE(DIM_Y)
  add_compile_options(-DDIM_Y=0)
ENDIF(DIM_Y)
IF(DIM_Z)
  add_compile_options(-DDIM_Z=1)
ELSE(DIM_Z)
  add_compile_options(-DDIM_Z=0)
ENDIF(DIM_Z)
IF(COP)
  add_compile_options(-DCOP)
  IF(React)
    add_compile_options(-DReact)
    add_compile_options(-DReaType=${ReaType})
  ENDIF(React)
ENDIF(COP)
IF(${DeSelect} MATCHES "nvidia")
  include(init_cuda)
ELSEIF(${DeSelect} MATCHES "amd")
  include(init_hip)
ELSEIF(${DeSelect} MATCHES "intel")
  include(init_intel)
ELSEIF (${DeSelect} MATCHES "host")
  include (init_host)
ENDIF(${DeSelect} MATCHES "nvidia")


IF(ENABLE_OPENMP)
  include(init_openmp)
ENDIF(ENABLE_OPENMP)

IF(USE_MPI)
  include(init_mpi)
ENDIF(USE_MPI)