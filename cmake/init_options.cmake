add_compile_options(-DNumFluid=1)
IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)#将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)

IF(Visc)
  add_compile_options(-DVisc)
ENDIF(Visc)

IF(Heat)
  add_compile_options(-DHeat)
ENDIF(Heat)

IF(Diffu)
  add_compile_options(-DDiffu)
ENDIF(Diffu)

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

  IF(Diffu)
    add_compile_options(-DDiffu)
  ENDIF(Diffu)
ENDIF(COP)

IF(SelectDv STREQUAL "nvidia")
  include(init_cuda)
ELSEIF(SelectDv STREQUAL "amd")
  include(init_hip)
ELSEIF(SelectDv STREQUAL "intel")
  include(init_intel)
ELSEIF(SelectDv STREQUAL "host")
  include(init_host)
ENDIF(SelectDv STREQUAL "nvidia")

IF(ENABLE_OPENMP)
  include(init_openmp)
ENDIF(ENABLE_OPENMP)

IF(USE_MPI)
  include(init_mpi)
ENDIF(USE_MPI)
