math(EXPR DIM_NUM "0" OUTPUT_FORMAT DECIMAL)

IF(DIM_X)
    math(EXPR DIM_NUM "${DIM_NUM} + 1" OUTPUT_FORMAT DECIMAL)
ENDIF(DIM_X)

IF(DIM_Y)
    math(EXPR DIM_NUM "${DIM_NUM} + 1" OUTPUT_FORMAT DECIMAL)
ENDIF(DIM_Y)

IF(DIM_Z)
    math(EXPR DIM_NUM "${DIM_NUM} + 1" OUTPUT_FORMAT DECIMAL)
ENDIF(DIM_Z)

IF(USE_MPI)
    set(APPEND "-mpi.ini")
ELSE()
    set(APPEND ".ini")
ENDIF()

set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/${COP_SPECIES}") # Be invalid while option COP_CHEME "ON"

# // =======================================================
# #### util sample
# // =======================================================
IF(INIT_SAMPLE STREQUAL "for-debug")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/for-debug")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-debug${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "guass-wave")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/guass-wave")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-guass-wave${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "sharp-interface")
    set(OUT_PLT "ON")
    set(DIM_X "ON")
    set(COP_CHEME "OFF")
    set(WENO_ORDER "6") # WENOCU6 has the larggest unrubost at Riemann separation
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/sharp-interface")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-1d-shock-tube${APPEND}")
    set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/Insert-SBI")

# // =======================================================
# #### 1d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "1d-insert-st")
    IF(DIM_NUM STREQUAL "1")
        set(OUT_PLT "ON")
        set(OUT_VTI "OFF")
        set(Visc "OFF")
        set(COP_CHEME "OFF")
        set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
        set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-1d-shock-tube${APPEND}")
        set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/1d-mc-insert-shock-tube")
    ELSEIF()
        message(FATAL_ERROR: "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-reactive-st")
    IF(DIM_NUM STREQUAL "1")
        set(OUT_PLT "ON")
        set(OUT_VTI "OFF")
        set(Visc "OFF")
        set(COP_CHEME "ON")
        set(COP_CHEME_TEST "ON")
        set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/reactive-st")
        set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-1d-reactive-st${APPEND}")
    ELSEIF()
        message(FATAL_ERROR: "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion")
    IF(DIM_NUM STREQUAL "1")
        set(OUT_PLT "ON")
        set(OUT_VTI "OFF")
        set(Visc "ON")
        set(Heat "ON")
        set(Diffu "ON")
        set(COP_CHEME "OFF")
        set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/diffusion")
        set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-1d-diffusion${APPEND}")
        set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/1d-mc-diffusion")
    ELSEIF()
        message(FATAL_ERROR: "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion-reverse")
    IF(DIM_NUM STREQUAL "1")
        set(OUT_PLT "ON")
        set(OUT_VTI "OFF")
        set(Visc "ON")
        set(Heat "ON")
        set(Diffu "ON")
        set(COP_CHEME "OFF")
        add_compile_options(-DDiffuReverse)
        set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/diffusion")
        set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-1d-diffusion${APPEND}")
        set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/1d-mc-diffusion-reverse")
    ELSEIF()
        message(FATAL_ERROR: "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

# // =======================================================
# #### 2d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "2d-shock-bubble")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(THERMAL "NASA") # NASA fit of Xe
    add_compile_options(-DSBICounts)
    message(STATUS "  Only NASA fit for Xe used in RSBI sample.")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/shock-bubble-intera")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-shock-bubble${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "2d-under-expanded-jet")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/under-expanded-jet")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-expanded-jet${APPEND}")

# // =======================================================
# #### 3d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "3d-shock-bubble")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(THERMAL "NASA") # NASA fit of Xe
    add_compile_options(-DSBICounts)
    message(STATUS "  Only NASA fit for Xe used in RSBI sample.")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/shock-bubble-intera")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-shock-bubble${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "3d-under-expanded-jet")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/under-expanded-jet")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-expanded-jet${APPEND}")
ELSE()
    message(FATAL_ERROR "ini sample isn't given.")
ENDIF()

IF(COP)
    add_compile_options(-DCOP)

    IF(COP_CHEME)
        add_compile_options(-DCOP_CHEME)
        set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/Reaction/${REACTION_MODEL}") # where to read species including reactions

        IF(COP_CHEME_TEST)
            add_compile_options(-DODESolverTest)
        ENDIF()

        IF(${CHEME_SOLVER} MATCHES "Q2")
            add_compile_options(-DCHEME_SOLVER=0)
        ELSEIF(${CHEME_SOLVER} MATCHES "CVODE")
            add_compile_options(-DCHEME_SOLVER=1)
        ELSE()
        ENDIF()
    ENDIF(COP_CHEME)
ELSE(COP)
    set(Gamma "1.4")
    add_compile_options(-DNUM_SPECIES=1)
    add_compile_options(-DNCOP_Gamma=${Gamma})
    add_compile_options(-DNUM_REA=0)
    set(INIT_SAMPLE "not-compoent")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/settings/sa-not-compoent.ini")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/not-compoent")
    set(COP_SPECIES "Only one species actived and set Gamma to ${Gamma}") # Be invalid while option COP_CHEME "ON"
ENDIF(COP)

add_compile_options(-DINI_SAMPLE="${INIT_SAMPLE}")
message(STATUS "Util features: ")
message(STATUS "  Running platform: ${SelectDv}")
message(STATUS "  Double precision: ${USE_DOUBLE}")
message(STATUS "  Output: VTI: ${OUT_VTI}, PLT: ${OUT_PLT}")
message(STATUS "  DIM_X: ${DIM_X}, DIM_Y: ${DIM_Y}, DIM_Z: ${DIM_Z}")
message(STATUS "  Positivity Preserving: ${POSITIVITY_PRESERVING}")
message(STATUS "  Capture unexpected errors: ${ESTIM_NAN}")
message(STATUS "  Fix negative mass fraction(Yi): ${ERROR_PATCH_YI}")
message(STATUS "    NOTE: may not damage the equations but decrease the accuracy of solution")
message(STATUS "  Fix nan primitive variables(rho,p,T): ${ERROR_PATCH_PRI}")
message(STATUS "Solvers' settings: ")
message(STATUS "  Multi-Component: ${COP}")
message(STATUS "  Species' Thermo Fit: ${THERMAL}")
message(STATUS "  Convention term scheme: WENO${WENO_ORDER}")
message(STATUS "  Viscous flux term: ${Visc}")

IF(Visc)
    message(STATUS "  Viscous Heat term: ${Heat}")
    message(STATUS "  Viscous Diffusion: ${Diffu}")
ENDIF()

message(STATUS "  Reaction term: ${COP_CHEME}")

IF(COP_CHEME)
    message(STATUS "  Reaction term Solver: ${CHEME_SOLVER}")
    message(STATUS "  Reacting ZeroDimensional TEST: ${COP_CHEME_TEST}")
ENDIF()

message(STATUS "Sample init include settings: ")
message(STATUS "  Sample select: ${INIT_SAMPLE}")
message(STATUS "  Sample init sample path: ${INI_SAMPLE_PATH}")
message(STATUS "  Sample COP  header path: ${COP_SPECIES}")
message(STATUS "  Sample ini  file   path: ${INI_FILE}")

set(COP_THERMAL_PATH "${CMAKE_SOURCE_DIR}/runtime.dat") # where to read .dat about charactersics of compoent gas
add_compile_options(-DIniFile="${INI_FILE}")
add_compile_options(-DRFile="${COP_SPECIES}")
add_compile_options(-DRPath="${COP_THERMAL_PATH}")

include_directories(
    BEFORE
    "${COP_SPECIES}"
    "${INI_SAMPLE_PATH}"
)

# ${COP_SPECIES}: 依据算例文件中的"case_setup.h"头文件自动设置NUM_SPECIES && NUM_REACTIONS #
# ${INI_SAMPLE_PATH}: 依据sample文件夹中的"ini_sample.hpp"文件选择 #
