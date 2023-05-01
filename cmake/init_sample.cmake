message(STATUS "Sample init include settings: ")

IF(USE_MPI)
    set(APPEND "-mpi.ini")
ELSE()
    set(APPEND ".ini")
ENDIF()

IF(INIT_SAMPLE STREQUAL "for-debug")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/for-debug")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-debug${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "1d-guass-wave")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/guass-wave")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-guss-wave${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "1d-insert-st")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-1d-shock-tube${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "1d-reactive-st")
    set(COP_CHEME "ON")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/reactive-st")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-1d-reactive-st${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "2d-guass-wave")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/2D-XY/guass-wave")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-guss-wave${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "2d-shock-bubble")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/2D-XY/shock-bubble-intera")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-shock-bubble${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "2d-under-expanded-jet")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/2D-XY/under-expanded-jet")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-expanded-jet${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "3d-shock-bubble")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/3D-XYZ/shock-bubble-intera")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-shock-bubble${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "3d-under-expanded-jet")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/3D-XYZ/under-expanded-jet")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-expanded-jet${APPEND}")
ENDIF()

IF(COP)
    add_compile_options(-DCOP)

    IF(COP_CHEME)
        add_compile_options(-DCOP_CHEME)
        set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/Reaction/${REACTION_MODEL}") # where to read species including reactions

        IF(${CHEME_SOLVER} MATCHES "Q2")
            add_compile_options(-DCHEME_SOLVER=0)
        ELSEIF(${CHEME_SOLVER} MATCHES "CVODE")
            add_compile_options(-DCHEME_SOLVER=1)
        ELSE()
        ENDIF()
    ENDIF(COP_CHEME)
ELSE(COP)
    add_compile_options(-DNUM_SPECIES=1)
    add_compile_options(-DNUM_REA=0)
ENDIF(COP)

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