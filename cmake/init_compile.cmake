IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(SelectDv "host") # define which platform and devices for compile options: host, nvidia, amd, intel
ENDIF()

# // =======================================================
IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
	# // =======================================================
	set(AdaptiveCpp_DIR "/home/ljl/Apps/OpenSYCL/lib/cmake/AdaptiveCpp")
	find_package(AdaptiveCpp CONFIG REQUIRED)

	IF(SelectDv STREQUAL "cuda-nvcxx")
		set(ARCH "cc${ARCH}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=set_but_not_used,declared_but_not_referenced,used_before_set,code_is_unreachable,unsigned_compare_with_zero")
	ELSEIF(SelectDv STREQUAL "cuda")
		set(ARCH "sm_${ARCH}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-cuda-version -Wno-format")
	ELSEIF(SelectDv STREQUAL "hip")
		set(ARCH "gfx${ARCH}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
	ELSE()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
	ENDIF()

	add_compile_options(-DDEFINED_OPENSYCL)

	# set(CMAKE_CXX_COMPILER "syclcc") # for OpenSYCL syclcc compiling system
	IF((SelectDv STREQUAL "omp") OR(SelectDv STREQUAL "host"))
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets='omp'") # get samples from syclcc --help
		set(ARCH "host")
	ELSE()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets='${SelectDv}:${ARCH}'") # get samples from syclcc --help
	ENDIF()

# // =======================================================
ELSEIF(SYCL_COMPILE_SYSTEM STREQUAL "oneAPI")
	# // =======================================================
	add_compile_options(-DDEFINED_ONEAPI)
	set(CMAKE_CXX_COMPILER "clang++") # for Intel oneAPI compiling system
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	IF((SelectDv STREQUAL "omp") OR(SelectDv STREQUAL "host"))
		set(SelectDv "host")
		set(ARCH "")
	ELSE()
	ENDIF()

	include(oneAPIdevSelect/init_${SelectDv})
ENDIF()

# // =======================================================
# #### about device select
# // =======================================================
message(STATUS "CMAKE STATUS:")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
add_compile_options(-DSelectDv="${SelectDv}") # device, add marco as string-value
