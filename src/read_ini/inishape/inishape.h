#pragma once

#include "global_setup.h"

typedef struct
{
	bool idocoor;
	size_t xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id;
	real_t xmin_coor, xmax_coor, ymin_coor, ymax_coor, zmin_coor, zmax_coor;
	real_t rho, P, T, u, v, w, *yi;
} IniBox;

typedef struct
{
	// bubble center
	real_t center_x, center_y, center_z;
	// bubble shape
	real_t C, _xa2, _yb2, _zc2;
	// fluid states inside bubble
	real_t rho, P, T, u, v, w, *yi;
} IniBubble;

class IniShape
{
private:
public:
	// cop_type: 0 for 1d set, 1 for bubble of cop
	// blast_type: 0 for 1d shock, 1 for circular shock
	int cop_type, blast_type, bubble_type;
	// blast position and states
	real_t blast_center_x, blast_center_y, blast_center_z, blast_radius,
		blast_density_in, blast_density_out, blast_pressure_in, blast_pressure_out,
		blast_T_in, blast_T_out, blast_u_in, blast_v_in, blast_w_in, blast_u_out, blast_v_out, blast_w_out;
	// bubble position
	real_t cop_center_x, cop_center_y, cop_center_z, cop_radius, cop_density_in, cop_pressure_in, cop_T_in;
	real_t Ma; // shock much number
	// bubble position; NOTE: Domain_length may be the max value of the Domain size
	real_t bubble_center_x, bubble_center_y, bubble_center_z, bubbleSz;
	// bubble shape
	real_t xa, yb, zc, C, _xa2, _yb2, _zc2, _xa2_in, _yb2_in, _zc2_in, _xa2_out, _yb2_out, _zc2_out;

public:
	size_t num_box, num_bubble;
	IniBox *iboxs;
	IniBubble *ibubbles;

	IniShape(){};
	~IniShape(){};
	IniShape(sycl::queue &q, size_t num_box, size_t num_bubble);
};