#pragma once

const bool g_mkdir = 1; // set to 1 to enable saving make directory for saving results
const bool g_save = 1;	// set to 1 to enable saving results

const bool g_save_data_group = 1; // save data group

const bool g_save_index = 1;			// img index
const bool g_save_K = 1;				// camera intrinsics
const bool g_save_success = 0;			// success flag
const bool g_save_model_radius = 0;		// model radius in mm
const bool g_save_time = 0;				// time used for tracking
const bool g_save_gt = 1;				// ground truth pose
const bool g_save_pose = 1;				// estimated pose
const bool g_save_filtered_pose = 1;	// filtered pose
const bool g_save_compensated_pose = 0; // compensated pose
const bool g_save_diff = 0;				// difference between estimated and ground truth pose
const bool g_save_roi = 0;				// region of interest
const bool g_save_mask = 0;				// 3D model's projection mask
const bool g_save_depth = 0;			// depth map
const bool g_save_normal = 0;			// normal map
const bool g_save_origin = 0;			// the input origin image
const bool g_save_overlay = 0;			// the overlay image with AR effect
const bool g_save_probability = 0;		// probability map
const bool g_save_valid = 1;
const bool g_save_info = 1;

const bool g_save_prev_corners = 0;
const bool g_save_fine_depth = 0;

// Just for visualization
const bool g_show_overlay = 1;
const bool g_show_probability = 1;

// Garantee that the routation R in the estimated pose is a regular rotation matrix
#define g_regular_rotation_error