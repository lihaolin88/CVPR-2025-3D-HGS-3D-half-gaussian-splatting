/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

struct float6 {
    float x, y, z, w, u, v;
};

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect_another(const float2 p, int width, int height, int width_small, int height_small, int width_another, int height_another, int width_small_another, int height_small_another, int type, uint2& rect_min, uint2& rect_max, uint2& rect_min_another, uint2& rect_max_another, dim3 grid)
{
    uint2 rect_min_large = {
		min(grid.x, max((int)0, (int)((p.x - width) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - height) / BLOCK_Y)))
	};
	uint2 rect_max_large = {
		min(grid.x, max((int)0, (int)((p.x + width + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + height + BLOCK_Y - 1) / BLOCK_Y)))
	};
	uint2 rect_min_small = {
		min(grid.x, max((int)0, (int)((p.x - width_small) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - height_small) / BLOCK_Y)))
	};
	uint2 rect_max_small = {
		min(grid.x, max((int)0, (int)((p.x + width_small + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + height_small + BLOCK_Y - 1) / BLOCK_Y)))
	};

	////
//	rect_min_small.x = (rect_max_large.x - rect_min_large.x)<3?rect_min_large.x: rect_min_small.x;
//	rect_min_small.y = (rect_max_large.y - rect_min_large.y)<3?rect_min_large.y: rect_min_small.y;
//	rect_max_small.x = (rect_max_large.x - rect_min_large.x)<3?rect_max_large.x: rect_max_small.x;
//	rect_max_small.y = (rect_max_large.x - rect_min_large.x)<3?rect_max_large.y: rect_max_small.y;
	////

	/////////////////////////
	uint2 rect_min_large_another = {
		min(grid.x, max((int)0, (int)((p.x - width_another) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - height_another) / BLOCK_Y)))
	};
	uint2 rect_max_large_another = {
		min(grid.x, max((int)0, (int)((p.x + width_another + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + height_another + BLOCK_Y - 1) / BLOCK_Y)))
	};
	uint2 rect_min_small_another = {
		min(grid.x, max((int)0, (int)((p.x - width_small_another) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - height_small_another) / BLOCK_Y)))
	};
	uint2 rect_max_small_another = {
		min(grid.x, max((int)0, (int)((p.x + width_small_another + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + height_small_another + BLOCK_Y - 1) / BLOCK_Y)))
	};
	/////////////////////////
    //return result based on type
    uint32_t type0 =  static_cast<int>(type == 0);// ? 1 : 0; //determine if type is 0, keep whole gaussian
    uint32_t type1 =  static_cast<int>(type == 1);// ? 1 : 0; //determine if type is 1,
    uint32_t type2 =  static_cast<int>(type == 2);// ? 1 : 0; //determine if type is 2
    uint32_t type3 =  static_cast<int>(type == 3);// ? 1 : 0; //determine if type is 3
    uint32_t type4 =  static_cast<int>(type == 4);// ? 1 : 0; //determine if type is 4

    rect_min = {(type0 + type1 + type3)*rect_min_large.x + (type2 + type4)*rect_min_small.x, (type0 + type1 + type2)*rect_min_large.y + (type3 + type4)*rect_min_small.y};
//    rect_max = {(type0 + type2 + type4)*rect_max_large.x + (type1 + type3)*rect_max_small.x, (type0 + type1 + type2)*rect_max_large.y + (type3 + type4)*rect_max_small.y};
    rect_max = {(type0 + type2 + type4)*rect_max_large.x + (type1 + type3)*rect_max_small.x, (type0 + type3 + type4)*rect_max_large.y + (type1 + type2)*rect_max_small.y};

    ///////////////////////
    rect_min_another = {(type0 + type4 + type2)*rect_min_large_another.x + (type3 + type1)*rect_min_small_another.x, (type0 + type4 + type3)*rect_min_large_another.y + (type2 + type1)*rect_min_small_another.y};
//    rect_max_another = {(type0 + type3 + type1)*rect_max_large_another.x + (type4 + type2)*rect_max_small_another.x, (type0 + type4 + type3)*rect_max_large_another.y + (type2 + type1)*rect_max_small_another.y};
    rect_max_another = {(type0 + type3 + type1)*rect_max_large_another.x + (type4 + type2)*rect_max_small_another.x, (type0 + type2 + type1)*rect_max_large_another.y + (type4 + type3)*rect_max_small_another.y};
    ///////////////////////
//    printf("rec top %d, %d, bottom %d, %d. Another top %d, %d, bottom %d, %d\n", rect_min.x,rect_min.y,rect_max.x,rect_max.y, rect_min_another.x,rect_min_another.y,rect_max_another.x,rect_max_another.y);
//    printf("rec top %d, %d, bottom %d, %d. Another top %d, %d, bottom %d, %d\n", rect_min_large.x,rect_min_large.y,rect_max_large.x,rect_max_large.y, rect_min_small.x,rect_min_small.y,rect_max_small.x,rect_max_small.y);
}

__forceinline__ __device__ void getRect(const float2 p, int width, int height, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - width) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - height) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + width + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + height + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif