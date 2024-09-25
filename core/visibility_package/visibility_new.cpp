// -------------------------------------------------------------------
// Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
// Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
// Released under Creative Commons
// Attribution-NonCommercial-ShareAlike 4.0 International License.
// http://creativecommons.org/licenses/by-nc-sa/4.0/
// -------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<at::Tensor> depth_image_cuda(at::Tensor input_uv, at::Tensor input_depth, at::Tensor input_index, at::Tensor output, at::Tensor output_index, unsigned int size, unsigned int width,unsigned int height);
at::Tensor visibility_filter_cuda(at::Tensor input, at::Tensor output, unsigned int width, unsigned int height, unsigned int threshold);
std::vector<at::Tensor> visibility_filter_cuda2(at::Tensor input_depth, at::Tensor intrinsic, at::Tensor in_index, at::Tensor output, at::Tensor out_index, unsigned int width,unsigned int height, float threshold, unsigned int radius);
at::Tensor downsample_flow_cuda(at::Tensor input_uv, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel);
at::Tensor downsample_mask_cuda(at::Tensor input_mask, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel);
//at::Tensor visibility_filter_cuda_shared(at::Tensor input_depth, at::Tensor input_points, at::Tensor output, unsigned int width,unsigned int height, float threshold);
at::Tensor image_warp_index_cuda(at::Tensor input, at::Tensor flow, at::Tensor warp_depth, at::Tensor output, unsigned int width, unsigned int height);
at::Tensor image_warp_cuda(at::Tensor input, at::Tensor flow, at::Tensor output, unsigned int width, unsigned int height);
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> depth_image(at::Tensor input_uv, at::Tensor input_depth,at::Tensor input_index, at::Tensor output, at::Tensor output_index, unsigned int size, unsigned int width,unsigned int height) {
    CHECK_INPUT(input_uv);
    CHECK_INPUT(input_depth);
    CHECK_INPUT(input_index);
    CHECK_INPUT(output);
    CHECK_INPUT(output_index);
    return depth_image_cuda(input_uv, input_depth, input_index, output,output_index, size, width, height);
}

at::Tensor visibility_filter(at::Tensor input, at::Tensor output, unsigned int width, unsigned int height, unsigned int threshold){
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    return visibility_filter_cuda(input, output, width, height, threshold);
}

std::vector<at::Tensor> visibility_filter2(at::Tensor input_depth, at::Tensor intrinsic, at::Tensor in_index, at::Tensor output, at::Tensor out_index, unsigned int width, unsigned int height, float threshold, unsigned int radius){
    CHECK_INPUT(input_depth);
    CHECK_INPUT(intrinsic);
    CHECK_INPUT(output);
    return visibility_filter_cuda2(input_depth, intrinsic, in_index, output, out_index, width, height, threshold, radius);
}

at::Tensor downsample_flow(at::Tensor input_uv, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    CHECK_INPUT(input_uv);
    CHECK_INPUT(output);
    return downsample_flow_cuda(input_uv, output, width_out, height_out, kernel);
}

at::Tensor downsample_mask(at::Tensor input_mask, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    CHECK_INPUT(input_mask);
    CHECK_INPUT(output);
    return downsample_mask_cuda(input_mask, output, width_out, height_out, kernel);
}

at::Tensor image_warp_index(at::Tensor input, at::Tensor flow, at::Tensor warp_depth, at::Tensor output, unsigned int width, unsigned int height)
{
    CHECK_INPUT(input);
    CHECK_INPUT(flow);
    CHECK_INPUT(warp_depth);
    CHECK_INPUT(output);
    return image_warp_index_cuda(input, flow, warp_depth, output, width, height);
}

at::Tensor image_warp(at::Tensor input, at::Tensor flow, at::Tensor output, unsigned int width, unsigned int height)
{
    CHECK_INPUT(input);
    CHECK_INPUT(flow);
    CHECK_INPUT(output);
    return image_warp_cuda(input, flow, output, width, height);
}

/*at::Tensor visibility_filter_shared(at::Tensor input_depth, at::Tensor input_points, at::Tensor output, unsigned int width, unsigned int height, float threshold){
    CHECK_INPUT(input_depth);
    CHECK_INPUT(input_points);
    CHECK_INPUT(output);
    return visibility_filter_cuda_shared(input_depth, input_points, output, width, height, threshold);
}*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_image", &depth_image, "Generate Depth Image(CUDA)");
  m.def("visibility", &visibility_filter, "visibility_filter (CUDA)");
  m.def("visibility2", &visibility_filter2, "visibility_filter2 (CUDA)");
  m.def("downsample_flow", &downsample_flow, "downsample_flow (CUDA)");
  m.def("downsample_mask", &downsample_mask, "downsample_mask (CUDA)");
  m.def("image_warp_index", &image_warp_index, "image_warp_index (CUDA)");
  m.def("image_warp", &image_warp, "image_warp (CUDA)");
  //m.def("visibility_shared", &visibility_filter2, "visibility_filter_shared (CUDA)");
}
