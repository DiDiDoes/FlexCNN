import numpy as np
import json
import argparse
import copy
import multiprocessing
import subprocess
import time
import pandas as pd
import math
import os
#from layer_latency_debug import *
#from layer_latency_solver import *
from gekko import GEKKO
import networkx as nx

def list_split(ori_list, split_num):
	chunk_size = int(np.ceil(float(len(ori_list)) / split_num))
	chunks = [ori_list[i: i + min(chunk_size, len(ori_list) - i)] for i in range(0, len(ori_list), chunk_size)]
	return chunks
	
def effective_dram_est(port_width, burst_len, fre):
	dram_latency = 250
	eff_bw = port_width * burst_len / 8 / ((dram_latency + burst_len) / (fre * 1e6)) / 1e9
	eff_port_width = port_width * burst_len / (dram_latency + burst_len)
	return eff_bw, eff_port_width

def cin_load_est(in_num_t, in_h_t, in_w_t, fh, fw, lane, dw, port_width, fre):
	burst_len = (in_w_t + fw - 1) * in_num_t / (port_width / dw)
	eff_bw, eff_port_width = effective_dram_est(port_width, burst_len, fre)
	load_phase_latency = in_num_t * (fh - 1 + in_h_t) * (fw - 1 + in_w_t) / (eff_port_width / dw)
	write_phase_latency = in_num_t * (fh - 1 + in_h_t) * (fw - 1 + in_w_t) / lane
	return max(load_phase_latency, write_phase_latency)

def weight_load_est(in_num_t, out_num_t, fh1, fw1, fh2, fw2, lane, dw1, dw2, dw3, port_width, point_en, fre):
	burst_len2 = in_num_t * out_num_t * fh2 * fw2 / (port_width / dw2)
	eff_bw2, eff_port_width2 = effective_dram_est(port_width, burst_len2, fre)
	burst_len3 = out_num_t / (port_width / dw3)
	eff_bw3, eff_port_width3 = effective_dram_est(port_width, burst_len3, fre)

	load_phase_latency = 0
	write_phase_latency = 0
	if point_en == 1:
		load_phase_latency += in_num_t * out_num_t * fh2 * fw2 / (eff_port_width2 / dw2)
		load_phase_latency += out_num_t / (eff_port_width3 / dw3)

	if point_en == 1:
		write_phase_latency = max(write_phase_latency, in_num_t * out_num_t * fh2 * fw2 / lane)
		write_phase_latency = max(write_phase_latency, out_num_t / lane)

	return load_phase_latency + write_phase_latency

def point_conv_est(in_num, in_num_t, out_num_t, in_h_t, in_w_t, out_h_t, out_w_t, fh1, fw1, fh2, fw2, lane, sa_rows, sa_cols, sa_lane, exp_factor):
	cin_load = in_num_t * (fh1 - 1 + in_h_t) * (fw1 - 1 + in_w_t) / lane
	weight_load = in_num_t * out_num_t * fh2 * fw2 / lane
	load_phase_latency = max(cin_load, weight_load)
	compute_phase_latency = in_num_t * out_num_t * (out_h_t/exp_factor) * (out_w_t/exp_factor) * fh2 * fw2 / sa_rows / sa_cols / sa_lane
	compute_drain_latency = out_num_t * out_w_t / sa_cols * out_h_t / np.ceil(in_num / in_num_t)
	cout_write = out_num_t * out_h_t * out_w_t / np.ceil(in_num / in_num_t) / lane
	write_phase_latency = cout_write
	return max(load_phase_latency, compute_phase_latency, compute_drain_latency, write_phase_latency)

def relu_est(in_num, in_num_t, out_num_t, out_h_t, out_w_t, lane):
	return out_num_t * out_h_t * out_w_t / lane / np.ceil(in_num / in_num_t)

def pool_est(in_num, in_num_t, out_num_t, out_h_t, out_w_t, lane):
	return out_num_t * out_h_t * out_w_t / lane / np.ceil(in_num / in_num_t)


def cout_write_est(in_num, in_num_t, out_num_t, out_h_t, out_w_t, stride, lane, dw, port_width, fre):
	load_phase_latency = out_num_t * out_h_t * out_w_t / lane / np.ceil(in_num / in_num_t)
	burst_len = out_w_t / stride * out_num_t / (port_width / dw)
	eff_bw, eff_port_width = effective_dram_est(port_width, burst_len, fre)
	write_phase_latency = out_num_t * out_h_t / stride * out_w_t / stride / np.ceil(in_num / in_num_t) / (eff_port_width / dw)
	return max(load_phase_latency, write_phase_latency)

def cin_load_est_s(in_num_t, in_h_t, in_w_t, fh, fw, lane, dw, port_width, fre, m):
	burst_len = (in_w_t + fw - 1) * in_num_t / (port_width / dw)
	eff_bw, eff_port_width = effective_dram_est(port_width, burst_len, fre)
	load_phase_latency = in_num_t * (fh - 1 + in_h_t) * (fw - 1 + in_w_t) / (eff_port_width / dw)
	write_phase_latency = in_num_t * (fh - 1 + in_h_t) * (fw - 1 + in_w_t) / lane
	return m.max2(load_phase_latency, write_phase_latency)

def weight_load_est_s(in_num_t, out_num_t, fh2, fw2, lane, dw1, dw2, dw3, port_width, point_en, fre, m):
	burst_len2 = in_num_t * out_num_t * fh2 * fw2 / (port_width / dw2)
	eff_bw2, eff_port_width2 = effective_dram_est(port_width, burst_len2, fre)
	load_phase_latency = in_num_t * out_num_t * fh2 * fw2 / (eff_port_width2 / dw2)
	write_phase_latency = in_num_t * out_num_t * fh2 * fw2 / lane
	latency = load_phase_latency + write_phase_latency#m.if3(point_en, 0, load_phase_latency + write_phase_latency)# m.if2(point_en, , 0)
	return m.min2(latency, latency*point_en)#m.if2(point_en, 0, latency)#72

def point_conv_est_s(in_num, in_num_t, out_num_t, in_h_t, in_w_t, out_h_t, out_w_t, fh2, fw2, lane, sa_rows, sa_cols, sa_lane, conv_en, exp_factor, m):
	cin_load = in_num_t * (fh2 - 1 + in_h_t) * (fw2 - 1 + in_w_t) / lane
	weight_load = in_num_t * out_num_t * fh2 * fw2 / lane
	load_phase_latency = m.max2(cin_load, weight_load)
	compute_phase_latency = in_num_t * out_num_t * (out_h_t/exp_factor) * (out_w_t/exp_factor) * fh2 * fw2 / sa_rows / sa_cols / sa_lane
	compute_drain_latency = out_num_t * out_w_t / sa_cols * out_h_t / (in_num / in_num_t)
	cout_write = out_num_t * out_h_t * out_w_t / (in_num / in_num_t) / lane
	write_phase_latency = cout_write
	latency = m.max2(m.max2(load_phase_latency, compute_phase_latency), m.max2(compute_drain_latency, write_phase_latency))
	return m.min2(latency, latency*conv_en)#m.if3(conv_en, 0, latency)#m.if2(conv_en, latency, 0)

def relu_est_s(in_num, in_num_t, out_num_t, out_h_t, out_w_t, lane):
	return out_num_t * out_h_t * out_w_t / lane / (in_num / in_num_t)

def cout_write_est_s(in_num, in_num_t, out_num_t, out_h_t, out_w_t, stride, lane, dw, port_width, fre, m):
	load_phase_latency = out_num_t * out_h_t * out_w_t / lane / (in_num / in_num_t)
	burst_len = out_w_t / stride * out_num_t / (port_width / dw)
	eff_bw, eff_port_width = effective_dram_est(port_width, burst_len, fre)
	write_phase_latency = out_num_t * out_h_t / stride * out_w_t / stride / (in_num / in_num_t) / (eff_port_width / dw)
	return m.max2(load_phase_latency, write_phase_latency)


def layer_latency_est(params):
	in_num = params['LAYER_IN_NUM']
	out_num = params['LAYER_OUT_NUM']
	in_h = params['LAYER_IN_H']
	in_w = params['LAYER_IN_W']
	in_num_t = params['LAYER_IN_NUM_T']
	out_num_t = params['LAYER_OUT_NUM_T']
	in_h_t = params['LAYER_IN_H_T']
	in_w_t = params['LAYER_IN_W_T']
	out_h_t = params['LAYER_OUT_H_T']
	out_w_t = params['LAYER_OUT_W_T']
	filter_s1 = params['LAYER_FILTER_S1']
	filter_s2 = params['LAYER_FILTER_S2']
	lane = params['SIMD_FACTOR']
	dw0 = params['DATA_W0']
	dw1 = params['DATA_W1']
	dw2 = params['DATA_W2']
	port_width = params['BUS_W']
	conv_en = params['CONV_EN']
	pool_en = params['POOL_EN']
	sa_rows = params['SA_ROWS']
	sa_cols = params['SA_COLS']
	sa_lane = params['SA_SIMD_LANE']
	stride = params['LAYER_STRIDE']
	fre = params['FRE']
	exp_factor = params['EXP_FACTOR']

	cin_load_latency = cin_load_est(in_num_t, in_h_t, in_w_t, max(filter_s1, filter_s2), max(filter_s1, filter_s2), lane, dw0, port_width, fre)
	weight_load_latency = weight_load_est(in_num_t, out_num_t, filter_s1, filter_s1, filter_s2, filter_s2, lane, dw0, dw1, dw2, port_width, conv_en, fre)
	if conv_en == 1:
		point_conv_latency = point_conv_est(in_num, in_num_t, out_num_t, in_h_t, in_w_t, out_h_t, out_w_t, filter_s1, filter_s1, filter_s2, filter_s2, lane, sa_rows, sa_cols, sa_lane, exp_factor)
	else:
		point_conv_latency = 0
	relu_latency = relu_est(in_num, in_num_t, out_num_t, out_h_t, out_w_t, lane)
	if pool_en == 1:
		pool_latency = pool_est(in_num, in_num_t, out_num_t, out_h_t, out_w_t, lane)
	else:
		pool_latency = 0
	cout_write_latency = cout_write_est(in_num, in_num_t, out_num_t, out_h_t, out_w_t, stride, lane, dw0, port_width, fre)

#  print("latency_breakdown: ", cin_load_latency, weight_load_latency, inter_load_latency, depth_conv_latency, point_conv_latency, relu_latency, pool_latency, inter_write_latency, cout_write_latency)
	stage_latency = max(cin_load_latency, weight_load_latency, point_conv_latency, relu_latency, pool_latency, cout_write_latency)
	# print(in_num_t, in_h_t, in_w_t, out_num_t)
	total_iter = np.ceil(in_num / in_num_t) * np.ceil(out_num / out_num_t) * np.ceil(in_h / in_h_t) * np.ceil(in_w / in_w_t)
#  print(in_num, out_num, in_h, in_w, in_num_t, out_num_t, in_h_t, in_w_t)
#  print("stage latency, total iter: ", stage_latency, total_iter)
	extra_latency = max(cin_load_latency, weight_load_latency) + cout_write_latency + point_conv_latency # the data drain latency is omitted
	total_latency = extra_latency + stage_latency * total_iter

#  dep_latency = max(cin_load_latency, weight_load_latency) + max(depth_conv_latency, point_conv_latency, relu_latency, pool_latency) + cout_write_latency
#  total_latency = max(stage_latency * total_iter, dep_latency)

	return total_latency

def latency_equations(layer_configs, hw_configs, in_num_t, out_num_t, in_h_t, in_w_t, m):
	#HW params constant for all layers
	lane = hw_configs['SIMD_LANE']
	dw0 = hw_configs['DATA_W0']
	dw1 = hw_configs['DATA_W1']
	dw2 = hw_configs['DATA_W2']
	port_width = hw_configs['BUS_W']
	fre = hw_configs['FRE']

	sa_rows = hw_configs['SA_ROWS']
	sa_cols = hw_configs['SA_COLS']
	sa_lane = hw_configs['SA_LANE']
	
	#layer params
	in_num = layer_configs['LAYER_IN_NUM']
	out_num = layer_configs['LAYER_OUT_NUM']
	in_h = layer_configs['LAYER_IN_H']
	in_w = layer_configs['LAYER_IN_W']
	# filter_s1 = layer_configs['LAYER_FILTER_S1']
	filter_s2 = layer_configs['LAYER_FILTER_S2']

	conv_en = layer_configs['CONV_EN']
	pool_en = layer_configs['POOL_EN']

	stride = layer_configs['LAYER_STRIDE']
	exp_factor = layer_configs['EXP_FACTOR']

	out_h_t = in_h_t*exp_factor/stride
	out_w_t = in_w_t*exp_factor/stride

	cin_load = cin_load_est_s(in_num_t, in_h_t, in_w_t, filter_s2, filter_s2, lane, dw0, port_width, fre, m)
	weight_load = weight_load_est_s(in_num_t, out_num_t, filter_s2, filter_s2, lane, dw0, dw1, dw2, port_width, conv_en, fre, m)
	conv = point_conv_est_s(in_num, in_num_t, out_num_t, in_h_t, in_w_t, out_h_t, out_w_t, filter_s2, filter_s2, lane, sa_rows, sa_cols, sa_lane, conv_en, exp_factor, m)
	relu = relu_est_s(in_num, in_num_t, out_num_t, out_h_t, out_w_t, lane)
	cout_write = cout_write_est_s(in_num, in_num_t, out_num_t, out_h_t, out_w_t, stride, lane, dw0, port_width, fre, m)
	tile_latency = m.max2(m.max2(m.max2(conv, cin_load), m.max2(cout_write, relu)), weight_load)
	total_iter = (out_num / out_num_t)*(in_num / in_num_t)*(in_w / in_w_t)*(in_h / in_h_t)
	total_latency = (tile_latency*total_iter)
	return total_latency
'''
sweep each layer, pick up the optimal in_num_t/out_num_t, in_h_t, in_w_t
SA_config is a list of [SA_ROWS, SA_COLS, SA_SIMD]
'''
def model_latency_est(params, layer_configs, sortedLayerNames, dynamic_tiling_level, total_layer, SA_config):
	latency = 0
	layer_id = 0
	layer_in_num_t_list = [None]*total_layer
	layer_out_num_t_list = [None]*total_layer
	layer_in_h_t_list = [None]*total_layer
	layer_in_w_t_list = [None]*total_layer
	visitedLayers = []
	layerOptParams = {}
	for layer_name in sortedLayerNames:
		layerOptParam = {}
		layerOptParam['LAYER_IN_NUM_T'] = 0
		layerOptParam['LAYER_OUT_NUM_T'] = 0
		layerOptParam['LAYER_IN_H_T'] = 0
		layerOptParam['LAYER_IN_W_T'] = 0
		layerOptParams[layer_name] = layerOptParam

	for layer_name in sortedLayerNames:
		layer_params = params.copy()
		layer_config = layer_configs[layer_name]
		layer_id = layer_config['LAYER_ID']
		layer_params.update(layer_config)
		layer_in_num_depend = layer_config['LAYER_IN_NUM_DEPEND']
		layer_out_num_depend = layer_config['LAYER_OUT_NUM_DEPEND']
		stride = layer_config['LAYER_STRIDE']
		exp_factor = layer_config['EXP_FACTOR']
		in_num_t = params['LAYER_IN_NUM_T']
		out_num_t = params['LAYER_OUT_NUM_T']
		in_h_t = params['LAYER_IN_H_T']
		in_w_t = params['LAYER_IN_W_T']
		sa_cols = params['SA_COLS']
		layer_name = layer_config['LAYER_NAME']

		if dynamic_tiling_level == 0:
			layer_in_num_t_candidates = [in_num_t]
			layer_out_num_t_candidates = [out_num_t]
		else:

			in_t_free = True
			pool_in_free = True
			if ('in', layer_name) in layer_in_num_depend:
				for tpl in layer_in_num_depend:
					if tpl[1] in visitedLayers:
						in_t_free = False
						layer_in_num_t_candidates = [layerOptParams[tpl[1]]['LAYER_OUT_NUM_T']] if tpl[0] == 'out' else [layerOptParams[tpl[1]]['LAYER_IN_NUM_T']]
						pool_in_free = False
						break
				if in_t_free:
					layer_in_num_t_candidates = list(filter(lambda x : x % SA_config[2] == 0, range(1, in_num_t + 1)))
			else:
				layer_in_num_t_candidates = list(filter(lambda x : x % SA_config[2] == 0, range(1, in_num_t + 1)))
			
			out_t_free = True
			pool_out_free = True
			if ('out', layer_name) in layer_out_num_depend:
				for tpl in layer_out_num_depend:
					if tpl[1] in visitedLayers:
						out_t_free = False
						layer_out_num_t_candidates = [layerOptParams[tpl[1]]['LAYER_IN_NUM_T']] if tpl[0] == 'in' else [layerOptParams[tpl[1]]['LAYER_OUT_NUM_T']]
						pool_out_free = False
						break
				if out_t_free:
					layer_out_num_t_candidates = list(filter(lambda x : x % SA_config[0] == 0, range(1, out_num_t + 1)))
			else:
				layer_out_num_t_candidates = list(filter(lambda x : x % SA_config[0] == 0, range(1, out_num_t + 1)))
			
			if(layer_config['POOL_EN']):
				if pool_in_free and pool_out_free:
					layer_io_num_t_candidates = list(filter(lambda x : x % SA_config[2] == 0, range(8, min(in_num_t, out_num_t) + 1)))
				elif pool_in_free:
					layer_io_num_t_candidates = layer_out_num_t_candidates#list(filter(lambda x : x % SA_config[2] == 0, range(8, out_num_t + 1)))
				elif pool_out_free:
					layer_io_num_t_candidates = layer_in_num_t_candidates
			# print(pool_in_free, pool_out_free)
			visitedLayers.append(layer_name)
		
		if dynamic_tiling_level == 0 or dynamic_tiling_level == 1:
			layer_in_h_t_candidates = [in_h_t]
			layer_in_w_t_candidates = [in_w_t]
		else:
			#layer_in_h_t_candidates = list(filter(lambda x : x % 2 == 0, range(1, in_h_t + 1)))
			layer_in_h_t_candidates = list(filter(lambda x : layer_config['LAYER_IN_H'] % x == 0, range(1, min(in_h_t,layer_config['LAYER_IN_H'])  + 1)))
			layer_in_w_t_candidates = list(filter(lambda x : x % sa_cols == 0 and layer_config['LAYER_IN_W'] % x == 0, range(1, min(in_w_t,layer_config['LAYER_IN_W'])  + 1)))
		
		
		opt_layer_latency = np.inf
		for layer_in_h_t in layer_in_h_t_candidates:
			for layer_in_w_t in layer_in_w_t_candidates:
				if(layer_config['POOL_EN']):
					for layer_io_num_t in layer_io_num_t_candidates:
						layer_params['LAYER_IN_NUM_T'] = layer_io_num_t
						layer_params['LAYER_OUT_NUM_T'] = layer_io_num_t
						layer_params['LAYER_IN_H_T'] = layer_in_h_t
						layer_params['LAYER_IN_W_T'] = layer_in_w_t
						layer_params['LAYER_OUT_H_T'] = int(layer_in_h_t*exp_factor/stride)
						layer_params['LAYER_OUT_W_T'] = int(layer_in_w_t*exp_factor/stride)
						layer_latency = layer_latency_est(layer_params)
						if layer_latency < opt_layer_latency:
							opt_layer_latency = layer_latency
							opt_layer_in_num_t = layer_io_num_t
							opt_layer_out_num_t = layer_io_num_t
							opt_layer_in_h_t = layer_in_h_t
							opt_layer_in_w_t = layer_in_w_t
				else:
					for layer_in_num_t in layer_in_num_t_candidates:
						for layer_out_num_t in layer_out_num_t_candidates:
							layer_params['LAYER_IN_NUM_T'] = layer_in_num_t
							layer_params['LAYER_OUT_NUM_T'] = layer_out_num_t
							layer_params['LAYER_IN_H_T'] = layer_in_h_t
							layer_params['LAYER_IN_W_T'] = layer_in_w_t
							layer_params['LAYER_OUT_H_T'] = int(layer_in_h_t*exp_factor/stride)
							layer_params['LAYER_OUT_W_T'] = int(layer_in_w_t*exp_factor/stride)
							layer_latency = layer_latency_est(layer_params)
							if layer_latency < opt_layer_latency:
								opt_layer_latency = layer_latency
								opt_layer_in_num_t = layer_in_num_t
								opt_layer_out_num_t = layer_out_num_t
								opt_layer_in_h_t = layer_in_h_t
								opt_layer_in_w_t = layer_in_w_t

		#print(layer_id, opt_layer_latency, opt_layer_in_num_t, opt_layer_out_num_t, opt_layer_in_h_t, opt_layer_in_w_t)
		layerOptParams[layer_name]['LAYER_IN_NUM_T'] = opt_layer_in_num_t
		layerOptParams[layer_name]['LAYER_OUT_NUM_T'] = opt_layer_out_num_t
		layerOptParams[layer_name]['LAYER_IN_H_T'] = opt_layer_in_h_t
		layerOptParams[layer_name]['LAYER_IN_W_T'] = opt_layer_in_w_t
		layer_in_num_t_list[layer_id] = opt_layer_in_num_t
		layer_out_num_t_list[layer_id] = opt_layer_out_num_t
		layer_in_h_t_list[layer_id] = opt_layer_in_h_t
		layer_in_w_t_list[layer_id] = opt_layer_in_w_t
		latency += opt_layer_latency
	
	params['LAYER_IN_NUM_T_LIST'] = layer_in_num_t_list
	params['LAYER_OUT_NUM_T_LIST'] = layer_out_num_t_list
	params['LAYER_IN_H_T_LIST'] = layer_in_h_t_list
	params['LAYER_IN_W_T_LIST'] = layer_in_w_t_list
	in_num_t = params['LAYER_IN_NUM_T']
	out_num_t = params['LAYER_OUT_NUM_T']
	in_h_t = params['LAYER_IN_H_T']
	in_w_t = params['LAYER_IN_W_T']
	#if in_num_t == 64 and out_num_t == 64 and in_h_t == 12 and in_w_t == 96:
	#  print(latency)
	return latency, params
#function to get the number of divisors of a number that are between a range of two numbers
def divisors(n, start, end):
		divisors = []
		for i in range(start, end):
				if n % i == 0:
						divisors.append(i)
		return len(divisors)
# print(divisors(256, 8, 96))
# exit()

def model_solver(params, layer_configs, sortedLayerNames, in_num_depend, out_num_depend, total_layer, SA_config):
	m = GEKKO(remote=False) # Initialize gekko
	# m.cleanup()
	# options.IMODE = 9 # Optimization
	# m._path = '/home/basalama/FlexCNN_TD/auto_compile/dse/dse_orig/'
	# m.path = '/home/basalama/FlexCNN_TD/auto_compile/dse/dse_orig/'
	m.options.SOLVER=1  # APOPT is an MINLP solver
	# optional solver settings with APOPT
	m.solver_options = ['minlp_maximum_iterations 500', \
						# minlp iterations with integer solution
						'minlp_max_iter_with_int_sol 10', \
						# treat minlp as nlp
						'minlp_as_nlp 0', \
						# nlp sub-problem max iterations
						'nlp_maximum_iterations 50', \
						# 1 = depth first, 2 = breadth first
						'minlp_branch_method 1', \
						# maximum deviation from whole number
						'minlp_integer_tol 0.05', \
						# covergence tolerance
						'minlp_gap_tol 0.01']
	params['SA_ROWS'] = SA_config[0]
	params['SA_COLS'] = SA_config[1]
	params['SA_LANE'] = SA_config[2]
	hw_configs = params

	#network params changing for each layer
	in_num_t = m.Array(m.Var, (total_layer))
	out_num_t = m.Array(m.Var, (total_layer))
	in_h_t = m.Array(m.Var, (total_layer))
	in_w_t = m.Array(m.Var, (total_layer))
	in_num_t_tmp = m.Array(m.Var, (total_layer))
	out_num_t_tmp = m.Array(m.Var, (total_layer))
	in_w_t_tmp = m.Array(m.Var, (total_layer))
	for i, layer_name in enumerate(sortedLayerNames):
		layer_config = layer_configs[layer_name]
		in_num = params['LAYER_IN_NUM_T']
		out_num = params['LAYER_OUT_NUM_T']
		in_w = params['LAYER_IN_W_T']
		in_h = params['LAYER_IN_H_T']

		# max_in_num_t = min(in_num, layer_config['LAYER_IN_NUM'])
		# in_num_t_tmp[i] = m.Var(value=1, lb=1,ub=max_in_num_t/8, integer=True)
		# in_num_t[i] = m.Intermediate(in_num_t_tmp[i]*8)

		# max_out_num_t = min(out_num, layer_config['LAYER_OUT_NUM'])
		# out_num_t_tmp[i] = m.Var(value=1, lb=1,ub=max_out_num_t/8, integer=True)
		# out_num_t[i] = m.Intermediate(out_num_t_tmp[i]*8)

		max_in_num_t = divisors(layer_config['LAYER_IN_NUM'], 8, min(in_num, layer_config['LAYER_IN_NUM']))
		in_num_t_tmp[i] = m.Var(value=1, lb=1,ub=max_in_num_t, integer=True)
		in_num_t[i] = m.Intermediate(2**(3+in_num_t_tmp[i]))

		max_out_num_t = divisors(layer_config['LAYER_OUT_NUM'], 8, min(out_num, layer_config['LAYER_OUT_NUM']))
		out_num_t_tmp[i] = m.Var(value=1, lb=1,ub=max_out_num_t, integer=True)
		out_num_t[i] = m.Intermediate(2**(3+out_num_t_tmp[i]))

		max_in_w_t = divisors(layer_config['LAYER_IN_W'], 8, min(in_w, layer_config['LAYER_IN_W']))
		in_w_t_tmp[i] = m.Var(value=1, lb=1,ub=max_in_w_t, integer=True)#[1,2,3,4,5,6,7,8,9,10,11,12]
		in_w_t[i] = m.Intermediate(2**(3+in_w_t_tmp[i]))#[8,16,24,32,40,48,56,64,72,80,88,96]
		
		# in_num_t[i] = m.sos1(list(filter(lambda x : x % 8 == 0 and layer_config['LAYER_IN_NUM'] % x == 0, range(1, in_num + 1))))
		
		# out_num_t[i] = m.sos1(list(filter(lambda x : x % 8 == 0 and layer_config['LAYER_OUT_NUM'] % x == 0, range(1, out_num + 1))))
		
		# in_w_t[i] = m.sos1(list(filter(lambda x : x % 8 == 0 and layer_config['LAYER_IN_W'] % x == 0, range(1, in_w + 1))))

		# in_num_t[i] = m.Var(value=8, lb=8,ub=in_num, integer=True)
		
		# out_num_t[i] = m.Var(value=8, lb=8,ub=out_num, integer=True)
		
		# in_w_t[i] = m.Var(value=8, lb=8,ub=in_w, integer=True)
		# max_in_h_t = divisors(layer_config['LAYER_IN_H'], 1, min(in_h, layer_config['LAYER_IN_H']))
		# print(max_in_h_t)
		# # in_w_t_tmp[i] = m.Var(value=1, lb=1,ub=max_in_h_t, integer=True)#[1,2,3,4,5,6,7,8,9,10,11,12]
		# # in_w_t[i] = m.Intermediate(2**(3+in_w_t_tmp[i]))#[8,16,24,32,40,48,56,64,72,80,88,96]
		# lowerBound = 2 if layer_config['LAYER_STRIDE']==2 else 1
		# upperBound = in_h
		in_h_t[i] = m.Var(value=in_h, lb=in_h, ub=in_h, integer=True)

	L = []
	for i, layer_name in enumerate(sortedLayerNames):
		layer_config = layer_configs[layer_name]
		# print(layer_config['STRIDE'], layer_config['EXP_FACTOR'])
		L.append(latency_equations(layer_config, hw_configs, in_num_t[i], out_num_t[i], in_h_t[i], in_w_t[i], m))
	# exit()
	for i in range(len(in_num_depend)):
		in_index = in_num_depend[i]
		out_index = out_num_depend[i]
		# print(out_index, in_index)
		m.Equation(out_num_t[out_index] == in_num_t[in_index])
	for i, layer_name in enumerate(sortedLayerNames):
		if 'pool' in layer_name:
			m.Equation(out_num_t[i] == in_num_t[i])
	
	m.Minimize(m.sum(L))
	m.solve(disp=False)
	latency = float(m.options.objfcnval)
	in_num_t_list = []
	out_num_t_list = []
	in_h_t_list = []
	in_w_t_list = []
	for i in range(total_layer):
		in_num_t_list.append(int(in_num_t[i].value[0]))
		out_num_t_list.append(int(out_num_t[i].value[0]))
		in_h_t_list.append(int(in_h_t[i].value[0]))
		in_w_t_list.append(int(in_w_t[i].value[0]))
	params['LAYER_IN_NUM_T_LIST'] = in_num_t_list
	params['LAYER_OUT_NUM_T_LIST'] = out_num_t_list
	params['LAYER_IN_H_T_LIST'] = in_h_t_list
	params['LAYER_IN_W_T_LIST'] = in_w_t_list
	return latency, params

def BRAM_SDP_predict_HLS(dw, s):
	if dw > 18:
		alpha = np.ceil(dw / 36)
		BRAM = alpha * np.ceil(s / dw / 512)
	else:
		alpha = np.ceil(dw / 18)
		BRAM = alpha * np.ceil(s / dw / 1024)
	return BRAM

def SA_BRAM_est(params):
  SIMD_FACTOR = params['SIMD_FACTOR']
  SA_ROWS = params['SA_ROWS']
  SA_COLS = params['SA_COLS']
  SA_SIMD_LANE = params['SA_SIMD_LANE']
  IN_NUM_T = params['LAYER_IN_NUM_T']
  OUT_NUM_T = params['LAYER_OUT_NUM_T']
  IN_H_T =  params['LAYER_IN_H_T']
  IN_W_T = params['LAYER_IN_W_T']
  OUT_H_T = params['LAYER_OUT_H_T']
  OUT_W_T = params['LAYER_OUT_W_T']
  K_T = params['K_T']

  U1_DATA0_WIDTH = params['DATA_W0']
  U1_DATA1_WIDTH = params['DATA_W1']
  U1_DATA2_WIDTH = params['DATA_W2']

  U1_IN_IMG_W_T = (IN_W_T+K_T-1)
  U1_IN_IMG_H_T = (IN_H_T+K_T-1)
  U1_K = K_T
  U1_OUT_NUM_T = OUT_NUM_T
  U1_IN_NUM_T = IN_NUM_T
  U1_OUT_IMG_H_T = OUT_H_T
  U1_OUT_IMG_W_T = OUT_W_T
  U1_SA_ROWS = SA_ROWS
  U1_SA_COLS = SA_COLS
  U1_SIMD_FACTOR = SIMD_FACTOR
  U1_ROW_IL_FACTOR = int(OUT_NUM_T/U1_SA_ROWS)
  U1_COL_IL_FACTOR = int(OUT_W_T/U1_SA_COLS)
  U1_LOCAL_REG_NUM = (OUT_H_T*U1_ROW_IL_FACTOR*U1_COL_IL_FACTOR) 
  U1_DATA0_FC_SIMD_FACTOR = SIMD_FACTOR
  U1_DATA0_FC_GROUP_FACTOR = 1
  U1_DATA0_FC_SPLIT_FACTOR = 1
  U1_DATA1_FC_SIMD_FACTOR = SIMD_FACTOR
  U1_DATA1_FC_GROUP_FACTOR = 1
  U1_DATA1_FC_SPLIT_FACTOR = 1
  U1_DATA2_FC_SIMD_FACTOR = SIMD_FACTOR
  U1_DATA2_FC_GROUP_FACTOR = 1
  U1_DATA2_FC_SPLIT_FACTOR = 1
  U1_DATA0_BUF_SIZE = (U1_IN_NUM_T * U1_IN_IMG_H_T * (U1_COL_IL_FACTOR+U1_K-1))	      			
  U1_DATA1_BUF_SIZE = (U1_IN_NUM_T * U1_ROW_IL_FACTOR * U1_K * U1_K)       
  U1_DATA2_BUF_SIZE = (U1_OUT_NUM_T * U1_OUT_IMG_H_T * U1_COL_IL_FACTOR)			

  estimate = 0

  #U1_DataFeed0Head
    # ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> cin_buf[U1_IN_NUM_T * U1_IN_IMG_H_T * U1_IN_IMG_W_T / U1_DATA0_FC_SIMD_FACTOR]

  U1_DataFeed0Head = BRAM_SDP_predict_HLS(U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR, U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR * U1_IN_NUM_T * U1_IN_IMG_H_T * U1_IN_IMG_W_T / U1_DATA0_FC_SIMD_FACTOR)
  # #print(U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR * U1_IN_NUM_T * U1_IN_IMG_H_T * U1_IN_IMG_W_T / U1_DATA0_FC_SIMD_FACTOR)
  #print('U1_DataFeed0Head',U1_DataFeed0Head, 128)
  estimate += U1_DataFeed0Head

  # #U1_DataFeed0Engine0
  # 	U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR ping_buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR]
  # 	U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR pong_buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR]

  U1_DataFeed0Engine0_data      = BRAM_SDP_predict_HLS(U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR, U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR * U1_DATA0_FC_GROUP_FACTOR * U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR) * 2
  U1_DataFeed0Engine0_feeder_id = BRAM_SDP_predict_HLS(32, 32 * U1_DATA0_FC_GROUP_FACTOR * U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR) * 2
  U1_DataFeed0Engine0_filter_s  = BRAM_SDP_predict_HLS(32, 32 * U1_DATA0_FC_GROUP_FACTOR * U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR) * 2
  U1_DataFeed0Engine0_new_pair  = BRAM_SDP_predict_HLS(1, 1 * U1_DATA0_FC_GROUP_FACTOR * U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR) * 2
  U1_DataFeed0Engine0_last_pair = BRAM_SDP_predict_HLS(1, 1 * U1_DATA0_FC_GROUP_FACTOR * U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR) * 2
  U1_DataFeed0Engine0 = U1_DataFeed0Engine0_data + U1_DataFeed0Engine0_feeder_id + U1_DataFeed0Engine0_filter_s + U1_DataFeed0Engine0_new_pair + U1_DataFeed0Engine0_last_pair
  #print('U1_DataFeed0Engine0',U1_DataFeed0Engine0, 78, U1_DataFeed0Engine0*8, 78*8)
  estimate += U1_DataFeed0Engine0*8

  # #U1_DataFeed1Head
  # 	ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> weight_buf[U1_IN_NUM_T * U1_OUT_NUM_T * U1_K * U1_K / U1_DATA1_FC_SIMD_FACTOR]

  U1_DataFeed1Head = BRAM_SDP_predict_HLS(U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR, U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR * U1_IN_NUM_T * U1_OUT_NUM_T * U1_K * U1_K / U1_DATA1_FC_SIMD_FACTOR)

  #print('U1_DataFeed1Head',U1_DataFeed1Head, 0)
  estimate += U1_DataFeed1Head

  #U1_DataFeed1Engine0
  # 	U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR ping_buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR]
  # 	U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR pong_buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR]
  U1_DataFeed1Engine0_data      = BRAM_SDP_predict_HLS(U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR, U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR * U1_DATA1_FC_GROUP_FACTOR * U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR) * 2
  U1_DataFeed1Engine0_feeder_id = BRAM_SDP_predict_HLS(32, 32 * U1_DATA1_FC_GROUP_FACTOR * U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR) * 2
  U1_DataFeed1Engine0_filter_s  = BRAM_SDP_predict_HLS(32, 32 * U1_DATA1_FC_GROUP_FACTOR * U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR) * 2
  U1_DataFeed1Engine0_new_pair  = BRAM_SDP_predict_HLS(1, 1 * U1_DATA1_FC_GROUP_FACTOR * U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR) * 2
  U1_DataFeed1Engine0_last_pair = BRAM_SDP_predict_HLS(1, 1 * U1_DATA1_FC_GROUP_FACTOR * U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR) * 2
  U1_DataFeed1Engine0 = U1_DataFeed1Engine0_data + U1_DataFeed1Engine0_feeder_id + U1_DataFeed1Engine0_filter_s + U1_DataFeed1Engine0_new_pair + U1_DataFeed1Engine0_last_pair

  #print('U1_DataFeed1Engine0',U1_DataFeed1Engine0, 78, U1_DataFeed1Engine0*8, 78*8)
  estimate += U1_DataFeed1Engine0*8

  # #U1_DataCollect2Head
  # 	ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> cout_buf[U1_OUT_IMG_H_T * U1_OUT_IMG_W_T * U1_OUT_NUM_T / U1_DATA2_FC_SIMD_FACTOR]

  U1_DataCollect2Head = BRAM_SDP_predict_HLS(U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR, U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR * U1_OUT_IMG_H_T * U1_OUT_IMG_W_T * U1_OUT_NUM_T / U1_DATA2_FC_SIMD_FACTOR)

  #print('U1_DataCollect2Head',U1_DataCollect2Head, 0)
  estimate += U1_DataCollect2Head

  # #U1_DataCollect2Engine0
  #   U1_data_t2 ping_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR]
  # 	U1_data_t2 pong_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR]

  U1_DataCollect2Engine0 = BRAM_SDP_predict_HLS(32, 32 * U1_DATA2_FC_GROUP_FACTOR * U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR * U1_DATA2_FC_SIMD_FACTOR) * 2

  #print('U1_DataCollect2Engine0',U1_DataCollect2Engine0, 32, U1_DataCollect2Engine0*8, 32*8)
  estimate += U1_DataCollect2Engine0*8

  # #U1_res_transfer
  #   U1_data_t2 local_buffer[U1_LOCAL_REG_NUM]

  U1_res_transfer = BRAM_SDP_predict_HLS(32, 32 * U1_LOCAL_REG_NUM)

  #print('U1_res_transfer',U1_res_transfer, 2, U1_res_transfer*8*8, 2*8*8)
  estimate += U1_res_transfer*8*8

  # U1_compute
    # U1_data_t2 local_buffer[U1_LOCAL_REG_NUM]

  U1_compute = BRAM_SDP_predict_HLS(32, 32 * U1_LOCAL_REG_NUM)

  #print('U1_compute',U1_compute, 2, U1_compute*8*8, 2*8*8)
  estimate += U1_compute*8*8

  return estimate

def res_est(params):
	SIMD_LANE = params['SIMD_FACTOR']
	SA_ROWS = params['SA_ROWS']
	SA_COLS = params['SA_COLS']
	SA_SIMD_LANE = params['SA_SIMD_LANE']
	LAYER_IN_NUM_T = params['LAYER_IN_NUM_T']
	LAYER_OUT_NUM_T = params['LAYER_OUT_NUM_T']
	LAYER_IN_H_T = params['LAYER_IN_H_T']
	LAYER_IN_W_T = params['LAYER_IN_W_T']
	LAYER_OUT_H_T = params['LAYER_OUT_H_T']
	LAYER_OUT_W_T = params['LAYER_OUT_W_T']
	LAYER_K_T = params['K_T']

	# estimate DSPs
	if params['DATA_T0'] == "float":
		DSP_per_MAC = 5
	elif params['DATA_T0'] == "ap_fixed<16>":
		DSP_per_MAC = 1
	# depth_conv
	depth_conv_DSP = (3 * 3 * SIMD_LANE + 1 * 1 * SIMD_LANE) * DSP_per_MAC
	# point_conv
	point_conv_DSP = SA_ROWS * SA_COLS * SA_SIMD_LANE * DSP_per_MAC
	DSP = depth_conv_DSP + point_conv_DSP

	# estimate BRAMs
	# cin_load
	cin_load_BRAM = BRAM_SDP_predict_HLS(params['BUS_W'], params['DATA_W0'] * LAYER_IN_NUM_T * (LAYER_IN_H_T + LAYER_K_T - 1) * (LAYER_IN_W_T + LAYER_K_T - 1)) * 2
	# weight_load
	weight_load_BRAM = BRAM_SDP_predict_HLS(params['BUS_W'], params['DATA_W1'] * LAYER_IN_NUM_T * LAYER_K_T * LAYER_K_T) + BRAM_SDP_predict_HLS(params['BUS_W'], params['DATA_W1'] * LAYER_IN_NUM_T * LAYER_OUT_NUM_T * LAYER_K_T * LAYER_K_T) + BRAM_SDP_predict_HLS(params['BUS_W'], params['DATA_W2'] * LAYER_OUT_NUM_T)
	# point_conv
	ROW_IL_FACTOR = LAYER_OUT_NUM_T / SA_ROWS
	COL_IL_FACTOR = LAYER_OUT_W_T / SA_COLS
	LOCAL_REG_NUM = LAYER_OUT_H_T * ROW_IL_FACTOR * COL_IL_FACTOR
	point_conv_BRAM = \
		BRAM_SDP_predict_HLS(params['DATA_W0'] * SIMD_LANE, LAYER_IN_NUM_T * (LAYER_IN_H_T + LAYER_K_T - 1) * (LAYER_IN_W_T + LAYER_K_T - 1) * params['DATA_W0']) + \
		BRAM_SDP_predict_HLS(params['DATA_W0'] * SIMD_LANE, LAYER_IN_NUM_T * (LAYER_IN_H_T + LAYER_K_T - 1) * (COL_IL_FACTOR + LAYER_K_T - 1) * params['DATA_W0']) * 2 * SA_COLS + \
		BRAM_SDP_predict_HLS(params['DATA_W1'] * SIMD_LANE, LAYER_IN_NUM_T * ROW_IL_FACTOR * LAYER_K_T * LAYER_K_T * params['DATA_W1']) * 2 * SA_ROWS + \
		BRAM_SDP_predict_HLS(params['DATA_W0'], LAYER_OUT_NUM_T * LAYER_OUT_H_T * COL_IL_FACTOR * params['DATA_W0'] / SIMD_LANE) * SIMD_LANE * 2 * SA_COLS + \
		BRAM_SDP_predict_HLS(params['DATA_W0'], LOCAL_REG_NUM * params['DATA_W0']) * 3 * SA_ROWS * SA_COLS
	# cout_write
	cout_write_BRAM = BRAM_SDP_predict_HLS(params['BUS_W'], params['DATA_W0'] * LAYER_OUT_H_T * LAYER_OUT_W_T * LAYER_OUT_NUM_T) * 2

	BRAM18K = cin_load_BRAM + weight_load_BRAM + point_conv_BRAM + cout_write_BRAM

	return DSP, BRAM18K

def performanceDetails(params, layer_configs):
	# print(layer)
	totalMACs = 0
	totalModelCycles = 0
	totalSolverCycles = 0
	totalCycles = 0
	# print('cin_load_latency, weight_load_latency, inter_load_latency, depth_conv_latency, point_conv_latency, relu_latency, pool_latency, inter_write_latency, cout_write_latency')
	for i, layer_name in enumerate(layer_configs):
		layer = layer_configs[layer_name]
		instDict = {}
		instDict['LAYER_IN_NUM'] = layer['LAYER_IN_NUM']
		instDict['LAYER_OUT_NUM'] = layer['LAYER_OUT_NUM']
		instDict['LAYER_IN_H'] = layer['LAYER_IN_H']
		instDict['LAYER_IN_W'] = layer['LAYER_IN_W']
		instDict['LAYER_OUT_H'] = layer['LAYER_OUT_H']
		instDict['LAYER_OUT_W'] = layer['LAYER_OUT_W']
		instDict['LAYER_IN_NUM_T'] = params['LAYER_IN_NUM_T_LIST'][i]
		instDict['LAYER_OUT_NUM_T'] = params['LAYER_OUT_NUM_T_LIST'][i]
		instDict['LAYER_IN_H_T'] = params['LAYER_IN_H_T_LIST'][i]
		instDict['LAYER_IN_W_T'] = params['LAYER_IN_W_T_LIST'][i]
		instDict['LAYER_OUT_H_T'] = params['LAYER_IN_H_T_LIST'][i]/layer['LAYER_STRIDE']
		instDict['LAYER_OUT_W_T'] = params['LAYER_IN_W_T_LIST'][i]/layer['LAYER_STRIDE']
		instDict['LAYER_FILTER_S1'] = 1
		instDict['LAYER_FILTER_S2'] = layer['LAYER_FILTER_S2']
		instDict['SIMD_FACTOR'] = params['SIMD_LANE']
		instDict['DATA_W0'] = 32
		instDict['DATA_W1'] = 32
		instDict['DATA_W2'] = 32
		instDict['BUS_W'] = 512
		instDict['CONV_EN'] = layer['CONV_EN']
		instDict['POOL_EN'] = layer['POOL_EN']
		instDict['SA_ROWS'] = params['SA_ROWS']
		instDict['SA_COLS'] = params['SA_COLS']
		instDict['SA_SIMD_LANE'] = params['SIMD_LANE']
		instDict['LAYER_STRIDE'] = layer['LAYER_STRIDE']
		instDict['FRE'] = params['FRE']
		layerModelCycles = layer_latency_debug(instDict)[0]
		layerSolverCycles = layer_latency_solver(instDict)[0]
		layerMACs  =  (instDict['LAYER_FILTER_S2']*\
									instDict['LAYER_FILTER_S2']*\
									instDict['LAYER_IN_NUM']*\
									instDict['LAYER_OUT_NUM']*\
									instDict['LAYER_OUT_H']*\
									instDict['LAYER_OUT_W'])\
									/instDict['LAYER_STRIDE'] if instDict['CONV_EN']==1 else 0
		dimensions = [layer['LAYER_IN_NUM'], layer['LAYER_OUT_NUM'], layer['LAYER_IN_H'], layer['LAYER_IN_W']]
		totalModelCycles += layerModelCycles
		totalSolverCycles += layerSolverCycles
		totalMACs += layerMACs
		# print('{:<10}'.format(layer['LAYER_NAME']), '{:<20}'.format(str(dimensions)), '{:<10}'.format(int(layer_latency_debug(instDict)[0])), '{:<15}'.format(layer_latency_debug(instDict)[1]), '{}'.format(layer_latency_debug(instDict)[3]))

	totalTheoriticalCycles = totalMACs/(params['SIMD_LANE']*params['SA_ROWS']*params['SA_COLS'])
	print('Netork total MACs:', totalMACs)
	print(totalTheoriticalCycles/(params['FRE'] * 1e6), totalModelCycles/(params['FRE'] * 1e6), totalSolverCycles/(params['FRE'] * 1e6))
	# print('opt DSP efficiency: ', '{:.2f}'.format(100*(totalTheoriticalCycles/totalModelCycles)), '%')
	# print("SA Shape: %dx%dx%d (ROWSxCOLSxSIMD)" % (params['SA_ROWS'], params['SA_COLS'], params['SA_SIMD_LANE']))

#function to create a list of first element from a list of tuples
def get_first_element(list_of_tuples):
	return [x[0] for x in list_of_tuples]

#function to sort list of tuples by second element of the tuples in decreasing order
def sort_by_second_element(list_of_tuples):
	return sorted(list_of_tuples, key=lambda x: x[1])

#sort a litst of dicts by a key
def sort_by_key(list_of_dicts, key):
	return sorted(list_of_dicts, key=lambda x: x[key])

def run(f_model, f_input_config, f_board, parallel_en, systolic_en, dynamic_tiling_level, solver_en, output_dir):
	print("*************************************************")
	# record start time
	global_timer_start = time.time()

	model = open(f_model, "r")
	with open(f_input_config, "r") as f:
		input_config = json.loads(f.read())
	with open(f_board, "r") as f:
		board_info = json.loads(f.read())

	config = {}
	config['BOARD'] = board_info
	config['DYNAMIC_TILING_LEVEL'] = dynamic_tiling_level
	print('Dynamic tiling level: ', dynamic_tiling_level)

	params = {}
	"""
	Data Precision
	"""
	params['DATA_W0'] = 32
	params['DATA_W1'] = 32
	params['DATA_W2'] = 32
	params['BUS_W'] = 512
	params['DATA_T0'] = "float"
	params['DATA_T1'] = "float"
	params['DATA_T2'] = "float"
	"""
	Tiling Size
	"""
	K_T = 5
	params['K_T'] = K_T


	# input info
	network_in_num = input_config["IN_NUM"]
	network_in_h = input_config["IN_H"]
	network_in_w = input_config["IN_W"]

	# get the maximal channel number throughout the network, get the layer configurations
	network_channel_max = network_in_num
	
	modelLines = model.readlines()

	in_num = network_in_num
	out_num = network_in_num
	in_h = network_in_h
	in_w = network_in_w
	layer_configs = {}
	total_layer = 0

	for i, line in enumerate(modelLines):
		## extract the information of each layer
		if i == 0: continue
		line = line.strip('\n')
		content = line.split(";")
		total_layer += 1
		network_channel_max = max(network_channel_max, int(content[2]))
		#content 0 [Name,InputTensor]
		tensors = eval(content[0])
		layer_name = tensors[0]
		prev_layer_list = list(tensors[1:]) 
		#content 1 Type
		layer_type = content[1]
		pool_en = 1 if content[1] == "Pool" else 0
		conv_en = 1 if content[1] == "Conv2D" else 0
		#content 2 InChannel
		in_num = int(content[2])
		#content 3 OutChannel
		out_num = int(content[3])
		#content 4 Filter
		filter_s = int(content[4])
		#content 5 ExpansionFactor
		exp_factor = int(content[5])
		#content 6 Stride
		stride = int(content[6])
		min_prev_h = np.inf
		min_prev_w = np.inf
		for prev_name in prev_layer_list:
			if prev_name != 'input':
				prev_layer_config = layer_configs[prev_name]
				if prev_layer_config['LAYER_OUT_H'] < min_prev_h:
					min_prev_h = prev_layer_config['LAYER_OUT_H']
				if prev_layer_config['LAYER_OUT_W'] < min_prev_w:
					min_prev_w = prev_layer_config['LAYER_OUT_W']
		#infer remaining network data
		in_h = network_in_h if "input" in prev_layer_list else min_prev_h
		in_w = network_in_w if "input" in prev_layer_list else min_prev_w
		out_h = int(math.ceil(in_h*exp_factor/stride))
		out_w = int(math.ceil(in_w*exp_factor/stride))

		layer_config = {}
		layer_config['LAYER_NAME'] = layer_name
		layer_config['LAYER_ID'] = i-1
		layer_config['LAYER_IN_NUM'] = in_num
		layer_config['LAYER_OUT_NUM'] = out_num
		layer_config['LAYER_IN_H'] = in_h
		layer_config['LAYER_IN_W'] = in_w
		layer_config['LAYER_OUT_H'] = out_h
		layer_config['LAYER_OUT_W'] = out_w
		if layer_type == 'Conv2D':
			layer_config['LAYER_FILTER_S1'] = 1
			layer_config['LAYER_FILTER_S2'] = filter_s
		elif layer_type == 'Pool' or layer_type == "Identity":
			layer_config['LAYER_FILTER_S1'] = 1
			layer_config['LAYER_FILTER_S2'] = 1
		layer_config['LAYER_STRIDE'] = stride
		layer_config['CONV_EN'] = conv_en
		layer_config['POOL_EN'] = pool_en
		layer_config['EXP_FACTOR'] = exp_factor
		layer_config['PREV_LAYERS_NAMES'] = prev_layer_list
		layer_config['NEXT_LAYERS_NAMES'] = []
		for prev_layer_name in prev_layer_list:
			if(prev_layer_name in list(layer_configs.keys())):
				layer_configs[prev_layer_name]['NEXT_LAYERS_NAMES'].append(layer_name)
		layer_configs[layer_name] = layer_config
	
	tples_list = []
	for key in layer_configs.keys():
		tples_list.append((key, layer_configs[key]['LAYER_ID']))
	tples_list = sort_by_second_element(tples_list)
	sortedLayerNames = get_first_element(tples_list)

	out_num_depend = []
	in_num_depend = []
	for layer_name in sortedLayerNames:
		layer_config = layer_configs[layer_name]
		next_layer_names = layer_config['NEXT_LAYERS_NAMES']
		for next_layer_name in next_layer_names:
			next_layer_config = layer_configs[next_layer_name]
			in_num_depend.append(next_layer_config['LAYER_ID'])
			out_num_depend.append(layer_config['LAYER_ID'])
	
	in_num_depend_v = []
	out_num_depend_v = []
	for layer_name in sortedLayerNames:
		layer_config = layer_configs[layer_name]
		next_layer_names = layer_config['NEXT_LAYERS_NAMES']
		for next_layer_name in next_layer_names:
			next_layer_config = layer_configs[next_layer_name]
			in_num_depend_v.append(('in', next_layer_config['LAYER_NAME']))
			out_num_depend_v.append(('out', layer_config['LAYER_NAME']))
			
	#convert two lists to graph
	G = nx.Graph()
	G.add_edges_from(zip(out_num_depend_v, in_num_depend_v))
	# get disconnected components
	components = list(nx.connected_components(G))

	for layer_name in sortedLayerNames:
		layer_config = layer_configs[layer_name]
		layer_config['LAYER_IN_NUM_DEPEND'] = []
		layer_config['LAYER_OUT_NUM_DEPEND'] = []
		for component in components:
			if ('in', layer_name) in component:
				layer_config['LAYER_IN_NUM_DEPEND'] = list(component)
			if ('out', layer_name) in component:
				layer_config['LAYER_OUT_NUM_DEPEND'] = list(component)
	
	# for layer_name in sortedLayerNames:
	# 	layer_config = layer_configs[layer_name]
	# 	in_h = layer_config['LAYER_IN_H']
	# 	in_w = layer_config['LAYER_IN_W']
	# 	out_h = layer_config['LAYER_OUT_H']
	# 	out_w = layer_config['LAYER_OUT_W']
	# 	print(layer_name, in_h, in_w, out_h, out_w)
	# exit()
	# first_layer = layer_configs['conv_1']
	# print(first_layer['LAYER_IN_NUM_DEPEND'], first_layer['LAYER_OUT_NUM_DEPEND'])
	# exit()
	# Start the design space exploration
	# It works in a greedy fashion, as we will minimize the latency layer by layer.
	opt_latency = np.inf
	opt_DSP = np.inf
	opt_BRAM18K = np.inf
	opt_params = {}

	params_list = []
	# Construct the list of all different tiling factors
	for IN_H_T in [2,4,8]:#list(filter(lambda x : network_in_h % x == 0 and x % 2 == 0, range(2, 16 + 1))): # upper_bound
		for IN_W_T in [8,16,32,64]:#list(filter(lambda x : network_in_w % x == 0 and x % 2 == 0, range(1, 128 + 1))): # upper_bound
			for IN_NUM_T in [8,16,32]:#list(filter(lambda x : x % 8 == 0, range(1, 64 + 1))): # upper_bound
				for OUT_NUM_T in [16,32]: # upper_bound
					for SIMD_LANE in [8]:#list(filter(lambda x : IN_NUM_T % x == 0 and x % 8 == 0, range(1, min(IN_NUM_T, 8) + 1))):
#  for IN_H_T in [12]: # upper_bound
#    for IN_W_T in [96]: # upper_bound
#      for IN_NUM_T in [64]: # upper_bound
#        for SIMD_LANE in [8]:
#          debug_cnt += 1
#          print(debug_cnt)
         		# print(IN_NUM_T, IN_W_T, IN_H_T, SIMD_LANE)
					# IN_NUM_T = 64
					# OUT_NUM_T = 64
					# IN_H_T = 2
					# IN_W_T = 64
					# OUT_H_T = 4
					# OUT_W_T = 128
						params['LAYER_IN_H_T'] = IN_H_T
						params['LAYER_IN_W_T'] = IN_W_T
						params['LAYER_OUT_H_T'] = IN_H_T*2 #2 for expansion
						params['LAYER_OUT_W_T'] = IN_W_T*2 #2 for expansion
						params['LAYER_IN_NUM_T'] = IN_NUM_T
						params['LAYER_OUT_NUM_T'] = OUT_NUM_T
						# print(params['LAYER_IN_NUM_T'], params['LAYER_OUT_NUM_T'], params['LAYER_IN_H_T'], params['LAYER_IN_W_T'])
						# print(IN_NUM_T*IN_W_T)
						params['SIMD_FACTOR'] = SIMD_LANE
						tmp_params = dict(params)
						params_list.append(tmp_params)

	num = 0
	for params_t in params_list:
		params = dict(params_t)
		IN_NUM_T = params['LAYER_IN_NUM_T']
		IN_H_T = params['LAYER_IN_H_T']
		IN_W_T = params['LAYER_IN_W_T']
		SIMD_LANE = params['SIMD_FACTOR']
#    print(IN_NUM_T, IN_W_T, SIMD_LANE)
		for SA_ROWS in list(filter(lambda x : IN_NUM_T % x == 0, range(1, IN_NUM_T + 1))):
			for SA_COLS in list(filter(lambda x : IN_W_T % x == 0, range(1, IN_W_T + 1))):
				for SA_SIMD_LANE in list(filter(lambda x : SIMD_LANE % x == 0, range(1, SIMD_LANE + 1))):
					num += 1
					
	layer_configs_pd = pd.DataFrame(layer_configs)
	layer_configs_pd.to_csv(os.path.join(output_dir, "output_layer.csv"))
	
	params_list_pd = pd.DataFrame(params_list)
	params_list_pd.to_csv(os.path.join(output_dir, "params.csv"))
	
	
	#return
	
	if parallel_en is True:
		num_processes = int(multiprocessing.cpu_count() * 0.75)
	else:
		num_processes = 1
	print('Parallelizing using %d processes...' % (num_processes))

	chunks = list_split(params_list, num_processes)
	pool = multiprocessing.Pool(processes = num_processes)
	results = pool.starmap(param_sweep, [(chunk, config, layer_configs, sortedLayerNames, in_num_depend, out_num_depend, total_layer, systolic_en, solver_en) for chunk in chunks])
#  result = param_sweep(params_list, config, model_config, layer_configs)
	sorted_results = sort_by_key(results, 'opt_latency')
	print('----------------------------top 10 designs----------------------------')
	for i in range(10):
		result = sorted_results[i]
		opt_params = result['opt_params']
		in_num_t_list = opt_params["LAYER_IN_NUM_T_LIST"]
		max_in_num_t = max(in_num_t_list)
		in_h_t_list = opt_params["LAYER_IN_H_T_LIST"]
		max_in_h_t = max(in_h_t_list)
		in_w_t_list = opt_params["LAYER_IN_W_T_LIST"]
		max_in_w_t = max(in_w_t_list)
		out_num_t_list = opt_params["LAYER_OUT_NUM_T_LIST"]
		max_out_num_t = max(out_num_t_list)
		print(result['opt_latency'], result['opt_BRAM18K'], max_in_num_t, max_out_num_t, max_in_h_t, max_in_w_t)
	solver_fails = 0
	model_fails = 0
	print('Aggregating results...')
	for result in results:
		cur_latency = result['opt_latency']
		cur_DSP = result['opt_DSP']
		cur_BRAM18K = result['opt_BRAM18K']
		cur_params = result['opt_params']
		solver_fails += result['solver_fails']
		model_fails += result['model_fails']

		if cur_latency < opt_latency:
			opt_latency = cur_latency
			opt_DSP = cur_DSP
			opt_BRAM18K = cur_BRAM18K
			opt_params = cur_params
		elif cur_latency == opt_latency:
			if cur_DSP < opt_DSP or (cur_DSP == opt_DSP and cur_BRAM18K < opt_BRAM18K):
				opt_latency = cur_latency
				opt_DSP = cur_DSP
				opt_BRAM18K = cur_BRAM18K
				opt_params = cur_params

#  opt_latency = result['opt_latency']
#  opt_DSP = result['opt_DSP']
#  opt_BRAM18K = result['opt_BRAM18K']
#  opt_params = result['opt_params']
	# print(opt_params)
	print("*************************************************")
	if(solver_en):
		print("solver fails: ", solver_fails)
	else:
		print("model fails: ", model_fails)
	print("*************************************************")
	#print("finish", cur_latency, opt_latency)
	opt_in_num_t = opt_params['LAYER_IN_NUM_T_LIST']
	opt_out_num_t = opt_params['LAYER_OUT_NUM_T_LIST']
	
	dependency_broken = False
	for component in components:
		compList = list(component)
		layer_name = compList[0][1]
		layer_config = layer_configs[layer_name]
		layer_id = layer_config['LAYER_ID']
		comp_t = opt_in_num_t[layer_id] if compList[0][0] == 'in' else opt_out_num_t[layer_id]
		for tpl in compList:
			layer_name = tpl[1]
			layer_config = layer_configs[layer_name]
			layer_id = layer_config['LAYER_ID']
			layer_t = opt_in_num_t[layer_id] if tpl[0] == 'in' else opt_out_num_t[layer_id]
			if(layer_t != comp_t):
				dependency_broken = True
				# print('Error: component %s is not consistent with the optimized parameters' % (component))
				# break
	print("*************************************************")
	if(dependency_broken):
		print('Error: component %s is not consistent with the optimized parameters' % (components))
	else:
		print('dependency maintainted')
	print("*************************************************")

# print out results
	# performanceDetails(opt_params, layer_configs)
	# performanceUsingDebugModel(opt_params, layer_configs, sortedLayerNames))
	latencySolver = performanceUsingSolverModel(opt_params, layer_configs, sortedLayerNames)
	# totalMACs = 0
	# for layer_name in layer_configs:
	# 	layer_config = layer_configs[layer_name]
	# 	layer_macs = (layer_config['LAYER_IN_NUM'] * layer_config['LAYER_OUT_NUM'] * layer_config['LAYER_OUT_H'] * layer_config['LAYER_OUT_W'] * layer_config['LAYER_FILTER_S2'] * layer_config['LAYER_FILTER_S2']) / layer_config['LAYER_STRIDE']
	# 	totalMACs += layer_macs
	# latency = (totalMACs/512)/(250*1e6)
	# print("theoritical opt latency: ", latency)
	# print("opt latency @(%d MHz): " % (opt_params['FRE']), opt_latency)
	opt_time = opt_latency / (opt_params['FRE'] * 1e6)
	opt_fps = 1 / opt_time
	# print("opt cycles: ", opt_latency)
	print("opt latency solver (s) @%dMHz: " % (opt_params['FRE']), latencySolver)
	print("opt latency        (s) @%dMHz: " % (opt_params['FRE']), opt_time)
	# print("DSP efficiency: ", (latency / opt_time)*100, "%")
	print("opt FPS: ", opt_fps)
	opt_BRAM18K_util = opt_BRAM18K / board_info['BRAM18K'] * 100
	opt_DSP_util = opt_DSP / board_info['DSP'] * 100
	print("opt BRAM18K: %d (%d%%)" % (opt_BRAM18K, opt_BRAM18K_util))
	print("opt DSP: %d (%d%%)" % (opt_DSP, opt_DSP_util))
	
	with open('opt_params.json', 'w') as f:
		json.dump(opt_params, f, indent = 2)
		
	wt = opt_params["LAYER_IN_W_T_LIST"]
	ht = opt_params["LAYER_IN_H_T_LIST"]
	nt = opt_params["LAYER_IN_NUM_T_LIST"]
	mt = opt_params["LAYER_OUT_NUM_T_LIST"]
	
	model_out = open("network_out.model", "w")
	for i, line in enumerate(modelLines):
		line = line.strip('\n')
		if i == 0:
			line += 'in_num_t,out_num_t,in_h_t,in_w_t'
		else:
			line += ';' + str(nt[i-1]) + ';' + str(mt[i-1]) + ';' + str(ht[i-1]) + ';' + str(wt[i-1])
		model_out.write(line + '\n')
	
	model_out.close()
	model.close()
	print("*************************************************")
	global_timer_end = time.time()
	print('Total elapsed time (s): %.3f' % (global_timer_end - global_timer_start))
	print("*************************************************")

	return opt_params['FRE'], opt_time, opt_fps, opt_BRAM18K, opt_BRAM18K_util, opt_DSP, opt_DSP_util, elapsed_time

def param_sweep(params_list, config, layer_configs, sortedLayerNames, in_num_depend, out_num_depend, total_layer, systolic_en, solver_en):
	opt_latency = np.inf
	opt_DSP = np.inf
	opt_BRAM18K = np.inf
	opt_params = {}
	solver_fails = 0
	model_fails = 0
	for i in range(len(params_list)):
		params_t = params_list[i]
		params = dict(params_t)
		IN_NUM_T = params['LAYER_IN_NUM_T']
		IN_H_T = params['LAYER_IN_H_T']
		IN_W_T = params['LAYER_IN_W_T']
		OUT_H_T = params['LAYER_OUT_H_T']
		OUT_W_T = params['LAYER_OUT_W_T']
		SIMD_LANE = params['SIMD_FACTOR']
		
		###################################################
		## Search through different systolic array sizes ##
		###################################################
		# Turn it off if you want to go with a predefined systolic array size
		for SA_ROWS in (list(filter(lambda x : IN_NUM_T % x == 0, range(1, IN_NUM_T + 1))) if systolic_en else [8]):
			for SA_COLS in (list(filter(lambda x : IN_W_T % x == 0, range(1, IN_W_T + 1))) if systolic_en else [8]):
				for SA_SIMD_LANE in (list(filter(lambda x : SIMD_LANE % x == 0, range(1, SIMD_LANE + 1))) if systolic_en else [8]):
					params['LAYER_IN_H_T'] = IN_H_T
					params['LAYER_IN_W_T'] = IN_W_T
					params['LAYER_OUT_H_T'] = OUT_H_T
					params['LAYER_OUT_W_T'] = OUT_W_T
					params['LAYER_IN_NUM_T'] = IN_NUM_T
					params['LAYER_OUT_NUM_T'] = IN_NUM_T
					params['SIMD_LANE'] = SIMD_LANE
					params['SA_ROWS'] = SA_ROWS
					params['SA_COLS'] = SA_COLS
					params['SA_SIMD_LANE'] = SA_SIMD_LANE
					# resource estimation
					DSP, BRAM18K = res_est(params)
					# hw pruning
					if IN_W_T % SA_COLS != 0 or IN_NUM_T % SA_SIMD_LANE != 0 or IN_NUM_T % SA_ROWS != 0:
						continue
					# resource pruning
					if DSP > config['BOARD']['DSP_THRES'] * config['BOARD']['DSP']:
						continue
					if BRAM18K > config['BOARD']['BRAM18K_THRES'] * config['BOARD']['BRAM18K']:
						continue

					# frequency adjustment
					# as the resource utilization will affect the frequency, we will adjust freqeuncy here using a simple step-wise function
					if DSP / config['BOARD']['DSP'] > 0.8 or BRAM18K / config['BOARD']['BRAM18K'] > 0.8:
						params['FRE'] = 180
					else:
						params['FRE'] = 219

#          if (IN_NUM_T == 32) and (IN_W_T == 2) and (SIMD_LANE == 2):
#            if (SA_ROWS == 1) and (SA_COLS == 1) and ((SA_SIMD_LANE == 2) or (SA_SIMD_LANE == 1)):
#              print(params)

					SA_CONFIG = [SA_ROWS, SA_COLS, SA_SIMD_LANE]
					# latency estimation
					# count += 1
					latency = np.inf
					if solver_en:
						try:
							# print(in_num_depend, out_num_depend)
							latency, params = model_solver(params, layer_configs, sortedLayerNames, in_num_depend, out_num_depend, total_layer, SA_CONFIG)
						except:
							solver_fails += 1
							continue
						
					else:
						latency, params = model_latency_est(params, layer_configs, sortedLayerNames, config['DYNAMIC_TILING_LEVEL'], total_layer, SA_CONFIG)
						if latency == np.inf:
							model_fails += 1
					# exit()
#          if (IN_NUM_T == 32) and (IN_W_T == 2) and (SIMD_LANE == 2):
#            print(latency, SA_ROWS, SA_COLS, SA_SIMD_LANE)
#            if (SA_ROWS == 1) and (SA_COLS == 1) and ((SA_SIMD_LANE == 2) or (SA_SIMD_LANE == 1)):
#              print(params)

					cur_fps = 219 * 1e6 * (1 / latency)
					opt_fps = 219 * 1e6 * (1 / opt_latency)
					
#          print(cur_fps)
					if cur_fps - opt_fps >= 0.05:
#            print("updated FPS (%.2f -> %.2f)" % (opt_fps, cur_fps))
#            if IN_NUM_T == 32 and IN_H_T == 2 and SIMD_LANE == 2:
#                print(params)
						opt_latency = latency
						opt_DSP = DSP
						opt_BRAM18K = BRAM18K
						opt_params['LAYER_IN_H_T'] = params['LAYER_IN_H_T']
						opt_params['LAYER_IN_W_T'] = params['LAYER_IN_W_T']
						opt_params['LAYER_OUT_H_T'] = params['LAYER_OUT_H_T']
						opt_params['LAYER_OUT_W_T'] = params['LAYER_OUT_W_T']
						opt_params['LAYER_IN_NUM_T'] = params['LAYER_IN_NUM_T']
						opt_params['LAYER_OUT_NUM_T'] = params['LAYER_OUT_NUM_T']
						opt_params['SIMD_LANE'] = params['SIMD_LANE']
						opt_params['SA_ROWS'] = params['SA_ROWS']
						opt_params['SA_COLS'] = params['SA_COLS']
						opt_params['SA_SIMD_LANE'] = params['SA_SIMD_LANE']
						opt_params['LAYER_IN_NUM_T_LIST'] = list(params['LAYER_IN_NUM_T_LIST'])
						opt_params['LAYER_OUT_NUM_T_LIST'] = list(params['LAYER_OUT_NUM_T_LIST'])
						opt_params['LAYER_IN_H_T_LIST'] = list(params['LAYER_IN_H_T_LIST'])
						opt_params['LAYER_IN_W_T_LIST'] = list(params['LAYER_IN_W_T_LIST'])
						opt_params['FRE'] = params['FRE']
	# print(count)
	res = {}
	res['opt_latency'] = opt_latency
	res['opt_DSP'] = opt_DSP
	res['opt_BRAM18K'] = opt_BRAM18K
	res['opt_params'] = opt_params
	res['solver_fails'] = solver_fails
	res['model_fails'] = model_fails
	
	
	return res

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Design space exploration.')
	"""
		Pass the following command line arguments or change the default value
		
			-m         : The generated file from protobuf_translation
			-i         : The name of the json file containing format of the image
			-b         : The name of the json file containing the number of resources of the target FPGA board
			--parallel : (True/False) Specify if you want to run the multi-threaded version of this code or not
			--systolic : (True/False) Specify whether you want to search for the shape of systolic array or not
			-dt        : The dynamic tiling level you want to have (0: Disabled
																															1: Only number of channels will be dynamic
																															2: All the dimensions will be dynamic)
	"""
	

	parser.add_argument('-m', '--model', metavar='MODEL', default='./network_s.model', help='model description', dest='model')
	parser.add_argument('-i', '--input-config', metavar='INPUT_CONFIG', default='./input.json', help='input configuration', dest='input_config')
	parser.add_argument('-b', '--board', metavar='BOARD', default='./vu9p.json', help='FPGA board information', dest='board')
	parser.add_argument('--parallel', help='multi-threading parallelization', default=True, action='store_false', dest='parallel')
	parser.add_argument('--systolic', help='systolic-array-search', default=True, action='store_false', dest='systolic')
	parser.add_argument('-dt', '--dynamic-tiling', metavar='DYNAMIC_TILING', help='dynamic tiling level (0:disabled, 1:channel 2:height/width)', required=False, type=int, default=2, dest='dynamic_tiling')
	parser.add_argument('--solver', help='use solver for model latency estimation', default=False, action='store_true', dest='solver')
	parser.add_argument('-o', '--output_dir', type=str, default='./', help='directory to output results')


	args = parser.parse_args()
	print("parallel:", args.parallel)
	print("sa:", args.systolic)
	if not os.path.isdir(args.output_dir):
		os.mkdir(args.output_dir)
	results = run(args.model, args.input_config, args.board, args.parallel, args.systolic, args.dynamic_tiling, args.solver, args.output_dir)
	#freq, time, fps, bram, bram_util, dsp, dsp_util, dse_time
	results = np.array(results)
	np.save(os.path.join(args.output_dir, "results.npy"), results)
