#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aux.py
Created 11.12.2014

Goal: Provides auxiliary functions useful for reservoir computing

"""

from matplotlib import pyplot as plt
import os.path


def kindofvector(vec, label=None):
	label = "" if label is None else "{}: ".format(label)
	shp = vec.shape
	if len(shp) == 1:
		print(label + 'vector of length %d' % shp[0])
	else:
		if shp[0] == 1:
			print(label + 'a long row (with %d columns)' % shp[1])
		elif shp[1] == 1:
			print(label + 'a long column (with %d rows)' % shp[0])
		elif shp[0] > shp[1]:
			print(label + 'a tall rectangle matrix (%d x %d)' % shp)
		elif shp[0] < shp[1]:
			print(label + 'a wide rectangle matrix (%d x %d)' % shp)
		elif shp[0] == shp[1]:
			print(label + 'a square matrix (%d x %d)' % shp)
		else:
			print(label + 'an alien matrix of shape: %s' % str(shp))

def try_save_fig(fname="figure", ext="png" , **kwargs):
	i = 0
	fullfile = fname + str(i) + "." + ext
	while os.path.exists(fullfile):
		i += 1
		fullfile = fname + str(i) + "." + ext

	plt.savefig(fullfile, **kwargs)
	print("saved figure as '{}'".format(fullfile))

def readable_seconds(total_seconds):
	h = int(total_seconds / 3600)
	m = int((total_seconds - h * 3600) / 60)
	s = int(total_seconds) - h * 3600 - m * 60

	if h == 0 and m == 0:
		return "{} s".format(s)
	elif h == 0:
		return "{} m, {} s".format(m, s)
	else:
		return "{} h, {} m, {} s".format(h,m,s)

