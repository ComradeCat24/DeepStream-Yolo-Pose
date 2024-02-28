/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __DEEPSTREAM_H__  // Header guard to prevent multiple inclusions
#define __DEEPSTREAM_H__

// Macro definitions for finding maximum and minimum values
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Standard C library headers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// GLib and GStreamer headers
#include <glib.h>
#include <gst/gst.h>

// NVIDIA DeepStream headers
#include <nvdsgstutils.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

// Custom module headers
#include "modules/interrupt.h"

// Static global variables
static gchar **SOURCE = NULL;       // Pointer to pointer of gchar (string) for source
static gchar *INPUT_FILE = NULL;     // Pointer to gchar (string) for input file
static gchar *CONFIG_INFER = NULL;   // Pointer to gchar (string) for inference configuration file
static guint STREAMMUX_WIDTH = 1280; // Width for stream multiplexer
static guint STREAMMUX_HEIGHT = 720;// Height for stream multiplexer
static guint GPU_ID = 0;             // GPU ID for CUDA operations

// Array defining connections for a human skeleton
const gint skeleton[][2] = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, 
    {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, 
    {2, 4}, {3, 5}, {4, 6}, {5, 7}
};

#endif  // End of header guard
