#pragma once
// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#include <float.h>
#include <math.h>
#include <cctype>
#include <vector>
#include "zero_finder.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Conversion from Mean Anomaly to Eccentric Anomaly via Kepler's equation
double Mean2Eccentric(const double, const double);

void Conversion(const double *, double *, double *, const double);

double norm(const double *, const double *);

double norm2(const double *);

void vett(const double *, const double *, double *);

double asinh(double);

double acosh(double);

double tofabn(const double &, const double &, const double &);

void vers(const double *, double *);

double x2tof(const double &, const double &, const double &, const int);
