#pragma once
#include "config.h"

struct State {
	State();

	/* shell/keyboard activated commands */
	void pan_right();
	void pan_left();
	void pan_down();
	void pan_up();
	void zoom_in();
	void zoom_out();
	void set_red(double a);
	void set_green(double a);
	void set_blue(double a);
	
	/* update state variables and redraw screen */
	void update();

	/* translate pixel coordinates to actual coordinates */
	double ax(double px);
	double ay(double py);

	/* zoom on fractal */
	double zoom;

	/* center of screen */ 
	double cx;
	double cy;

	/* min. x and y coordinates */
	double mx;
	double my;

	/* screen step size for x and y */
	double sx;
	double sy;

	/* global color setting */
	double R;
	double G;
	double B;

	/* flag set when rendering */ 
	bool render;
};

/* shared global state variable */
extern State state;


