// Matlab style plot functions for OpenCV by Changbo (zoccob@gmail).

#include "cvplot/cvplot.h"
#include "caffe2/util/window.h"

namespace CvPlot
{

//  use anonymous namespace to hide global variables.
namespace
{
	const CvScalar CV_BLACK = CV_RGB(0,0,0);
	const CvScalar CV_WHITE = CV_RGB(255,255,255);
	const CvScalar CV_GREY = CV_RGB(150,150,150);

	PlotManager pm;
}


Series::Series(void)
{
	data = NULL;
	label = "";
	Clear();
}

Series::Series(const Series& s):count(s.count), label(s.label), auto_color(s.auto_color), color(s.color)
{
	data = new float[count];
	memcpy(data, s.data, count * sizeof(float));
}


Series::~Series(void)
{
	Clear();
}

void Series::Clear(void)
{
	if (data != NULL)
		delete [] data;
	data = NULL;

	count = 0;
	color = CV_BLACK;
	auto_color = true;
	label = "";
}

void Series::SetData(int n, float *p)
{
	Clear();

	count = n;
	data = p;
}

void Series::SetColor(int R, int G, int B, bool auto_color)
{
	R = R > 0 ? R : 0;
	G = G > 0 ? G : 0;
	B = B > 0 ? B : 0;
	color = CV_RGB(R, G, B);
	SetColor(color, auto_color);
}

void Series::SetColor(CvScalar color, bool auto_color)
{
	this->color = color;
	this->auto_color = auto_color;
}

Figure::Figure(const std::string name)
{
	figure_name = name;

	custom_range_y = false;
	custom_range_x = false;
	backgroud_color = CV_WHITE;
	axis_color = CV_BLACK;
	text_color = CV_BLACK;

	figure_size = cvSize(600, 200);
	border_size = 50;
	figure_type = Line;

	plots.reserve(10);
}

Figure::~Figure(void)
{
}

std::string Figure::GetFigureName(void)
{
	return figure_name;
}

Series* Figure::Add(const Series &s)
{
	plots.push_back(s);
	return &(plots.back());
}

void Figure::Clear()
{
      plots.clear();
}

void Figure::Size(CvSize size)
{
      figure_size = size;
}

void Figure::Type(FigureType type)
{
      figure_type = type;
}

void Figure::Initialize()
{
	color_index = 0;

	// size of the figure
	if (figure_size.width <= border_size * 2 + 100)
		figure_size.width = border_size * 2 + 100;
	if (figure_size.height <= border_size * 2 + 200)
		figure_size.height = border_size * 2 + 200;

	y_max = (figure_type == Histogram ? 0 : FLT_MIN);
	y_min = (figure_type == Histogram ? 0 : FLT_MAX);

	x_max = 0;
	x_min = 0;

	// find maximum/minimum of axes
	for (std::vector<Series>::iterator iter = plots.begin();
		iter != plots.end();
		iter++)
	{
		float *p = iter->data;
		for (int i=0; i < iter->count; i++)
		{
			float v = p[i];
			if (v < y_min)
				y_min = v;
			if (v > y_max)
				y_max = v;
		}

		int count = (figure_type == Histogram ? iter->count - 1 : iter->count);
		if (x_max < count)
			x_max = count;
	}

	// calculate zoom scale
	// set to 2 if y range is too small
	float y_range = y_max - y_min;
	float eps = 1e-9f;
	if (y_range <= eps)
	{
		y_range = 1;
		y_min = y_max / 2;
		y_max = y_max * 3 / 2;
	}

	x_scale = 1.0f;
	if (x_max - x_min > 0)
		x_scale = (float)(figure_size.width - border_size * 2) / (x_max - x_min);

	y_scale = (float)(figure_size.height - border_size * 2) / y_range;
}

CvScalar Figure::GetAutoColor()
{
	// 	change color for each curve.
	CvScalar col;

	switch (color_index)
	{
	case 1:
		col = CV_RGB(60,60,255);	// light-blue
		break;
	case 2:
		col = CV_RGB(60,255,60);	// light-green
		break;
	case 3:
		col = CV_RGB(255,60,40);	// light-red
		break;
	case 4:
		col = CV_RGB(0,210,210);	// blue-green
		break;
	case 5:
		col = CV_RGB(180,210,0);	// red-green
		break;
	case 6:
		col = CV_RGB(210,0,180);	// red-blue
		break;
	case 7:
		col = CV_RGB(0,0,185);		// dark-blue
		break;
	case 8:
		col = CV_RGB(0,185,0);		// dark-green
		break;
	case 9:
		col = CV_RGB(185,0,0);		// dark-red
		break;
	default:
		col =  CV_RGB(200,200,200);	// grey
		color_index = 0;
	}
	color_index++;
	return col;
}

void Figure::DrawAxis(IplImage *output)
{
	int bs = border_size;
	int h = figure_size.height;
	int w = figure_size.width;

	// size of graph
	int gh = h - bs * 2;
	int gw = w - bs * 2;

	// draw the horizontal and vertical axis
	// let x, y axies cross at zero if possible.
	float y_ref = y_min;
	if ((y_max > 0) && (y_min <= 0))
		y_ref = 0;

	int x_axis_pos = h - bs - cvRound((y_ref - y_min) * y_scale);

	cvLine(output, cvPoint(bs,     x_axis_pos),
		           cvPoint(w - bs, x_axis_pos),
				   axis_color);
	cvLine(output, cvPoint(bs, h - bs),
		           cvPoint(bs, h - bs - gh),
				   axis_color);

	// Write the scale of the y axis
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,0.7, 0,1,CV_AA);

	int chw = 6, chh = 10;
	char text[16];

	// y max
	if ((y_max - y_ref) > 0.05 * (y_max - y_min))
	{
		snprintf(text, sizeof(text)-1, "%.1f", y_max);
		cvPutText(output, text, cvPoint(bs / 5, bs - chh / 2), &font, text_color);
	}
	// y min
	if ((y_ref - y_min) > 0.05 * (y_max - y_min))
	{
		snprintf(text, sizeof(text)-1, "%.1f", y_min);
		cvPutText(output, text, cvPoint(bs / 5, h - bs + chh), &font, text_color);
	}

	// x axis
	snprintf(text, sizeof(text)-1, "%.1f", y_ref);
	cvPutText(output, text, cvPoint(bs / 5, x_axis_pos + chh / 2), &font, text_color);

	// Write the scale of the x axis
	snprintf(text, sizeof(text)-1, "%.0f", x_max );
	cvPutText(output, text, cvPoint(w - bs - strlen(text) * chw, x_axis_pos + chh),
		      &font, text_color);

	// x min
	snprintf(text, sizeof(text)-1, "%.0f", x_min );
	cvPutText(output, text, cvPoint(bs, x_axis_pos + chh),
		      &font, text_color);


}
void Figure::DrawPlots(IplImage *output)
{
	int bs = border_size;
	int h = figure_size.height;
	int w = figure_size.width;

	// draw the curves
	for (std::vector<Series>::iterator iter = plots.begin();
		iter != plots.end();
		iter++)
	{
		float *p = iter->data;

		// automatically change curve color
		if (iter->auto_color == true)
			iter->SetColor(GetAutoColor());

		switch(figure_type){
		case Line: {
				CvPoint prev_point;
				for (int i=0; i<iter->count; i++)
				{
					int y = cvRound(( p[i] - y_min) * y_scale);
					int x = cvRound((   i  - x_min) * x_scale);
					CvPoint next_point = cvPoint(bs + x, h - (bs + y));
					cvCircle(output, next_point, 1, iter->color, 1);

					// draw a line between two points
					if (i >= 1)
						cvLine(output, prev_point, next_point, iter->color, 1, CV_AA);
					prev_point = next_point;
				}
			}
			break;
		case Histogram: {
				for (int i=0; i<iter->count; i++)
				{
					int y = cvRound(( p[i] - y_min) * y_scale);
					int x = cvRound((   i  - x_min) * x_scale);
					CvPoint base_point = cvPoint(bs + x + 5, h - (bs));
					CvPoint next_point = cvPoint(bs + x - 5, h - (bs + y));
					cvCircle(output, next_point, 1, iter->color, 1);

					// draw a line between two points
					cvRectangle(output, base_point, next_point, iter->color, -1, CV_AA);
				}
			}
			break;
		}

	}

}

void Figure::DrawLabels(IplImage *output, int posx, int posy)
{

	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,1.0, 0,1,CV_AA);

	// character size
	int chw = 6, chh = 8;

	for (std::vector<Series>::iterator iter = plots.begin();
		iter != plots.end();
		iter++)
	{
		std::string lbl = iter->label;
		// draw label if one is available
		if (lbl.length() > 0)
		{
			cvLine(output, cvPoint(posx, posy - chh / 2), cvPoint(posx + 15, posy - chh / 2),
				   iter->color, 2, CV_AA);

			cvPutText(output, lbl.c_str(), cvPoint(posx + 20, posy),
					  &font, iter->color);

			posy += int(chh * 1.5);
		}
	}

}

// whole process of draw a figure.
void Figure::Show()
{
	Initialize();

	IplImage *output = cvCreateImage(figure_size, IPL_DEPTH_8U, 3);
	cvSet(output, backgroud_color, 0);

	DrawAxis(output);

	DrawPlots(output);

	DrawLabels(output, figure_size.width - 100, 10);

	caffe2::imshow(figure_name.c_str(), cv::cvarrToMat(output));
	cvWaitKey(1);
	cvReleaseImage(&output);

}



bool PlotManager::HasFigure(std::string wnd)
{
	return false;
}

// search a named window, return null if not found.
Figure* PlotManager::FindFigure(std::string wnd)
{
	for(std::vector<Figure>::iterator iter = figure_list.begin();
		iter != figure_list.end();
		iter++)
	{
		if (iter->GetFigureName() == wnd)
			return &(*iter);
	}
	return NULL;
}

// search a named window, create figure if not found.
Figure* PlotManager::GetFigure(std::string wnd)
{
	auto active_figure = FindFigure(wnd);
	if ( active_figure == NULL)
	{
		Figure new_figure(wnd);
		figure_list.push_back(new_figure);
		active_figure = FindFigure(wnd);
		if (active_figure == NULL)
			exit(-1);
	}
	return active_figure;
}

// plot a new curve, if a figure of the specified figure name already exists,
// the curve will be plot on that figure; if not, a new figure will be created.
void PlotManager::Plot(const std::string figure_name, const float *p, int count, int step,
					   int R, int G, int B)
{
	if (count < 1)
		return;

	if (step <= 0)
		step = 1;

	// copy data and create a series format.
	float *data_copy = new float[count];

	for (int i = 0; i < count; i++)
		*(data_copy + i) = *(p + step * i);

	Series s;
	s.SetData(count, data_copy);

	if ((R > 0) || (G > 0) || (B > 0))
		s.SetColor(R, G, B, false);

	// search the named window and create one if none was found
	active_figure = FindFigure(figure_name);
	if ( active_figure == NULL)
	{
		Figure new_figure(figure_name);
		figure_list.push_back(new_figure);
		active_figure = FindFigure(figure_name);
		if (active_figure == NULL)
			exit(-1);
	}

	active_series = active_figure->Add(s);
	active_figure->Show();

}

// add a label to the most recently added curve
void PlotManager::Label(std::string lbl)
{
	if((active_series!=NULL) && (active_figure != NULL))
	{
		active_series->label = lbl;
		active_figure->Show();
	}
}

// plot a new curve, if a figure of the specified figure name already exists,
// the curve will be plot on that figure; if not, a new figure will be created.
// static method
template<typename T>
void plot(const std::string figure_name, const T* p, int count, int step,
		  int R, int G, int B)
{
	if (step <= 0)
		step = 1;

	float  *data_copy = new float[count * step];

	float   *dst = data_copy;
	const T *src = p;

	for (int i = 0; i < count * step; i++)
	{
		*dst = (float)(*src);
		dst++;
		src++;
	}

	pm.Plot(figure_name, data_copy, count, step, R, G, B);

	delete [] data_copy;
}

// delete all plots on a specified figure
void clear(const std::string figure_name)
{
	Figure *fig = pm.FindFigure(figure_name);
	if (fig != NULL)
	{
		fig->Clear();
	}

}

// adjust the size of the plot
void size(const std::string figure_name, CvSize size)
{
	Figure *fig = pm.GetFigure(figure_name);
	fig->Size(size);
}

// adjust the size of the plot
void type(const std::string figure_name, FigureType type)
{
	Figure *fig = pm.GetFigure(figure_name);
	fig->Type(type);
}

// add a label to the most recently added curve
// static method
void label(std::string lbl)
{
	pm.Label(lbl);
}


template
void plot(const std::string figure_name, const unsigned char* p, int count, int step,
		  int R, int G, int B);

template
void plot(const std::string figure_name, const int* p, int count, int step,
		  int R, int G, int B);

template
void plot(const std::string figure_name, const short* p, int count, int step,
		  int R, int G, int B);

template
void plot(const std::string figure_name, const float* p, int count, int step,
		  int R, int G, int B);

};
