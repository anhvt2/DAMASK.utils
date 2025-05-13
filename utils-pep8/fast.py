#!/usr/bin/env python
# coding: utf-8

# # Fast Color Map
# 
# This Jupyter notebook takes a list of points that make up the Fast colormap and derives interpolated tables and figures. Fast comes from ParaView and is documented in the paper "A New Default Colormap for ParaView" by Samsel, Scott, and Moreland.
# 
# This code relies on the [python-colormath](http://python-colormath.readthedocs.org/en/latest/index.html) module. See [its documentation](http://python-colormath.readthedocs.org/en/latest/index.html) for information such as installation instructions. (It can be installed with either pip or macports.)

# In[1]:


from colormath.color_objects import *
from colormath.color_conversions import convert_color


# Mostly because it's habit, I am also using [pandas](http://pandas.pydata.org/) dataframes to organize the data. (Pandas can be installed with macports.)

# In[2]:


import pandas
import numpy


# We will also be using [toyplot](https://toyplot.readthedocs.org) for making visuals (version 0.10.0 or later required). See its documentation for installation instructions.

# In[3]:


import toyplot
import toyplot.svg


# We will also be importing data in JSON files, so load a package for that.

# In[4]:


import json


# Load the detailed color table (256 values) from a JSON file exported from ParaView.

# In[5]:


file_descriptor = open('fast.json', 'r')
raw_color_data = json.load(file_descriptor)[0]


# Run through the "RGBPoints" array, pull out the scalar interpolant and RGB colors, and create a pandas data frame from them.

# In[6]:


scalar = []
rgb_values = []
for i in range(0, len(raw_color_data['RGBPoints']), 4):
    scalar.append(raw_color_data['RGBPoints'][i+0])
    rgb_values.append(sRGBColor(
        raw_color_data['RGBPoints'][i+1],
        raw_color_data['RGBPoints'][i+2],
        raw_color_data['RGBPoints'][i+3]
    ))

data = pandas.DataFrame({'scalar': scalar, 'rgb_values': rgb_values})


# Convert RGB colors to Lab colors.

# In[7]:


data['lab_values'] = data['rgb_values'].apply(lambda rgb: convert_color(rgb, LabColor))


# Make functions that will take a scalar value (in the range of 0 and 1) and return the appropriate RGB triple.

# In[8]:


def color_lookup_sRGBColor(x):
    if x < 0:
        return sRGBColor(0, 0, 0)
    for index in range(0, data.index.size-1):
        low_scalar = data['scalar'][index]
        high_scalar = data['scalar'][index+1]
        if (x > high_scalar):
            continue
        low_lab = data['lab_values'][index]
        high_lab = data['lab_values'][index+1]
        interp = (x-low_scalar)/(high_scalar-low_scalar)
        mid_lab = LabColor(interp*(high_lab.lab_l-low_lab.lab_l) + low_lab.lab_l,
                           interp*(high_lab.lab_a-low_lab.lab_a) + low_lab.lab_a,
                           interp*(high_lab.lab_b-low_lab.lab_b) + low_lab.lab_b,
                           observer=low_lab.observer,
                           illuminant=low_lab.illuminant)
        return convert_color(mid_lab, sRGBColor)
    return sRGBColor(1, 1, 1)

def color_lookup(x):
    return color_lookup_sRGBColor(x).get_value_tuple()

def color_lookup_upscaled(x):
    return color_lookup_sRGBColor(x).get_upscaled_value_tuple()


# Make a long table of colors. This is a very high resolution table of colors that can be easily trimmed down with regular sampling.

# In[9]:


colors_long = pandas.DataFrame({'scalar': numpy.linspace(0.0, 1.0, num=1024)})
colors_long['RGB'] = colors_long['scalar'].apply(color_lookup_upscaled)
colors_long['sRGB'] = colors_long['scalar'].apply(color_lookup)


# The colors are all stored as tuples in a single column. This is convenient for some operations, but not others. Thus, also create separate columns for the three RGB components.

# In[10]:


def unzip_rgb_triple(dataframe, column='RGB'):
    '''Given a dataframe and the name of a column holding an RGB triplet,
    this function creates new separate columns for the R, G, and B values
    with the same name as the original with '_r', '_g', and '_b' appended.'''
    # Creates a data frame with separate columns for the triples in the given column
    unzipped_rgb = pandas.DataFrame(dataframe[column].values.tolist(),
                                    columns=['r', 'g', 'b'])
    # Add the columns to the original data frame
    dataframe[column + '_r'] = unzipped_rgb['r']
    dataframe[column + '_g'] = unzipped_rgb['g']
    dataframe[column + '_b'] = unzipped_rgb['b']

unzip_rgb_triple(colors_long, 'RGB')
unzip_rgb_triple(colors_long, 'sRGB')


# Plot out the color map.

# In[11]:


data


# In[12]:


data.apply(lambda row: '{:1.2f}, {}'.format(row['scalar'], str(row['rgb_values'].get_upscaled_value_tuple())), axis=1)


# In[13]:


colors_palette = toyplot.color.Palette(colors=colors_long['sRGB'].values)
colors_map = toyplot.color.LinearMap(palette=colors_palette,
                                     domain_min=0, domain_max=1)


# In[14]:


canvas = toyplot.Canvas(width=130, height=300)
numberline = canvas.numberline(x1=16, x2=16, y1=-7, y2=7)
numberline.padding = 30
numberline.axis.spine.show = False
numberline.colormap(colors_map,
                    width=30,
                    style={'stroke':'lightgrey'})

control_point_labels =     data.apply(lambda row: '{:1.2f}, {}'.format(
            row['scalar'],
            str(row['rgb_values'].get_upscaled_value_tuple())
            ),
        axis=1,
        )
numberline.axis.ticks.locator =     toyplot.locator.Explicit(locations=data['scalar'],
                             labels=control_point_labels)
numberline.axis.ticks.labels.angle = -90
numberline.axis.ticks.labels.style = {'text-anchor':'start',
                                      'baseline-shift':'0%',
                                      '-toyplot-anchor-shift':'-15px'}


# In[15]:


toyplot.svg.render(canvas, 'fast.svg')


# Create several csv files containing color tables for this color map. We will create color tables of many different sizes from 8 rows to 1024. We also write out one set of csv files for "upscaled" color bytes (values 0-255) and another for floating point numbers (0-1).

# In[16]:


for num_bits in range(3, 11):
    table_length = 2 ** num_bits
    color_table = pandas.DataFrame({'scalar': numpy.linspace(0.0, 1.0, num=table_length)})
    color_table['RGB'] = color_table['scalar'].apply(color_lookup_upscaled)
    unzip_rgb_triple(color_table, 'RGB')
    color_table.to_csv('fast-table-byte-{:04}.csv'.format(table_length),
                       index=False,
                       columns=['scalar', 'RGB_r', 'RGB_g', 'RGB_b'])
    color_table['sRGB'] = color_table['scalar'].apply(color_lookup)
    unzip_rgb_triple(color_table, 'sRGB')
    color_table.to_csv('fast-table-float-{:04}.csv'.format(table_length),
                       index=False,
                       columns=['scalar', 'sRGB_r', 'sRGB_g', 'sRGB_b'],
                       header=['scalar', 'RGB_r', 'RGB_g', 'RGB_b'])


# In[ ]:




