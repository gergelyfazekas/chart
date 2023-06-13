#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[2]:


"""chart = Chart()
chart.create()
chart.get_ylabel_bbox()
chart.get_yaxis_bbox()
chart.get_bars_bbox()"""


# In[87]:


data = {1:8,2:6,3:10,4:5}


# In[4]:


fig, ax = plt.subplots()
ax.bar(x=list(data.keys()), height=list(data.values()))


# In[5]:


label_position = ax.get_yaxis().get_label_position()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Render the figure onto a canvas
canvas = FigureCanvasAgg(fig)
canvas.draw()

# Get the pixel data from the canvas as a NumPy array
width, height = fig.get_size_inches() * fig.get_dpi()
image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)


# In[63]:


text = ax.get_ymajorticklabels()[1]

bounding_boxes = [text.get_tightbbox(fig.canvas.get_renderer())]
buffer = bounding_boxes[0].width
    

b = ax.get_position()
x = int((b.x0*width)-buffer)
y = int((b.y0*height)-buffer)
w = int((b.x0*width)+buffer)
h = int((b.y0+b.height)*height)
rect = cv2.rectangle(image_array, (x, y), (w, h), (255, 0, 0), 2)


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(rect[y:h, x:w])
plt.axis('off')


# In[45]:


text = ax.get_ymajorticklabels()[1]

bounding_boxes = [text.get_tightbbox(fig.canvas.get_renderer())]
for box in bounding_boxes:
    rect = cv2.rectangle(rect, (int(box.x0), int(box.y0)), (int(box.x0+box.width), int(box.y0+box.height)), (0, 255, 100), 2)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(rect)
plt.axis('off')


# In[134]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Create a figure
fig, ax = plt.subplots()
ax.bar(x=list(data.keys()), height=list(data.values()))

# Render the figure onto a canvas
canvas = FigureCanvasAgg(fig)
canvas.draw()

# Get the pixel data from the canvas as a NumPy array
width, height = fig.get_size_inches() * fig.get_dpi()
image_array = np.asarray(canvas.renderer.buffer_rgba())

y_labels = ax.get_yticklabels()

# Get the bounding boxes around the y-axis labels
renderer = canvas.get_renderer()
bounding_boxes = [label.get_window_extent() for label in y_labels]

# Draw rectangles around the bounding boxes
for box in bounding_boxes:
    y_plt = height - box.y0 - box.height  # Invert the y-coordinate
    rect = cv2.rectangle(image_array, (int(box.x0), int(y_plt)), (int(box.x0+box.width), int(y_plt+box.height)),
                         (0, 0, 0, 1000), 2)

plt.imshow(image_array)
plt.axis('off')
plt.show()


# In[147]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Create a figure
fig, ax = plt.subplots()
ax.bar(x=list(data.keys()), height=list(data.values()))

# Render the figure onto a canvas
canvas = FigureCanvasAgg(fig)
canvas.draw()

# Get the pixel data from the canvas as a NumPy array
width, height = fig.get_size_inches() * fig.get_dpi()
image_array = np.asarray(canvas.renderer.buffer_rgba())

bounding_boxes=[ax.get_yaxis().get_tightbbox(renderer)]
for box in bounding_boxes:
    y_plt = height - box.y0 - box.height  # Invert the y-coordinate
    rect = cv2.rectangle(image_array, (int(box.x0), int(y_plt)), (int(box.x0+box.width), int(y_plt+box.height)),
                         (0, 0, 0, 1000), 2)
plt.imshow(image_array)
plt.axis('off')
plt.show()


# In[162]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Create a figure
fig, ax = plt.subplots()
ax.bar(x=list(data.keys()), height=list(data.values()))

# Render the figure onto a canvas
canvas = FigureCanvasAgg(fig)
canvas.draw()

# Get the pixel data from the canvas as a NumPy array
width, height = fig.get_size_inches() * fig.get_dpi()
image_array = np.asarray(canvas.renderer.buffer_rgba())

bounding_boxes=[item.get_tightbbox(renderer) for item in ax.get_yticklines()]
for box in bounding_boxes:
    y_plt = height - box.y0 - box.height  # Invert the y-coordinate
    rect = cv2.rectangle(image_array, (int(box.x0), int(y_plt)), (int(box.x0+box.width), int(y_plt+box.height)),
                         (0, 0, 0, 1000), 2)
plt.imshow(image_array)
plt.axis('off')
plt.show()

