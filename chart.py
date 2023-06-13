import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import random

class Rectangle:
    def __init__(self, id_, left, right, top, bottom, class_=0):
        self.id_ = id_
        self.class_ = class_
        self.top = 1-top
        self.bottom = 1-bottom
        self.left = 1-left
        self.right = 1-right
        self.width = abs(self.right - self.left)
        self.height = abs(self.top - self.bottom)
        self.x_center = 1-(self.left + (self.width/2))
        self.y_center = 1-(self.bottom + (self.height/2))
        
    @classmethod
    def from_xywh(cls, id_, x, y, w, h, class_=0):
        left = x-(w/2)
        right = x+(w/2)
        top = y+(h/2)
        bottom = y-(h/2)
        return cls(id_, left, right, top, bottom, class_)
    

class Label:
    def __init__(self, path):
        self.path = path
    
    def read(self):
        """reads the labels from path and parses them to be compatible with numpy"""
        result = []

        with open(self.path, "r") as f:
            labels_all = f.readlines()
        
        for labels in labels_all:
            labels = labels.split("\n")
            try:
                labels.remove("")
            except ValueError:
                pass
            
            labels = labels[0].split(" ")
            labels = [float(item) for item in labels]
            result.append(labels)
        return result
    

class Image:
    def __init__(self, image_path, bar_label_path=None):
        self.image_path = image_path
        self.bar_label_path = bar_label_path
        self.numpy = plt.imread(self.image_path)
        self.bars_bbox = None
        self.bar_labels = None        

        if self.bar_label_path is not None:
            # Read labels
            self.bar_labels = Label(path=self.bar_label_path).read()
            # Convert into rects
            rects_with_ids = list(enumerate(self.bar_labels))
            self.bars_bbox = [Rectangle.from_xywh(id_=item[0], x=item[1][1], y=item[1][2], w=item[1][3], h=item[1][4], class_=item[1][0]) for item in rects_with_ids]
            # Convert rects to pixels (has to be rounded to be able to use it as a numpy slice)
            self.bars_bbox_in_pixels = [Rectangle(id_=bar_bbox.id_,
                                    left=self.convert_to_pixels(bar_bbox.left, horizontal=True),
                                    right=self.convert_to_pixels(bar_bbox.right, horizontal=True),
                                    top=self.convert_to_pixels(bar_bbox.top, horizontal=False),
                                    bottom=self.convert_to_pixels(bar_bbox.bottom, horizontal=False))
                                    for bar_bbox in self.bars_bbox]
        
    def convert_to_pixels(self, number: float, horizontal: bool):
        """Converts a number between 0 and 1 into a pixel value based on the shape of the image"""
        direction = 1 if horizontal else 0
        return round(number*self.numpy.shape[direction])
            

    def show_bars_bbox(self, color=(10,0,0,1), line_width=1, fill=False):
        if self.bars_bbox is not None:
            self.numpy_with_bars_bbox = self.numpy.copy()

            if fill:
                for bar_bbox in self.bars_bbox_in_pixels:
                    self.numpy_with_bars_bbox[bar_bbox.bottom:bar_bbox.top, bar_bbox.left:bar_bbox.right] = color
            else:
                for bar_bbox in self.bars_bbox_in_pixels:
                    self.numpy_with_bars_bbox[bar_bbox.bottom:bar_bbox.top, bar_bbox.left-line_width:bar_bbox.left+line_width] = color
                    self.numpy_with_bars_bbox[bar_bbox.bottom:bar_bbox.top, bar_bbox.right-line_width:bar_bbox.right+line_width] = color
                    self.numpy_with_bars_bbox[bar_bbox.top-line_width:bar_bbox.top+line_width, bar_bbox.left:bar_bbox.right] = color
                    self.numpy_with_bars_bbox[bar_bbox.bottom-line_width:bar_bbox.bottom+line_width, bar_bbox.left:bar_bbox.right] = color
        else:
            raise AttributeError("Give a valid label_path at init or specify the rects argument here.")
        
        plt.imshow(self.numpy_with_bars_bbox)
        plt.show()


class Chart:
    def __init__(self, chart_id, random_state=False):
        self.random_state = random_state
        self.chart_id = chart_id
        self.create()
        print(f"Created chart: {self.chart_id}")

    @staticmethod
    def get_wiki():
        url = requests.get("https://en.wikipedia.org/wiki/Special:Random")
        soup = BeautifulSoup(url.content, "html.parser")
        title = soup.find(class_="firstHeading").text
        text = soup.find("div", {"id": "mw-content-text"}).text
        return text, title

    def generate_x(self):
        text, _ = self.get_wiki()
        choice = random.choices(list(set(text[0:500].split(sep=" "))), k=self.num_cols)
        return choice
    
    def generate_y(self):
        center = random.randint(2,20000)
        shape = np.random.uniform(0.2, 0.8, self.num_cols)
        values = center * shape
        return values.round()

    def generate_data(self):
        if self.random_state:
            self.num_cols = random.randint(2, 12)
            x = self.generate_x()
            y = self.generate_y()
            self.data = dict(zip(x, y))
            self.num_cols = len(self.data)
        else:
            self.num_cols = 5
            self.data = {"A": 12, "B": 8, "C": 15, "D": 5, "E": 10}
            self.num_cols = len(self.data)

    def generate_style(self):
        self.style = random.sample(['default','classic','ggplot','seaborn','bmh','dark_background',
                                    'fivethirtyeight','tableau-colorblind10','Solarize_Light2','grayscale'], k=1)
        self.colors = random.sample(['blue','green','red','cyan','magenta','yellow','black',
                                    'white','gray','grey','orange','purple','brown','pink'],
                                    k=self.num_cols)
        self.background_color = np.random.uniform(0.8, 1, 3).tolist()
        self.background_alpha = [random.uniform(0.3, 1)]


    def create(self):
        # Generate the data
        self.generate_data()
        self.generate_style()

        # Set style
        style.use(self.style)

        self.fig, self.ax = plt.subplots()
        self.bars = self.ax.bar(list(self.data.keys()), list(self.data.values()), color=self.colors)
        self.xlabel = self.ax.set_xlabel("Countries")
        self.ylabel = self.ax.set_ylabel("Number of Medals")
        self.title = self.ax.set_title("Olympic Medals by Country")

        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Add a background color
        plt.gca().set_facecolor(tuple(self.background_color + self.background_alpha))

        # Map parameters to the Chart instance
        self.rcParams = plt.rcParams

        # Calculate subplot width and height as a ratio of the whole figure
        self.subplot_height_percent = self.rcParams["figure.subplot.top"] - self.rcParams["figure.subplot.bottom"]
        self.subplot_width_percent = self.rcParams["figure.subplot.right"] - self.rcParams["figure.subplot.left"]

        # Calculate the width and height of the subplot in displayed data units (not figure percent)
        self.subplot_height_displayed_units = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        self.subplot_width_displayed_units = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]

    def get_y_axis_neighborhood(self):
        ticks = self.ax.get_yticklabels()
        pass

    def convert_to_percent(self, n: int, horizontal: bool) -> float:
            """Truns the displayed units into percentage of figure size
            n: number to convert
            horizontal: if this number should be compared to the width or height of the subplot/figure, e.g. column height --> horizontal=False
            """
            # Calculate what is the percentage of n on the display
            if horizontal:
                inside_ratio = n / self.subplot_width_displayed_units
                final_ratio = inside_ratio * self.subplot_width_percent
            else:
                inside_ratio = n / self.subplot_height_displayed_units
                final_ratio = inside_ratio * self.subplot_height_percent
            return final_ratio

    def get_bars_bbox(self):
        # Create an empty list for the rectangles
        self.bars_bbox = []
        paddings_percent = []
        widths_percent = []

        # For every column in the chart
        for i in range(self.num_cols):
            # Calculate the ratio of the bar within the subplot
            try:
                bar_height_percent = self.convert_to_percent(self.bars[i].get_height(), horizontal=False)
                bar_width_percent = self.convert_to_percent(self.bars[i].get_width(), horizontal=True)
            except IndexError:
                print(self.data)
                print(self.num_cols)
                print(self.bars)

            if i==0:
                paddings_percent.append(self.convert_to_percent(abs(self.ax.axis()[0] - self.bars[i].xy[0]), horizontal=True))
            else:
                paddings_percent.append(self.convert_to_percent(abs(self.bars[i].xy[0] - (self.bars[i-1].xy[0] + self.bars[i-1].get_width())),
                                                                        horizontal=True))
            
            widths_percent.append(self.convert_to_percent(self.bars[i].get_width(), horizontal=True))
            if i == 0:
                left_edge_percent = sum(paddings_percent)
            else:
                left_edge_percent = sum(paddings_percent) + sum(widths_percent[:-1])
            right_edge_percent = sum(paddings_percent) + sum(widths_percent)

            # Generate target bounding box (THIS IS THE Y VARIABLE FOR REGRESSION !!!)
            # The top is a ratio in [0,1] but in numpy it has to be inverted to 1-top and then the pixels can be founds as image_array.shape[0]*(1-top)
            top = (1 - self.rcParams["figure.subplot.bottom"]) - bar_height_percent
            bottom = (1 - self.rcParams["figure.subplot.bottom"])

            left =  self.rcParams["figure.subplot.left"] + left_edge_percent
            right = self.rcParams["figure.subplot.left"] + right_edge_percent

            self.bars_bbox.append(Rectangle(id_=i, left=left, right=right, top=top, bottom=bottom))
        return self.bars_bbox

    def get_yaxis_bbox(self):

        self.yaxis_bbox = []
        return self.yaxis_bbox


    def show(self):
        plt.show()

    def verbose(self):
        print("Target", self.target)

    def save(self, name, with_label=True, split="train"):
        """
        creates the yolo style label format in txt
        class_: the class of the object, probably only bar object with class 0
        x,y,width,height: center, center, witdth, height
        split: one of "train", "test", "valid"
        """
        if split.upper() not in ["TRAIN", "TEST", "VALID"]:
            raise ValueError("Invalid split method. Choose one of ['train', 'test', 'valid']")
        n = self.chart_id
        self.fig.savefig(os.path.join(f"/Users/gergelyfazekas/Documents/python_projects/bar_chart/datasets/{split}/images", f"{name}{n}.png"))

        if with_label:
            self.get_bars_bbox()

            l = [[str(item.class_), str(item.x_center + item.width), str(item.y_center), str(item.width), str(item.height)] for item in self.bars_bbox]
            l = [str(" ").join(item) for item in l]
            l = str("\n").join(l)
            
            path = os.path.join(f"/Users/gergelyfazekas/Documents/python_projects/bar_chart/datasets/{split}/labels", f"{name}{n}.txt")
                
            with open(path, "w") as f:
                f.write(l)


def generate_dataset(num_images, random_state=True, split="train"):
    for i in range(num_images):
        chart = Chart(chart_id=i, random_state=random_state)
        chart.save(name="fig", split=split)
        plt.close(chart.fig)



if __name__ == "__main__":

    #generate_dataset(num_images=30, split="valid")
    """ chart = Chart(chart_id=0, random_state=False)
    chart.get_bars_bbox()
    chart.show()
    chart.save(name="plot", split="train") """
    #chart.show()
    #chart.save(name="fig", split="train")
    
    #label = Label(path='/Users/gergelyfazekas/Documents/python_projects/bar_chart/datasets/train/labels/fig1.txt')
    image = Image(image_path='/Users/gergelyfazekas/Documents/python_projects/bar_chart/datasets/train/images/plot0.png',
                  bar_label_path='/Users/gergelyfazekas/Documents/python_projects/bar_chart/datasets/train/labels/plot0.txt')
    image.show_bars_bbox(fill=False)

    



    

    


