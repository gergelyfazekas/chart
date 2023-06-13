import keras_ocr
import matplotlib.pyplot as plt
import itertools

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()
img = [keras_ocr.tools.read("fig.png")]
prediction_groups = pipeline.recognize(img)

num_list = []
for pred in prediction_groups[0]:
    try:
        num_list.append(int(pred[0]))
    except:
        print("pass")
        pass

distances = sorted([abs(item[0]-item[1]) for item in list(itertools.combinations(num_list, 2))])
distance = distances[1]
print("distance: ", distance)



fig, ax = plt.subplots()
keras_ocr.tools.drawAnnotations(image=img[0], predictions=prediction_groups[0], ax=ax)

plt.imshow(img[0])
plt.show()