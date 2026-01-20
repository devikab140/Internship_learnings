from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open("C:/Users/devik/Downloads/star_image.jpg").convert("L")
img.show()
mask = np.array(img)
print(mask)
reader = PdfReader("C:/Users/devik/Downloads/harrypotter.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()


wordcloud = WordCloud(
    background_color="white",
    mask=mask,
    colormap="cool",
    max_words=300
).generate(text)

plt.figure(figsize=(8,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#since not lemmatized -- words may repeat