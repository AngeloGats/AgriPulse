# Leaf-image dataset (Artificial Neural Network)

A subset of the public **PlantVillage** tomato-leaf photos, downsized so the whole
project stays small and groupmates need no large download. These are real photos, taken
on a plain background, so a model that scores well here may do worse on field photos.

Contents (committed):

- `<class>/<class>_NNNN.jpg`: 900 images per class, shrunk to 64x64
- `train.csv` and `test.csv`: stratified 80/20 split (3,600 train, 900 test) with `path,label`
- `schema.json`: classes, display names and image size

Classes: `healthy`, `early_blight`, `late_blight`, `bacterial_spot`, `yellow_leaf_curl`.

## Regenerate (optional)

Only needed if you want a different subset. It needs the raw download, which is **not**
committed: put PlantVillage in `data/ANN/PlantVillage/<folder>/*.JPG` (for example
`Tomato_healthy`, `Tomato_Early_blight`), then from the project root:

```powershell
.\.venv\Scripts\python.exe ml\prepare_ann_images.py
.\.venv\Scripts\python.exe ml\train_ann.py
```

The random choice uses a fixed seed (42), so the same raw download gives the same subset.

## Model

`trained_models/ann/leaf_cnn.keras` is a small convolutional neural network (CNN): four
convolution layers (32, 64, 96 and 128 filters, each with batch normalization and ReLU,
the first three followed by max pooling), global average pooling, dropout, a 64-neuron
dense layer and 5 softmax outputs (about 195,000 weights). It reads the 64x64 colour image
directly. During training the photos are randomly flipped, rotated, zoomed and brightened,
and training stops early when the validation score stops improving. It scores about 92% on
the unseen test photos (chance is 20%), against about 85% for the earlier plain
pixel-by-pixel network trained on the same photos. Early and bacterial spot are confused
most often. Retraining takes about 10 minutes on a CPU and the result is not bit-for-bit
identical between machines.
