# Low-Light Image Enhancement with a Convolutional Network

A convolutional encoder-decoder that learns to brighten and de-noise dark photos.

## How it works

1. **Build training pairs.** Take ordinary photos (Kaggle "image-classification" dataset, *art and culture* folder, resized to 500×500),
   then simulate low light by cutting brightness to 20% in HSV space and adding salt-and-pepper noise. The original photo is the target.
2. **Model.** A fully convolutional network (Keras functional API) with parallel branches of 3×3 and 2×2 convolutions (16–64 filters),
   combined by element-wise addition (residual-style) and projected back to a 500×500×3 image. It's trained with mean-squared-error loss and Adam
   for 53 epochs.
3. **Evaluate visually.** The notebook compares ground truth, simulated low-light input and enhanced output on held-out photos from a
   different folder (*travel and adventure*).

## Limitations

- **Synthetic darkening.** Real low-light noise (sensor noise, colour casts) differs from this, so results on real night photos will be
  weaker. Paired real datasets such as LOL (LOw-Light) are the standard benchmark.
- **No numeric evaluation.** There's only visual comparison. PSNR and SSIM against the ground truth would make the results comparable.
- **Old stack.** Written for Keras 2 on Python 3.6 (`keras.layers.merge`, `keras.utils.vis_utils`). Running it on TensorFlow 2.16+
  needs `tf.keras` imports (for example `tf.keras.layers.Add`).

## Run it

Written as a Kaggle notebook using the paths `../input/image-classification/...`. Attach that dataset in Kaggle, or point `InputPath` and
`TestPath` at local folders of photos.

Tools: Keras / TensorFlow, OpenCV, NumPy, matplotlib.

Notebook: [`low-light-image-enhancement-with-cnn.ipynb`](low-light-image-enhancement-with-cnn.ipynb)
