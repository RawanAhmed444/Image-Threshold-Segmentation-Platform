# Threshold Realities & Image Segmentation Application

This application is a user-friendly toolkit for digital image segmentation and thresholding, built with Python. It allows users to explore and apply a variety of classic and advanced segmentation techniques—including global and local thresholding, k-means clustering, mean shift, region growing, and agglomerative clustering—on their own images. All core algorithms are implemented from scratch, with OpenCV used only for image I/O and display.

---

## Features

- **Global & Local Thresholding**
  - Apply Otsu's, adaptive, and custom local thresholding methods
  - Visualize binary and multi-level thresholded images

- **K-Means Clustering Segmentation**
  - Segment images into regions using k-means clustering
  - Visualize clustered regions with distinct colors

- **Mean Shift Segmentation**
  - Perform mean shift clustering in 5D feature space (color + position)
  - Efficient implementation using KD-tree optimization

- **Region Growing Segmentation**
  - Interactive or automatic region growing based on intensity similarity

- **Agglomerative Clustering Segmentation**
  - Hierarchical clustering of superpixels using custom linkage strategies
  - Visualize clusters and boundaries

- **Modular & Extensible**
  - Clean code structure for easy experimentation and extension
  - Jupyter notebooks for interactive exploration

_All segmentation and thresholding algorithms are implemented from scratch, without using OpenCV for the main processing steps in Python._

---

## Project Structure

```
Task4-Threshold-Realities-Image-Segmentation/
  ├── Data/
  ├── k_means/
  ├── src/
  │     ├── segmentation/
  │     ├── thresholding/
  │     └── utils/
  ├── requirements.txt
  └── main.py
```

---

## Screenshots

<!-- Add screenshots or demo images here -->
*Agglomerative Clustering Segmentation*

![image](https://github.com/user-attachments/assets/3a863738-ee41-41e4-8ab6-a34c5a90760a)

*K-Means Segmentation*

![image](https://github.com/user-attachments/assets/53c1dace-228a-4564-8578-b57dc5f1b755)

*Mean Shift Segmentation*

![image](https://github.com/user-attachments/assets/6b760b9f-cdd1-4b34-b6c1-3f4422525e2a)


*Region Growing Segmentation*

![image](https://github.com/user-attachments/assets/243abde8-9e63-400d-9728-58225788fdeb)

*Thresholding Example*

![image](https://github.com/user-attachments/assets/143712ac-1e14-43f9-b199-6174bba70815)
![image](https://github.com/user-attachments/assets/324edcca-95ca-42ad-a72a-cf185b1405de)

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- OpenCV (for image I/O and display only)
- Matplotlib
- SciPy (for KD-tree in mean shift)

### Installation

```sh
cd Task4-Threshold-Realities-Image-Segmentation
pip install -r requirements.txt
python main.py
```

---

## Contributors

* **Rawan Shoaib**: [GitHub Profile](https://github.com/RawanAhmed444)
* **Ahmed Aldeeb**: [GitHub Profile](https://github.com/AhmedXAlDeeb)
* **Ahmed Mahmoud**: [GitHub Profile](https://github.com/ahmed-226)
* **Eman Emad**: [GitHub Profile](https://github.com/Alyaaa16)

---

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-image Segmentation Guide](https://scikit-image.org/docs/stable/user_guide/tutorial_segmentation.html)
- [Mean Shift Clustering Paper](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/papers/Comaniciu_MeanShift.pdf)
- [Agglomerative Clustering Overview](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

*You can also have a look at our [report](https://drive.google.com/file/d/15cwDQ7J_kcL8NU-0F3znew6tUgDfxzOc/view?usp=sharing)*

---
