---
title: "3DP-TP-00 | 3D segmentation and object shape checking based on RGB-D sensor with CloudCompare"
date: 2022-10-03T08:06:25+06:00
description: "3D objects quality control."
summary: "3D objects quality control."

math: true 
highlight: true
hightlight_languages: ["python","bash"]

authors: ["Claire Labit-Bonis"]

hero: featured.png

tags: ["Master", "3D Perception"]

menu:
  sidebar:
    name: Segmentation
    identifier: cc-segmentation
    parent: 3d-perception
    weight: 10
---

<!-- +++
title = "3D segmentation and object shape checking based on RGB-D sensor"


tags = ["teaching", "perception"]
summary = "3D objects quality control."

+++ -->

## Contents download
Practical session files can be downloaded through your course on Moodle, or at [this link](files/files.zip).

## Goal description {#anchor-step-0}
This practical session aims at performing 3D segmentation and shape checking on polyhedral objects, i.e. checking if these objects are correct with respect to a reference geometrical model and/or have defects (holes, residues, etc.). 

{{< img src="images/objets_a_comparer.png" align="center" title="Objects to compare" >}}
{{< vs 1 >}}

To do this, we must first build this reference model from an RGB-D image of a defect-free object. Then, for any view of an unknown (also called "test") object, we must segment it from the background and compare it with the reference model. This process of checking the shape must be independent of the viewpoint and,
This shape verification process must be viewpoint independent and therefore requires the registration of each associated point cloud against the reference one.

{{< alert type="info" >}}
We propose to break this objective down into three steps:

- [step 1](#anchor-step-1) : **extract the 3D points** of the objects to be compared (both reference and test objects) by deleting all the points of the scene not belonging to the object. To avoid a redundant process, this step will be performed only on the reference scene contained in `data01.xyz`; this has already been done on the objects to be tested, and stored in the files `data02_object.xyz` and `data03_object.xyz`.
- [step 2](#anchor-step-2) : **register** the points of each test object to the reference model in order to compare them i.e. align their respective 3D point clouds to the reference coordinate frame.
- [step 3](#anchor-step-3) : **compare** the control and reference models and conclude on the potential flaws of the control models.
{{< /alert >}}

## ***Step 1***: 3D model extraction from the reference scene

The first step of the practical session consists in extracting the point cloud of the reference model from the RGB-D scene acquired with a Kinect :

{{< img src="images/extraction_objet.png" align="center" title="Reference model extraction" >}}
{{< vs 1 >}}

This step aims at tracing a plane surface on the ground plane, and only keeping the center box by calculating the distance of each of its points from this plane, before applying a filtering threshold.

To do this, open CloudCompare (the main program, not the viewer) and import the points of the `data01.xyz` scene. Select the cloud by clicking on it in the workspace. 
Using the segmentation tool (**Edit > Segment**, or directly the "scissors" shortcut in the shortcut bar), divide the cloud into three subsets in order to extract the ground plane and a rough area around the box.
The result is shown in the following figure:

{{< img src="images/extraction_modele.gif" align="center" title="Division of the scene into three clouds" >}}
{{< vs 1 >}}

{{< alert type="warning" >}}
In CloudCompare, to work on a point cloud, the corresponding line must be selected in the workspace. You know if the cloud is selected when a yellow box is displayed around it.

Checking the box does not select the cloud, it simply makes it visible/invisible in the display.
{{< /alert >}}

Create a surface fitting the ground plane cloud using the **Tools > Fit > Plane** tool. 
By selecting the newly created plane and the cloud that contains the box, it is now possible to calculate, for each of the points of this cloud, its distance to the plane using the **Tools > Distances > Cloud/Mesh Distance** tool:

{{< img src="images/fit_plane_compute_distance.png" align="center" title="Plane surface and distance" >}}
{{< vs 1 >}}

The distance tool adds a fourth field to each point of the cloud: the newly calculated distance. Using the cloud properties, filter the points with respect to this scalar field to keep only the points belonging to the box :

{{< img src="images/filter_by_value.gif" align="center" title="Filtering by the distance to the plane" >}}
{{< vs 1 >}}

By clicking on *split*, two clouds are created, corresponding to the two sides of the filter:

{{< img src="images/split.png" align="center" title="Extracted box" >}}
{{< vs 1 >}}

Make sure that the newly created cloud contains about 10,000 points (the number of points is accessible in the properties panel on the left).

Only select the box cloud before saving it in ASCII Cloud format as `data01_segmented.xyz` in your `data` folder.
 
{{< alert type="info" >}}
As a precaution, save your CloudCompare project: remember to **select all point clouds**, and save the project in CloudCompare format.
{{< /alert >}}

## ***Step 2***: 3D points registration

If you have opened the complete scenes `data02.xyz` and `data03.xyz` in CloudCompare, you will have noticed that each scene was taken from a slightly different point of view, and that the objects themselves have moved:

{{< img src="images/points_vue_diff.gif" align="center" title="Different views of the scenes" >}}
{{< vs 1 >}}

In order to compare the models between them, we propose to overlay them and to calculate their cumulative distance point to point. The smaller this distance, the more the models overlap and resemble each other; the larger it is, the more the models differ.
The following example shows the superposition of the correct model on the previously extracted reference model:

{{< img src="images/plot_after_icp.png" align="center" title="ICP applied to the correct model" >}}
{{< vs 1 >}}

Transforming the points of a model via a rotation/translation matrix to overlay it on another cloud is called *point registration*. 
The ***Iterative Closest Point*** algorithm allows this registration, and we propose to use it in Python. 
The code to be modified is only in `qualitycheck.py`, the goal being to apply ICP on both the correct model `data02_object.xyz`, and on the faulty model `data03_object.xyz`.

### Loading models
The first part of the code loads the `.xyz` models extracted with CloudCompare, stores the reference model in the `ref` variable and the model to be compared in the `data` variable.
To run the code on either `data02_object` or `data03_object`, just comment out the corresponding line.

```python
# Load pre-processed model point cloud
print("Extracting MODEL object...")
model = datatools.load_XYZ_data_to_vec('data/data01_segmented.xyz')[:,:3]

# Load raw data point cloud
print("Extracting DATA02 object...")
data02_object = datatools.load_XYZ_data_to_vec('data/data02_object.xyz')

# Load raw data point cloud
print("Extracting DATA03 object...")
data03_object = datatools.load_XYZ_data_to_vec('data/data03_object.xyz')

ref = model_object
data = data02_object
# data = data03_object
```

### ICP call

The second part of the code consists in coding the call to the `icp` function of the `icp` library... 

```python
##########################################################################
# Call ICP:
#   Here you have to call the icp function in icp library, get its return
#   variables and apply the transformation to the model in order to overlay
#   it onto the reference model.

matrix = np.eye(4,4)        # Transformation matrix returned by icp function
errors = np.zeros((1,100))  # Error value for each iteration of ICP
iterations = 100            # The total number of iterations applied by ICP
total_time=0                # Total time of convergence of ICP

# ------- YOUR TURN HERE -------- 

# Draw results
fig = plt.figure(1, figsize=(20, 5))
ax = fig.add_subplot(131, projection='3d')
# Draw reference
datatools.draw_data(ref, title='Reference', ax=ax)

ax = fig.add_subplot(132, projection='3d')
# Draw original data and reference
datatools.draw_data_and_ref(data, ref=ref, title='Raw data', ax=ax)
```

...and store the return of the function in the variables `T`, `errors`, `iterations` and `total_time` as defined by the function definition header in the file `icp.py`:

```python
def icp(data, ref, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        errors: Euclidean distances (errors) for max_iterations iterations in a (max_iterations+1) vector. distances[0] is the initial distance.
        i: number of iterations to converge
        total_time : execution time
    '''
```

### Model transformation

The `T` transformation matrix from ICP is the homogeneous passage matrix that allows us to map the `data` model, passed as a parameter to the `icp` function, onto the `ref` model.
As a reminder, the application of a homogeneous matrix to transform a set of points from an initial reference frame \\(\mathcal{R_i}\\) to a final reference frame \\(\mathcal{R_f}\\) is performed as follows:

$$P_f^{(4 \times N)} = T^{(4 \times 4)} . P_i^{(4 \times N)}$$

In the code, the third part consists in applying the transformation matrix to the model to be compared. An example of how to apply a homogeneous transformation to the matrix can be written as follows:

```python
# EXAMPLE of how to apply a homogeneous transformation to a set of points
''' 
# (1) Make a homogeneous representation of the model to transform
homogeneous_model = np.ones((original_model.shape[0], 4))   ##### Construct a [N,4] matrix
homogeneous_model[:,0:3] = np.copy(original_model)          ##### Replace the X,Y,Z columns with the model points
# (2) Construct the R|t homogeneous transformation matrix / here a rotation of 36 degrees around x axis
theta = np.radians(36)
c, s = np.cos(theta), np.sin(theta)
homogeneous_matrix = np.array([[1, 0, 0, 0],
                               [0, c, s, 0],
                               [0, -s, c, 0],
                               [0, 0, 0, 1]])
# (3) Apply the transformation
transformed_model = np.dot(homogeneous_matrix, homogeneous_model.T).T
# (4) Remove the homogeneous coordinate
transformed_model = np.delete(transformed_model, 3, 1)
```

{{< alert type="info" >}}
The `original` variable is an array of size \\(N \times 3\\), \\(N\\) being the number of points of the model and 3 its coordinates \\(X\\), \\(Y\\) and \\(Z\\).

You need to add a homogeneous coordinate and apply the necessary transpositions for the matrix multiplication to work.
Use the example given in the code to perform this step.
{{< /alert >}}

You can then display the result by uncommenting and completing the line `datatools.draw_data...`.

### Error display
Uncomment and display the error in the last part of the code, changing the "..." to the corresponding variables:

```python
# Display error progress over time
# **************** To be uncommented and completed ****************
# fig1 = plt.figure(2, figsize=(20, 3))
# it = np.arange(0, len(errors), 1)
# plt.plot(it, ...)
# plt.ylabel('Residual distance')
# plt.xlabel('Iterations')
# plt.title('Total elapsed time :' + str(...) + ' s.')
# plt.show()
```

## ***Step 3***: Models comparison

Compare the application of ICP on the models `data02` and `data03`, notice the evolution of the error and the differences in values.
What does this error represent? What can we say about the two models? Based on the errors, what decision threshold could you choose to determine whether a model is faulty or not?

### ICP in CloudCompare

The ICP algorithm can also be used directly in CloudCompare. Open it and import `data01_segmented.xyz`, `data02_object.xyz` and `data03_object.xyz`.

Select for example the clouds of `data01_segmented` and `data02_object`, use the **Tools > Registration > Fine registration (ICP)** tool. Make sure the reference is `data01` and apply ICP.
Running it returns the transformation matrix calculated by the algorithm, and applies it to the object.

{{< img src="images/registration_cc.png" align="center" title="ICP in CloudCompare" >}}
{{< vs 1 >}}

We can then, still selecting the two clouds, calculate the distance between the points with **Tools > Distance > Cloud/Cloud Distance**. Make sure that the reference is `data01` and click on OK/Compute/OK.
Select `data02_object` and display the histogram of its distances to the reference cloud *via* **Edit > Scalar fields > Show histogram**.

{{< img src="images/histogram_dists.png" align="center" title="Histogram of point to point distances of data02 vs. data01" >}}
{{< vs 1 >}}

Do the same thing with `data03_object` and compare the histograms. How do you interpret them? How can you compare them?

Created by our teacher [Claire Labit-Bonis](https://clairelabitbonis.github.io/).