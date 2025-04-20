
    "[![author](https://img.shields.io/badge/Author-jean&nbsp;augusto-red.svg)](https://www.linkedin.com/in/jean-augusto-morais/)\n",
    "[![](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)\n",
    "\n",
    "# Case iFood - Data analyst\n",
    "\n",
    "A well-established food retail company serves nearly one million consumers annually, offering products in five main categories (wines, meats, exotic fruits, specially prepared fish, and sweets), available in both regular and premium versions. Sales are made through physical stores, catalogs, and the company’s website. Although the company has seen solid financial results over the past three years, future profit growth projections are not promising. To address this, several strategic initiatives are being considered, including enhancing marketing campaign performance.\n",
    "\n",
    "![pairplot](images/pairplot_clusters.png)\n",
    "\n",

"**Data Science Project. Based on the selection process for the Data Analyst position at iFood, available [in this repository](https://github.com/ifood/ifood-data-business-analyst-test).**.\n",
    "\n",
    "Complete case description [here](case/README.md).\n",
    "\n",

    "</p>\n",
    "\n",
    "## Objetivos\n",
    "\n",

    "\n",
.\n",
    "\n",
## **Detailed Objectives:**

- Conduct a robust exploratory data analysis.
- Segment the customers in the provided dataset.
- Build a classification model to predict whether a customer will purchase the product offered in the campaign.
- Present a Data Science project structure, including notebooks, scripts, reports, and a GitHub repository.
- Demonstrate best practices in Python programming, such as using functions and script files to facilitate code reuse.
- Apply best practices with the SciKit-Learn library, including using pipelines and hyperparameter optimization.

## **Repository Structure**

```
├── case
├── data
├── images
├── notebooks
├── reports
```

- The `data` folder contains the data used in the project. The file `ml_project1_data.csv` is the original dataset, and the other files are datasets generated during the project.
- The `images` folder contains the images used in this README.
- The `notebooks` folder contains the notebooks with the project development.
- The `reports` folder contains the reports generated using the [ydata-profiling](https://github.com/ydataai/ydata-profiling) library.

## **Dataset Details and Results Summary**

A detailed description of the dataset used is available [here](data/README.md).

With a pipeline including preprocessing, PCA, and K-Means, the dataset was segmented into 3 clusters:

![clusters](images/pca_clusters.png)

### **Cluster Analysis:**

- **Cluster 0:** 
  - High income
  - High spending
  - Likely no children
  - More likely to accept campaigns
  - No members with basic education
  - No prominent age group

- **Cluster 1:** 
  - Low income
  - Low spending
  - Likely has children
  - Low propensity to accept campaigns
  - Only cluster with a significant percentage of people with basic education
  - Younger people

- **Cluster 2:** 
  - Mid-range income
  - Mid-range spending
  - Likely has children
  - May accept campaigns
  - Older people

Next, three classification models were trained to predict whether a customer would purchase the product in the campaign. The models used were:

- Logistic Regression
- Decision Tree
- KNN

A **DummyClassifier** was used as a baseline. The models were evaluated based on 6 metrics:

![comparing_models](images/comparing_models.png)

Based on this comparison, the **Logistic Regression** model was chosen for hyperparameter optimization.

## **How to Reproduce the Project**

The project was developed using **Python 3.11.5**. To reproduce the project, create a virtual environment with Conda (or similar tool) using Python 3.11.5 and install the following libraries:

| Library          | Version |
| ---------------- | ------ |
| Imbalanced-Learn | 0.11.0 |
| Matplotlib       | 3.7.2  |
| NumPy            | 1.24.3 |
| Pandas           | 1.5.3  |
| Scikit-Learn     | 1.3.0  |
| Seaborn          | 0.12.2 |

These are the main libraries used in the project. The report was generated using the [ydata-profiling](https://github.com/ydataai/ydata-profiling) library. Install it if you wish to reproduce the report.
