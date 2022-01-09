<!-- PROJECT LOGO -->
<br />
<p align="center">
  <img src="umcg_logo.png" alt="Logo" width="540" height="240">

  <h3 align="center">Divide and conquer</h3>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>	
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#images">Images</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Independent component analysis (ICA) is used to disentangle gene expression data into biological pathways. Current implementations of ICA use principal component analysis to drop a percentage of variance of the data to make it computational feasible.  However, the percentage of dropped variance can contain important information about rare cancer types. We propose a solution called divide and conquer. In this research we show that by first using high dimensional data clustering (HDDC) to cluster a dataset, and then running ICA with no dropped variance on each of the clusters, new information is found that was otherwise dropped. HDDC was chosen because it shows a good silhouette score combined with easy-to-understand cluster decisions based on used genes.  Our approach found an estimated source describing a pathway related to a rare form of cancer called mantle cell lymphoma. This estimated source has not been found previously with ICA. Results demonstrate that divide and conquer is capable of finding new pathways that were otherwise missed.  We anticipate our paper to be the starting point in developing a sophisticated divide and conquer approach capable of splitting datasets and using this to find every possible biological pathway present among the samples. 


This Github page is about the first part of the project, the clustering. High dimensional data clustering (HDDC), mini batch K-means, Hiearchical clustering and UMAP+HDBSCAN are gridsearch and tested on three different datasets. The datasets are the GPL570, CCLE, and TCGA datasets. 


### Built With

* Python 3.8.5
* R 3.6.1



<!-- GETTING STARTED -->
## Getting Started

The scripts can't be run without access to the Peregrine cluster of the RUG. On this server the datasets are stored and access needs to be provided on request. The datasets are available as public repositories and can be found the following way:

For the GEO platform, healthy and cancer samples were selected. These samples were selected with a two-step approach. First, automatic keyword filtering was applied. In this approach, the simple omnibus format in text (SOFT) was scanned. SOFT files contain metadata for each sample, this includes experimental condition and patient information. In this search approach only samples were kept if certain keywords can be matched with the descriptive field in the SOFT file. These keywords were chosen very broadly like 'breast' or 'lung'. Because of this broad approach a manual check was needed to remove false positives. In this step, only samples were kept if raw data was available and the samples represented a healthy or cancer tissue of patients. Cell lines, cultured human biopsies, and animal-derived tissue were excluded in this step. 

For the TCGA the data was obtained from 34 cancer datasets available at the Broad GDAC Fire hose portal https://gdac.broadinstitute.org/. Here gene normalized RNA-sequence data was downloaded. Fragments per kilo-base of transcript per million mapped reads upper quartile normalization https://docs.gdc.cancer.gov/Data/PDF/Data_UG.pdf was used to normalize RNA-Seq expression level read counts.

The CCLE dataset contains raw mRNA data of human cell lines. The following research conducted a detailed genetic characterization of these cell lines (The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity, Barretina) 

### Prerequisites
Jupiter notebook, Python and R should be installed and working before the main script can be used. 

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MarkF-hanze/UMCG_Thesis
   ```
2. Install the required packages
   ```sh
   pip3 install -r requirements.txt
   ```
3. Get acces to the requered files


<!-- USAGE EXAMPLES -->
## Usage
- Bash_Runs/      Contains scripts to run in the cluster.
  * Jobs: contains the to be run files.
  * Outputs: contains the output results of these runs
- Scripts/Gridsearch Contains scripts to run the gridsearch
  * Main: Search the best parameters for K-means and UMAP+HDBSCAn algorithm
  * HDDC: give parameters (filepath) (clusternumber) search for the best HDDC parameters for this number of cluster for this dataset
  * TestTimeR: Temporary file to test how long the HDDC algorithm will run on the comple GPL570 dataset to see if it is computationally feasible
  * make_scores: Functions to calculate the silhouette score for a given dataset with labels
- JupyterNotebook: This file contains notebooks that were used to further analyise some of the results
- Results: Contains all the resulting images from all the analysis. Contains the best parameters and other results for every dataset and every algorithm. Also contains *dashboard.html* files. These files contain some manual evaluation results for how each clustering algorithm clusters the cancer type. Also the algorithms clustering behaviour is further analysed with the help of a sankey plot.



<!-- LICENSE -->
## License

Distributed under the mozilla license. See `LICENSE` for more information.

<!-- IMAGES -->
## Images
 ### End results comparison algorithms
   <img src="Scripts/Results/IMAGES/Best_Scores.png" alt="End Result">




