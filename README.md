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
Independent component analysis (ICA) is used to disentangle gene expression data into biological pathways.
Current implementations of ICA use principal component analysis to drop a percentage of variance of the data to
make it computational feasible. However, the percentage of dropped variance can contain important information
about rare cancer types. We propose a solution called divide and conquer. In this research we show that by
first using high dimensional data clustering (HDDC) to cluster a dataset, and then running ICA with no dropped
variance on each of the clusters, new information is found that was otherwise dropped. HDDC was chosen because
it shows a good silhouette score combined with easy-to-understand cluster decisions based on used genes. Our
approach found an estimated source describing a pathway related to a rare form of cancer called mantle cell
lymphoma. This estimated source has not been found previously with ICA. Results demonstrate that divide and
conquer is capable of finding new pathways that were otherwise missed. We anticipate our paper to be the starting
point in developing a sophisticated divide and conquer approach capable of splitting datasets and using this to
find every possible biological pathway present among the samples


This Github page is about the second part of the project, the behaviour of ICA. 2000 random samples of the GPL570 dataset were taken and the behaviour is analysed. The biological interpretation of this algorithm is analysed by running it on the complete GPL570 dataset split by cancer type. 


### Built With

* Python 3.9.6
* R 3.6.1



<!-- GETTING STARTED -->
## Getting Started

For the GEO platform, healthy and cancer samples were selected. These samples were selected with a two-step approach. First, automatic keyword filtering was applied. In this approach, the simple omnibus format in text (SOFT) was scanned. SOFT files contain metadata for each sample, this includes experimental condition and patient information. In this search approach only samples were kept if certain keywords can be matched with the descriptive field in the SOFT file. These keywords were chosen very broadly like 'breast' or 'lung'. Because of this broad approach a manual check was needed to remove false positives. In this step, only samples were kept if raw data was available and the samples represented a healthy or cancer tissue of patients. Cell lines, cultured human biopsies, and animal-derived tissue were excluded in this step. 

From this dataset 2000 random samples were selected

### Prerequisites
Jupiter notebook, Python and R should be installed and working before the main script can be used. 

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MarkF-hanze/ICA
   ```
2. Install the required packages
   ```sh
   pip3 install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage
- Main: Script to analyse the behaviour of a single split run (for example two splits of random splitting). Filepath to split should be put into this file to get it working on new data. File order in this file should be: Directories(names of split) with every directory containing the files ICARUN. ICARUN should contain ICA results as given by the analysertool. 
- Main_Merged: Script to compare the different splitting optionts. Script to compare the behaviour beteween different splitting optionts. This script probably shouldn't be used anymore because a lot of manual inputs are given to specificly compare splitting methods and optionts available to this research. Newly added optionts or splits can be manually added but needs changes in a lot of places.
- PCA_Check: Script to check the normalization behaviour of two random splits
- /Resutls/ : Contains the results of all the different scripts. /Clusterd and /Random contain the Main results from the respective splitting tactics. /Random_vs_Clustered contains the results of the Main_Merged script comparing the two different splitting options. Cancer_Type contains the results of the complete GPL570 dataset split by cancer type for the biological interpertation
- /Old: Scripts and results that weren't used in the report. Only left in case they would be usefull for further reserach
- /Jupyter_Notebooks: Containing scripts that had some manual checks to see biological interpertations. Shouldn't be changed but can be looked at for some deeper insights not always present in the report
- /HelperClasses: Folder contain different classes that are used in multiple files. 
  * BiologicalInterpertation:  Classes doing different analysis explained in the paper. (Maximun correlation, interesting sources)
  * CitrusPlot: Class creating the citrusplots
  * Correlation: Class calculating the correalion between two different dataframes
  * Heatmap: Making the heatmap and calculating different clusters based on this heatmap
  * LoadData: Loading the different datastes with consensus estimated sources.



<!-- LICENSE -->
## License

Distributed under the mozilla license. See `LICENSE` for more information.





