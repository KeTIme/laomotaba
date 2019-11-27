# laomotaba
Project notebook is a q/py notebook, the one we are going to submit
EDA notebook is a py only notebook used for exploratory analysis to make life easier.

After doing anything meaningful, document everything in both notebooks. it is crucial to have good markdown cells on project.ipynb to get a good grade

## current progress:
  -able to download data through q query
  -select pairs by correlation
  -cointegration test to further sift through pairs
  -rudimentary pair strategy back testing
  
## TODO:
  -Practicality issues
     - fraction of a share issue
     - tcost concerns
     - liquidity concerns
          - we are using mid now try to use bid ask information for signals
          - use the depth of bid ask/strength of signal to dynamically determine position size. Position size is fixed now.
     
  - parameter tuning/feature adding
     - try new timing ideas, maybe basket strategy is better than two-pair alone, idk
     - optimize strategy parameters AFTER solving practicality issues, do NOT do this before finalizing assumptions and strategy
     
  
