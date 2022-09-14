++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+                               Directories et files descriptions                          +                                                                                    
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     -----Directories :
             *2d_projection : contains lidar files from xgboost cable prediction and subdirectories of their 
                              respective 2d image projection. 
             *data : contains raw lidar data 
             *intermediate : contains intermediate results of preprocessing pipeline filter
             *preprocessed_data : contains preprocessed lidar data 
             *results : contains the results of XGBoost and Hough respectively in the subdirectories xgboost and hough

     -----files :
             *utils : contains essential functions for hough transform
             *sectioning : contains functions to split lidar into several sections
             *projection : contains functions to split lidar into several slices and do 2d image projection
             *preprocessing : preprocessing pipeline 
             *prediction_xgboost : contains functions for the training and prediction with xgboost 
             *LayeredHoughTransformOpenCV : postprocessing with Hough Transform

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+                                        Execution                                         +                                                                                    
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    There are three  main  parts : preprocessing, xgboost prediction, and postprocessing.
             *preprocessing : depending on the input file, it may take several minutes or even hours. But there are already
					preprocessed files in "preprocessed_data". Only execute this phase if you want to bring some
					changes to the pipeline.  

             *prediction_xgboost :  Since there is already preprocessed data, you can directly run this file to see xgboost
    						prediction. you must  give the filename to predict without extension in the main.
                                    It will then automatically adjust training and test data. 
             
             *LayeredHoughTransformOpenCV : Hough Transform postprocessing, you need to  have xgboost prediction before running this file. 

 
