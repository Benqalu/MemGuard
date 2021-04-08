# Modified MemGuard

- This code is forked from [the original MemGuard](https://github.com/jjy1994/MemGuard);
- To run the code, you need tensorflow 1.14 with compatible version of Keras;
- In the original version, you need to run a list of Python scripts one by one, and some of the final results that are need in our project are not collected, so I modified their code to save those intermediate results. Now you only need to run the following commend:
	- ``` python run.py DATASET_NAME``` (The argument ```[DATASET_NAME]``` could one of: ```adult```, ```broward```, ```hospital``` or ```compas```);
- **Before you proceed the experiments, please read ```run.py``` carefully to make sure you understand the whole work flow of the script;**
- In our current stage we only run it for the first three datasets, just in case the dataset is changed and compas is added later, I prepared it as well;
- The original version of MemGuard only takes a dataset name *location*, so in my version, I load those datasets I mentioned above and generate the same formate of dataset to pretend it to be the location dataset and let MemGuard run on it;
- For the original location dataset, the MemGuard reads a configuration file *config.ini*, however in our case, since we replaced the dataset we must keep the configuration consistent, so I created a template file named *config_template.txt*, my script ```run.py``` will read the template and change the parameters in it according to the dataset you choose, and overwrite current config.ini. 
- There are still some parameters that you can tune. But remember, **when you tune the parameters, you need to change them in config_template.txt, not config.ini**. 
- All parameters in config_template.txt that are not a format string (like %d, %f), can be tuned.
- When you run ```run.py```, the code will output a ```.npz``` file in ```./result/``` named with the dataset you choose and a time stamp;
- To get a stable experiment results, you need to run the script for each dataset multiple times, like 10, you can manually run them or use ```exec.py``, which is a very simple script to do so;
- The last a few lines in ```run.py``` indicates the structure of the outputed file, you can collect them manually, or run my code ```parse_npz.py``` in ```./results/``` to get the need results. However, my collection code only collects specific metrics, if the professor requires more results, you need to understand the structure and do it yourself.