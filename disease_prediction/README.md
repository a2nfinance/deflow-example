## Workflow
![](images/training_workflow.drawio.png)
## Metadata Collection
```
python collection.py --output_dir <your_output_directory> --output_zip <your_output_zip_file_name>

```
## Data Preprocessing
```
python preprocessing.py --input_data <your_data_directory> --output_dir <your_output_zip_file_name>

```
## Data Generation
```
python generation.py --output_dir <your_output_zip_file_name>

```
## Feature Selection
```
python train.py --meta <your_meta_data_directory> \
                --data <your_data_directory> \
                --gen_data <your_generated_data_directory> \
                --output_dir <your_output_zip_file_name>
```
## Prediction
```
python train.py --input_data <your_data_directory> --output_dir <your_output_zip_file_name>
```