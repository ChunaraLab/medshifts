# train and test models

# region
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 4 --random_runs 100 --min_samples 5000 --sens_attr race --group --group_type regions --limit_samples
# beds
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 4 --random_runs 100 --min_samples 10000 --sens_attr race --group --group_type beds --limit_samples
# region, beds
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 5 --random_runs 100 --min_samples 5000 --sens_attr race --group --group_type regions_beds --limit_samples
# region, beds, teaching
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 6 --random_runs 100 --min_samples 4000 --sens_attr race --group --group_type regions_beds_teaching --limit_samples
# hospitals
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 10 --random_runs 100 --min_samples 1631 --sens_attr race --limit_samples

# plot model performance

# region
python generate_hosp_plot.py --datset eicu --path orig --test_type multiv --num_hosp 4 --random_runs 100 --min_samples 5000 --sens_attr race --group --group_type regions --limit_samples
# beds
python generate_hosp_plot.py --datset eicu --path orig --test_type multiv --num_hosp 4 --random_runs 100 --min_samples 10000 --sens_attr race --group --group_type beds --limit_samples
# region, beds
python generate_hosp_plot.py --datset eicu --path orig --test_type multiv --num_hosp 5 --random_runs 100 --min_samples 5000 --sens_attr race --group --group_type regions_beds --limit_samples
# region, beds, teaching
python generate_hosp_plot.py --datset eicu --path orig --test_type multiv --num_hosp 6 --random_runs 100 --min_samples 4000 --sens_attr race --group --group_type regions_beds_teaching --limit_samples
# hospitals
python generate_hosp_plot.py --datset eicu --path orig --test_type multiv --num_hosp 10 --random_runs 100 --min_samples 1631 --sens_attr race --limit_samples
