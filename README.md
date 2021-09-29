# medshifts

How does dataset shifts across hospitals affect model performance? This codebase analyzes the multi-hospital eICU database to help answer this question.

Pre-print link: https://www.medrxiv.org/content/10.1101/2021.07.14.21260493v1

Scipts for extracting and pre-processing eICU datasets can be accessed from this repo by Alistair Johnson at https://github.com/alistairewj/icu-model-transfer.

We build on the scripts for two-sample testing and model training-testing by Stephan Rabanser from this repo https://github.com/steverab/failing-loudly.

---

## Run instructions

Main file is `hosp_pipeline_parallel.py`.

Following commands are run on parallel across processes. Adjust `num_cores=1` in `hosp_pipeline_parallel.py` to disable parallelization.

To run all experiments:

```
bash run.sh
```

To run main script for comparing across hospitals:

```
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 10 --random_runs 100 --min_samples 1631 --sens_attr race --limit_samples
```

To generate heat maps for results from above command:
```
python generate_hosp_plot.py --datset eicu --path orig --test_type multiv --num_hosp 10 --random_runs 100 --min_samples 1631 --sens_attr race --limit_samples
```

---

## Dependencies

Install following with pip:

```
pip install numpy scikit-learn scipy pandas matplotlib seaborn
```

Install following from respective sites:

- `keras`: https://github.com/keras-team/keras
- `tensorflow`: https://github.com/tensorflow/tensorflow
- `pytorch`: https://github.com/pytorch/pytorch
- `torch-two-sample`: https://github.com/josipd/torch-two-sample
- `keras-resnet`: https://github.com/broadinstitute/keras-resnet

---

## Summary

With the growing use of predictive models in clinical care, it is imperative to assess failure modes of predictive models across regions and different populations. In this retrospective cross-sectional study based on a multi-center critical care database, we find that mortality risk prediction models developed in one hospital or geographic region setting exhibited lack of generalizability to different hospitals/regions. Moreover, distribution of clinical (vitals, labs and surgery) variables significantly varied across hospitals and regions. We postulate that dataset shifts in race and clinical variables due to hospital or geography result in mortality prediction differences according to causal inference results, and the race variable commonly mediated changes in clinical variable shifts. Findings demonstrate evidence that such models can exhibit disparities in performance across racial groups even while performing well in terms of average population-wide metrics. Therefore, assessing subgroup performance differences should be included in model evaluation guidelines. Based on shifts in variables mediated by the race variable, understanding and provenance of data generating processes by population sub-group are needed to identify and mitigate sources of variation and can be used to decide whether to use a risk prediction model in new environments.

