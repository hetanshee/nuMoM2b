# nuMoM2b: Predicting Adverse Pregnancy Outcomes with Machine Learning

This repository contains analysis code and documentation developed for the Master's thesis *Predicting Adverse Pregnancy Outcomes Using Machine Learning: An Analysis of the nuMoM2b Dataset* (Loyola Marymount University, December 2024).

The project develops machine learning pipelines for maternal health risk prediction using the NIH/NICHD Nulliparous Pregnancy Outcomes Study: Monitoring Mothers-to-Be (nuMoM2b), a prospective multi-site cohort of 10,038 nulliparous women. The analysis integrates mental health, sleep, nutrition, physical activity, demographic, and clinical variables to evaluate whether multi-domain models improve prediction of maternal mental health outcomes compared with conventional clinical and demographic features alone.

**Scope of this repository.** This repository contains the core domain-level analysis notebooks and documentation that supported the thesis. Because the nuMoM2b dataset is controlled-access, participant-level data and derived intermediate files are not included. The thesis PDF provides the complete methodology, cohort derivation, variable definitions, final-stage model integration, and results; the code here is intended to document the analytical workflow and implementation details.

**This repository does not contain participant-level data.** The nuMoM2b dataset is controlled-access and is distributed through the NICHD Data and Specimen Hub (DASH).

---

## Why This Project Matters

Many maternal and women's health outcomes are shaped by interacting biological, psychological, behavioural, and social factors, yet prediction models often rely narrowly on conventional clinical variables. By integrating mental health, sleep, nutrition, physical activity, and demographic features within a single modelling framework, this work examines whether multi-domain machine learning can provide earlier and more interpretable risk signals for maternal health, and which domains carry the most predictive value.

---

## Key Results

For maternal mental health prediction, the multi-domain model combining mental health, sleep, nutrition, and sociodemographic features achieved an AUC of 0.9 and 85.4% accuracy in stratified 5-fold cross-validation. The multi-domain model excluding sleep features achieved an AUC of 0.8 and 77.2% accuracy on the larger analytical sample.

> **Important caveat:** The AUC = 0.9 result was obtained on the smaller n=218 sleep-actigraphy subsample, while the AUC = 0.8 result was obtained on the larger n=5,720 analytical sample without sleep features. These figures should be read as evidence of the potential incremental value of sleep measures, not as a one-to-one comparison across identical cohorts.

Among individual domains, the sleep model reached AUC = 0.8 (XGBoost), while maternal demographics and maternal health each reached AUC = 0.7. Feature importance analysis identified psychological factors (Hassle/Uplift Intensity Ratio, average perceived stress, resilience), sleep metrics (efficiency, fragmentation, Wake After Sleep Onset), and dietary quality (AHEI-2010) as the leading predictors.

Two further targets were attempted with limited success. A combined adverse neonatal outcome (low birth weight with preterm birth) could not be modelled reliably because its prevalence was below 2%, producing severe class imbalance. Breastfeeding intention reached only AUC ≈ 0.5 across all domains, indicating that the relevant determinants are not captured in the nuMoM2b variables.

Full results, including the complete methodology and cohort derivation, are reported in the thesis.

---

## Repository Structure

```
nuMoM2b/
├── src/             Analysis notebooks and helper modules
├── src/Medication/  Exploratory medication analyses and derived tables
├── thesis.pdf       Complete thesis (authoritative methodology and results)
└── README.md        This file
```

The `src/` directory contains the notebooks and reusable code used for the thesis analyses. Because the underlying dataset and derived intermediate files cannot be redistributed, the notebooks are provided to show the analytical workflow, modelling approach, and implementation details supporting the thesis.

**Shared modeling code**
- `shared_modeling.py` — common preprocessing, score computation, split management, and model-training helpers used across notebooks

**Mental health outcome models**
- `mental_health_outcome/Combined.ipynb` — combined multi-domain mental health model
- `mental_health_outcome/Maternal Health.ipynb` — maternal health feature set
- `mental_health_outcome/Mother_demo.ipynb` — maternal demographics
- `mental_health_outcome/Demo_father.ipynb` — paternal demographics
- `mental_health_outcome/Food_Model.ipynb` — nutrition and dietary quality
- `mental_health_outcome/Health_knowledge_model.ipynb` — health knowledge
- `mental_health_outcome/Physical Activity.ipynb` — physical activity
- `mental_health_outcome/Sleep_model.ipynb` — actigraphy-based sleep measures
- `mental_health_outcome/Drugs.ipynb` — medication-related variables

**Pregnancy outcome models**
- `pregnancy_outcomes/` — notebook-based outcome modelling and feature preparation for pregnancy-related endpoints

**Exploratory analysis utilities**
- `data_analysis/` — clustering and plotting helpers used during exploratory analysis

The `src/Medication/` directory contains exploratory analyses of medication-related variables, including clustering of medications and their recorded reasons for use.

> *Note: the older standalone scoring notebooks have been retired in favour of the shared modeling workflow in `shared_modeling.py`.*

---

## Data Access

The nuMoM2b dataset is a controlled-access resource distributed by the *Eunice Kennedy Shriver* National Institute of Child Health and Human Development (NICHD). Access requires a formal request and approval through the NICHD Data and Specimen Hub (DASH).

Researchers may request access through the DASH portal at <https://dash.nichd.nih.gov/study/226675>. The official study documentation, including protocols, data collection forms, codebooks, and variable dictionaries, is provided to approved users through DASH. The variables used in this analysis, with their nuMoM2b field names, are documented in Chapter 3 and the appendix of the thesis.

---

## Methodology Summary

Starting from the full cohort of 10,038 nulliparous women recruited across eight U.S. clinical centres between 2010 and 2013, complete-case filtering produced a working sample of 7,790 participants with full sociodemographic, physical health, and pregnancy outcome data. Of these, 5,720 had completed all mental health assessments, forming the primary analytical sample, and a subsample of 218 had valid overnight actigraphy, used for the sleep-inclusive model.

Independent variables were drawn from validated instruments documented in the nuMoM2b protocol, including the State-Trait Anxiety Inventory (STAI), the Edinburgh Postnatal Depression Scale (EPDS), the Connor-Davidson Resilience Scale (CD-RISC-25), the Perceived Stress Scale, the Everyday Discrimination Scale, the Pittsburgh Sleep Quality Index, and the Pregnancy Physical Activity Questionnaire, together with nutritional intake measures and the Alternative Healthy Eating Index (AHEI-2010).

Preprocessing applied one-hot encoding, z-score standardisation, and complete-case analysis. Class imbalance was addressed with SMOTE applied only to training folds to avoid leakage. For each domain, candidate algorithms (Logistic Regression, Random Forest, XGBoost, Support Vector Machine) were compared on cross-validated AUC, and the best model per domain was retained. A stacking ensemble combined Logistic Regression, Random Forest, and XGBoost through a meta-classifier. All metrics were computed using stratified 5-fold cross-validation, with Cohen's d effect sizes and 95% confidence intervals for group comparisons, and feature importance combined with SHAP-based attribution for interpretability. The complete methodology is documented in the thesis.

---

## Relevance to Clinical AI Research

Although this project uses tabular clinical and behavioural data rather than medical images, it addresses several challenges central to clinical AI research:

- integrating heterogeneous, multi-site clinical data into a single modelling framework;
- managing controlled-access health datasets responsibly;
- designing leakage-aware validation pipelines;
- handling severe class imbalance in low-prevalence outcomes;
- using interpretability methods to support clinical reasoning rather than opaque scoring;
- recognising when predictive performance is constrained by missing or weakly captured variables.

These considerations carry directly into medical imaging AI, where model performance depends not only on architecture but also on data quality, cohort definition, annotation standards, validation design, and clinical usability.

---

## Limitations

This work has several limitations. The sleep-inclusive model was trained on a much smaller actigraphy subsample, which limits the generalisability of its performance. Complete-case filtering may introduce selection bias by excluding participants with missing assessments. The analysis used structured clinical, behavioural, and survey-derived variables rather than raw imaging or longitudinal time-series data. Some clinically important factors, such as postpartum support systems and care environment, were not captured in the dataset and likely explain the weak performance of breastfeeding intention prediction.

These limitations informed the thesis conclusion that clinical AI models should be evaluated not only on predictive performance, but also on data availability, cohort representativeness, interpretability, and clinical usefulness.

---

## Thesis

The thesis is the complete and authoritative record of the methodology, results, and conclusions summarised here. The full thesis is available in this repository as `thesis.pdf`.

> Shah, H. (2024). *Predicting Adverse Pregnancy Outcomes Using Machine Learning: An Analysis of the nuMoM2b Dataset*. Master's thesis, Loyola Marymount University.

---

## Acknowledgements

This work was completed under the supervision of Dr. Mandy Korpusik at the Department of Computer Science, Loyola Marymount University. The nuMoM2b study was funded by the *Eunice Kennedy Shriver* National Institute of Child Health and Human Development (NICHD) and the National Institutes of Health (NIH), with co-funding from the NIH Office of Research on Women's Health.

---

## License

This code is released under the MIT License. See the `LICENSE` file for details. The license applies only to the code in this repository; the nuMoM2b dataset remains subject to its NICHD Data Use Agreement.

---

## Contact

Hetanshee Shah — hetansheeshah@gmail.com — [LinkedIn](https://linkedin.com/in/hetansheeshah) — [GitHub](https://github.com/hetanshee)
