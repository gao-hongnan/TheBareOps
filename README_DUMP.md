## GOOGLE DIAGRAM

The terminology of "Orchestrated Experiment" and "Automated Pipeline" in
Google's MLOps diagram refers to different stages of a machine learning project.

Orchestrated Experiment: This stage typically occurs in a development or
experimental setting. Here, data scientists or machine learning engineers are
exploring different models, feature engineering strategies, and hyperparameters
to find the most effective solution for the problem at hand. The steps in this
stage, while methodical, often involve trial and error, exploration, and
backtracking. Orchestrating these experiments means organizing and managing them
in a systematic way, often using tools such as notebooks (e.g., Jupyter),
experiment tracking tools (e.g., MLflow, TensorBoard), version control (e.g.,
Git), and data versioning tools (e.g., DVC).

Automated Pipeline: Once an effective solution has been found in the experiment
stage, the next step is to automate this solution so that it can be run
repeatedly and reliably. This involves translating the experimental code into a
production-grade pipeline, which includes not only the model training code, but
also data extraction, preprocessing, validation, model serving, and monitoring.
This pipeline is typically set up to run automatically, either on a regular
schedule or in response to certain triggers. The purpose of automation is to
ensure consistency, efficiency, and reliability. It allows the machine learning
solution to operate at scale and in real-world environments.

Essentially, the orchestrated experiment stage is about finding the best
solution, while the automated pipeline stage is about deploying and operating
that solution at scale. Both stages involve similar steps (like data validation,
data preparation, model training, model evaluation, and model validation), but
the context and objectives are different.

## Confusion

I think it dawned upon me after re-reading mlops diagram from google.

They have two components, one is exp/dev environment where ml engineers offline
extract data to somewhere for data analysis. Subsequently, google boxed the
follow five steps under "orchestrated experiment".

- data validation
- data preparation
- model training
- model evaluation
- model validation

Then they point this boxed area to another box called source code and point it
to another box called source repository. Subsequently, they point this source
repository to another box called pipeline deployment. The key is now this
pipeline deployment cross over to the second component, namely the
staging/production environment. Let me detail the box in the staging/production
environment.

The box is called "automated pipeline" where it contains the following steps:

- data extraction (here i see an arrow from a data warehouse/feature store)
- data validation
- data preparation
- model training
- model evaluation
- model validation

then this pipeline goes to model registry to CD: model serving.

What I did wrong with the dvc is that I included the dvc add and push in the
production environment. This is wrong. The dvc add and push should be done in
the exp/dev environment. The dvc pull should be done in the production
environment. Am i correct? Please dont hesitate to correct me.
