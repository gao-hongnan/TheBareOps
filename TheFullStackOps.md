# The FullStackOps

## DataOps

1. Source (live URL) -> Google Storage -> Google BigQuery using AirByte.
2. The intermediate data lake is Google Cloud Storage Bucket, the so called
   staging area. Airbyte will sync the data from the source to the staging then
   to the destination (BigQuery).
3. Then it goes to dbt to transform the data into the desired schema.

### DataOps CI/CD Workflow with Docker, Airbyte, dbt, BigQuery, and GCS

1. **Continuous Integration (CI)**: Developers push changes to data processing
   scripts, dbt models, or other code to a version control system like Git.

2. **Build**: An automated build system, such as Jenkins or GitLab CI/CD,
   detects the changes and triggers a build:

    - If there are changes to data processing scripts, the **Airbyte Docker
      Container** can be built with the new code, connecting to various data
      sources, extracting data, and loading it into a staging area in BigQuery.

    - If there are changes to dbt models, the **dbt Docker Container** can be
      built with the new models. dbt can then transform the data in the staging
      area, creating updated versions of your data models in BigQuery.

    - Docker ensures these containers are built consistently, with the same
      dependencies and environment, regardless of where they are run.

3. **Test**: Automated tests are run on the transformed data to validate the
   changes. Tests can be written in dbt to ensure data quality and consistency.
   Docker ensures that these tests are run in an environment that mirrors the
   production environment.

4. **Continuous Delivery (CD)**: If the tests pass, the updated containers are
   pushed to a Docker registry, from where they can be pulled and deployed to
   the production environment.

    - The Airbyte container runs the extraction and load processes, populating
      the staging area in BigQuery.

    - The dbt container transforms the data, updating the data models in
      BigQuery.

5. **Reporting and Analysis**: Business intelligence tools connect to BigQuery
   and use the updated data models for reporting and analysis.

6. **Data Backup**: The processed data in BigQuery can be exported to GCS for
   backup and long-term storage.

7. **Monitoring and Re-processing**: The performance and health of the data
   pipeline can be monitored. If issues are detected, or if there are
   significant changes to the data, the CI/CD pipeline can be triggered again to
   reprocess the data using Docker to rebuild the necessary containers.

## MLOps
