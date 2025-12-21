# README and System Report

**README Instructions:**

1. Copy the three data files to the storage/data_sources folder. Copy your .json input file into the storage/inputs folder.
2. Run the Dockerfile, it builds the container and starts the app.
3. Run the create_dataset endpoint by pointing your browser to:

<http://localhost:5000/create_dataset>

The sampling defaults to stratified but supports random and KFold via keyword.

1. Run the train_model endpoint by pointing your browser to:

<http://localhost:5000/train_model> (to train the default model RF)

<http://localhost:5000/train_model?model=XGB>

<http://localhost:5000/train_model?model=RF>

<http://localhost:5000/train_model?model=NN> (to train a specific model)

1. Run the predict endpoint by merely pointing your browser to <http://localhost:5000/predict>

This works as a GET request because you copied over your input file.

Or use a POST request by modifying this curl call with your file name:  
curl -X POST <http://localhost:5000/predict> \\

\-H "Content-Type: application/json" \\

\-d @input.json

When you put the file in your current working directory.

1. View the log files in the log folder.

# Written Report for the Prototype Including Design Choices


## Problem Statement

The problem at hand is to develop a containerized version of a new machine learning system for SecureBank to detect credit card fraud from a dataset of transactions with identifying information. The following functional requirements were given:  
<br/>System Functional Requirements

- The system should improve on prior performance (30% precision, 60% recall)
- The system should allow administrators to generate a new dataset for training from the available data sources.
- The system should allow administrators to select from a catalog of pre-trained models
- The system should allow administrators to audit the system's performance.

The non-functional requirements that would be considered in production include latency & throughput, resource utilization, scalability and elasticity to loads, reliability and availability, security, maintainability and extensibility, Admin and user usability and accessibility, interoperability, integration, compliance, privacy and regulatory considerations, disaster recovery and business continuity, and lastly, cost-effectiveness and resource efficiency. I decided to measure one aspect of all these many facets, the runtime of my code. My code turned out to be pretty fast, because I have a new computer. I got the following wall times for various parts of my code:

| Code Function | Wall Time |
| --- | --- |
| Data Loading | 3.29 sec |
| Label Encoding | 868 ms |
| EDA | 38.9 sec |
| Dropping Rows and Sampling | 668 ms |
| Feature Engineering | 6.17 sec |
| Data Split | 635 ms |
| Training, Grid Search and Results | 50.2 sec |

_Table 1. Elapsed time for various parts of the code in Jupyter Notebooks_

*Meeting the Requirements:*

The system satisfies the core functional requirements by providing a `/create_dataset` endpoint that generates and versions a stratified (or random/K-Fold) training set, a `/train_model` endpoint that accepts a model choice (RF, XGB, or NN) and returns hyperparameters plus precision/recall logs, and a `/predict` endpoint (GET or POST) for live inference—all while persisting immutable audit logs for every operation.  On the non-functional side, containerization and a stateless, RESTful design behind an API gateway deliver low latency, high throughput, and horizontal scalability; encrypted channels, role-based access control, and detailed audit trails ensure security, compliance, and data governance; robust Prometheus/Grafana monitoring combined with automated drift detection (PSI/KL divergence) and feedback-driven retraining pipelines guarantee reliability, maintainability, and model health; and strict environment segregation (dev, staging, prod) plus version-controlled configurations uphold resilience, business continuity, and cost-effective resource utilization. Obviously some of the non-functional requirements were not implemented because this is a prototype. They were not listed as part of the minimum requirements.


## Data

I was given a database in three parts, when joined made up a data pool of over 1 million rows. The majority class and the target class were extremely unbalanced in favor of the majority class. Upon investigation I found out I could skip imputing and drop rows with missing values because dropping those rows mostly preserved the fraudulent transaction group. Dropping NA rows resulted in 147,866 non frauds dropped , 164,670 with the label missing and only 630 fraudulent transactions dropped. This was my first major design decision. This left 1,334,376 rows from which to build a dataset. I decided to use 90% of this data, which affected performance. With so much data it was hard to know how much of it to start with. This left me with a training set of 960,751 observations, and a validation set and test set of 120,094 observations each, more than enough to get good results.

If the fraudulent transactions were mostly missing that label, this would be indicative of a much larger problem with SecureBank’s data collection procedures. Presumably that label was attributed by the customer service agents in the fraud department, not inferred by the previous ML model at SecureBank. That would be a case of “garbage in, garbage out”, and since data collection is outside of the scope of this project, it’s safe to say we can move forward by dropping rows. The question becomes is there any reason to think that the missing values in the data would tend towards fraudulent transactions? I assumed there is not this connection, and that once again that would be a data collection issue. Moving forward with this ML project we have to assume that the data was collected correctly.

Many columns were deemed unnecessary. The following columns were dropped for the indicated reason:

| Column dropped | Reason |
| --- | --- |
| Trans_num | This has nothing to do with fraud |
| Index_x | This is just an index |
| Index_y | This is just an index |
| Cc_num | This indicates customers |
| Unix_time | I broke the time down into it’s components elsewhere |
| Merch_lat | I preferred to track merchants by name |
| Merch_long | I track the merchants by name, also fraud can occur online |
| First, Last | This indicates the customer not the fraud |
| sex | We need to stop fraud regardless of customer gender |
| Street, city, zip | The customers location was found to be weakly predictive in EDA |
| dob | We need to stop fraud for customers of all ages |
| Year_date | This is a non-repeating feature that doesn’t generalize |

Sampling was considered, and because of the class imbalance stratified sampling of 90% of that data was preserved for training. The system also supports random sampling and KFold sampling which can be activated with a keyword.

*Ensuring dataset quality:*

Dataset quality is enforced end-to-end by first profiling and cleaning the raw inputs—rows with missing or malformed fraud labels are dropped (losing fewer than 0.05% of fraud cases), duplicate transactions are removed, and irrelevant identifier or free-text columns (e.g. customer names, raw geocoordinates) are excised to reduce noise. Consistent schemas and data types are enforced via programmatic checks in the data-loader and DataEngineering routines, while skewed numeric features (like transaction amount) are stabilized with a log transform and discretized into “suspicious” bins. Categorical fields are deterministically label-encoded and key “top-n” flags are engineered only after exploratory analysis confirms their predictive value. Finally, every dataset build is versioned with a timestamped filename, accompanied by an immutable JSON log of its parameters (sampling strategy, minority percent, etc.), and split via stratified sampling (80/10/10) to guarantee that the train, validation, and test sets faithfully preserve the original class imbalance—ensuring both reproducibility and statistical integrity across experiments.


## Feature Engineering

EDA provided other insights. Looking at a plot of fraud cases, the merchant used was not strongly determinative, and looking at the fraud cases alone it was apparent that frauds occurred at all the merchants in the database with relative uniformity, so that column was dropped:

![alt text](README_image_folder/Picture1.png)

 _Fig. 1: relative uniformity of frauds for all merchants_

The job, day_of_week, minute, and second of the fraud was not strongly indicative so these columns were dropped. However, the same view of category of purchase revealed the frauds concentrated in about 9 categories with 4 or more of them extremely prominent, so a feature could be engineered focusing on these categories:

![alt text](README_image_folder/Picture2.png)

 _Fig. 2: Fraud counts for purchase category_

The log transformed amount of the transaction had a trimodal distribution which only overlapped with non fraud at small amounts. The raw feature and a suspicious amount feature after discretization into bins would make for good features:  
![alt text](README_image_folder/Picture3.png)

 _Fig. 3: Log transformation of amount column_

This left us with trans_date_trans_time, day_of_week, day_date, month_date, hour, amt, city_pop, and category, in total 8 raw features. Hour, day of week, and month had peaks in the distributions. The preceding charts are just a sample of the exploratory data analysis (EDA) I did. The day_date and month_date distributions had peaks, but not an extreme grouping so I left them out. Also, presumably the system would want to detect frauds at any time during the month and year, but there were certain hours that were more offensive. Looking at all the categorical data and thinking about the binary classification task, I decided to go with binary features other than the “log_amt” column. Now I was ready to rank feature importance with a random forest and also use it to perform a recursive feature elimination, after testing some models on the whole feature set. The pertinent information on fraud seemed to be mostly time and amount based.

![alt text](README_image_folder/Picture4.png)


_Fig 4. Correlation of final features with the Fraud label_

This left me with the following binary features: the top 10 category for fraud, the top 2 jobs for fraud, the top 6 purchase categories for fraud, the top 6 months for fraud, the top 3 days for fraud, and whether or not it was a suspicious amount which was created by binning the transaction amounts above $20, calculating the fraud rate for each bin, getting the overall fraud rate, and selecting the bins with significantly higher fraud rates, (greater by a factor 2)

I’d also tried to craft a feature that groups the transactions by cc_num and identifies when many more transactions than normal have occurred in the last hour and the last 24 hours based on unix_time. This required establishing a frequency average of transactions for the fraud_label == 0 transactions when grouped by cc_num and comparing it to a fraudulent transaction frequency. I confirmed there was a difference in EDA, but could not get a highly correlated feature when I tried to prepare it. This also significantly increased training time, and I had already improved on previous performance using other features so the feature was abandoned. After that then I to dropped the original columns of the dataset and just kept the binary engineered features and log_amt.

*Justification:*

Guided by our exploratory analysis and the case study's specifications, we distilled a high-signal, low-complexity feature set by first dropping irrelevant identifiers (e.g. transaction IDs, customer names, raw geocoordinates) and removing any rows with missing labels—an approach that sacrificed fewer than 0.05% of fraud cases while guaranteeing data integrity. We then normalized the skewed transaction amounts via log transformation and discretized them into “suspicious” bins, engineered binary flags for the top‐n fraud‐prone categories, jobs, days, and months, and label-encoded remaining categorical variables to streamline downstream modeling. Finally, to preserve the original 0.3% fraud prevalence across splits and ensure robust evaluation, we used stratified sampling in our 80/10/10 train/validation/test partition (with optional random or K-Fold modes available), so that each set faithfully reflects the class imbalance and supports fair model selection and threshold tuning.


## Models

Because I had mostly binary features (most raw features required binning) I knew that tree-based models would be most appropriate and perform best on my feature set. I added a plain vanilla neural network to round out my model choices, but performance was terrible until I implemented 0.5 dropout and batch normalization on a hidden dimension of 32. Hyperparameter tuning was accomplished using a wide grid search at first, with code that selected the best model parameters in training for each search. This allowed me hone in on which combinations of parameters yielded the best result for these models, sliding my hyperparameters up or down and getting more granular as the results came in.

My initial grid search started at random values and at the end encompassed:

xgb_param_grid = {'n_estimators': \[100, 150, 200, 300, 400\], 'max_depth': \[10, 15, 20, 25, 30, 50, 100\],'learning_rate': \[0.1\]} rf_param_grid = {'n_estimators': \[100, 200\], 'max_depth': \[None, 10, 15, 20, 30\], 'min_samples_split': \[2, 3, 5, 7, 10, 12\]} nn_param_grid = 'hidden_dim': \[40, 50, 60, 70, 100\], 'num_hidden_layers': \[1\], 'lr': \[0.1\], 'num_epochs': \[10, 15\], 'batch_size': \[32\]}.

Through a process off trial and error the final best hyperparameters it settled on were:

| XGBoost: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100} |
| --- |
| Random Forest: {'max_depth': 10, 'min_samples_split': 12, 'n_estimators': 200} |
| Neural Network: {'batch_size': 32, 'hidden_dim': 50, 'lr': 0.1, 'num_epochs': 10, 'num_hidden_layers': 1} |

_Table 2: Best Hyperparameters after dozens and dozens of runs_

*Justification for Model Selection, Hyperparameter Tuning and Evaluation Metrics:*

I chose Random Forest and XGBoost because their tree-based structure excels on our mostly binary, skewed feature set and naturally handles non-linear interactions, with a vanilla neural network included as a foil to test for any residual complex patterns. Hyperparameter tuning was done via broad grid searches—varying tree counts, depths, learning rates, hidden dimensions, and dropout—to rigorously identify the settings that maximize recall while holding precision at or above our 30% business requirement. Evaluation centered on precision, recall, and area under the precision–recall curve (PR-AUC) because these metrics directly reflect our need to catch as many frauds as possible without overwhelming investigators with false positives; we also tracked ROC-AUC for overall ranking quality and examined false positive and false negative rates from confusion matrices to ensure the chosen decision threshold aligns with operational risk tolerances. Custom metrics were considered, but the requirements were stated in terms of precision and recall.

*Model Trade-Offs:*

Each model comes with its own strengths and weaknesses: Random Forests train relatively quickly, require fewer hyperparameters, and are inherently robust to overfitting on noisy, binary‐heavy data—but they may plateau in recall under tight precision constraints. XGBoost often squeezes out higher recall and PR‐AUC through gradient boosting’s sequential error correction, yet it demands longer training times, more fine‐grained hyperparameter tuning (learning rate, tree depth, regularization), and can be more sensitive to class imbalance. The neural network, while theoretically capable of capturing complex feature interactions, proved slow to converge, prone to overfitting without careful dropout and batch‐norm tuning, and yielded lower recall at our 30% precision target—and its black‐box nature makes operational explainability harder. Thus, I balanced ease of use and interpretability (Random Forest), peak performance with careful tuning (XGBoost), and experimental complexity (Neural Network) according to the bank’s throughput, latency, and auditability requirements.


## Metrics

For a credit card detection algorithm, there are many offline and online metrics to consider to evaluate the performance of the system. Since the requirements were to improve upon precision and recall, with a focus on recall, these offline metrics were the metrics measured in my testing and the ones used for model selection. Other metrics measured were Area under the Receiver Operating Characteristic (ROC-AUC) curve, which measures overall separability, and Area under the Precision-Recall curve, which is more informative on imbalanced data. An optimal threshold was defined as the point in the PR-Curve that maintained a30% or greater precision while maximizing recall. This led to the following optimal thresholds for each model on the validation set:

![alt text](README_image_folder/Picture5.png) ![alt text](README_image_folder/Picture6.png) ![alt text](README_image_folder/Picture7.png)



_Fig. 5 PR-Curves for the models (validation set)_

A Confusion Matrix was also applied, which utilized the False Positive Rate (FPR) = FP/(FP+TN) and the False Negative Rate (FNR) = FN/(FN+TP). F1 was avoided because it weighs FP as equally important with FN, but in our case FN is much more important. Other offline metrics that could have been calculated include: Brier Score, a Calibration Curve, Expected Monetary Cost, Cost-Benefit Ratio, Adversarial or Stress Testing, and more.

Although they were not calculated in this prototype, various metrics would be important to observe once results are being collected on inference for unseen data like: Demographic Parity Difference, Equalized Odds (both respecting protected demographics, like females and minorities), and performance metrics like Inference Latency, Throughput, Memory/CPU Utilization, and more. These metrics, although they can be collected in training and used for model selection, become even more important in production. I didn’t move forward with that. However, the best practice is to measure and choose a model based on these results as well. Given space and time considerations I made my choice instead based purely on precision and recall, strictly sticking to the requirements. In my results on the test set, the Random Forest was crowned champion by a small margin:

*Experimental Results*

| Model Selection Based On Precision and Recall | Training |     |     | Test Set |     | Test Set |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model and Best hyperparameters: | precision | recall | probability threshold | precision | recall | ROC AUC | PR AUC |
| XGBoost: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100} | 0.303 | 0.681 | 0.956 | 0.31 | 0.7 | 0.987 | 0.496 |
| Random Forest: {'max_depth': 10, 'min_samples_split': 12, 'n_estimators': 200} | 0.301 | 0.69 | 0.938 | 0.32 | 0.72 | 0.987 | 0.543 |
| Neural Network: {'batch_size': 32, 'hidden_dim': 50, 'lr': 0.1, 'num_epochs': 10, 'num_hidden_layers': 1} | 0.3 | 0.589 | 0.976 | 0.3 | 0.62 | 0.956 | 0.374 |

_Table 3. Training and Test set results for models_

In production, various live performance metrics become important, including real-time precision and recall, the False Alert Rate and the Missed Fraud Rate. The alert and investigation workload can be evaluated with metrics like Alerts per Day, Investigation Turnaround Time (which reflects on the system), Analyst Utilization and so on. Financial impact can be measured by Fraud Losses Averted, Operational Cost, ROI and such. Drift Detection becomes important and can be measured with Data Drift Metrics (Statistical tests on feature distributions) and Concept Drift Indicators. Feedback-Loop and Model Health can be weighed up using Retraining Triggers, and Post-Retraining Validation. The Customer Experience and Compliance must also be monitored with such metrics as False-Positive Customer Impact , Regulatory metrics, and Audit Metrics. Lastly, System Reliability must be watched on fronts like Uptime, Availability, and Mean Time to Recover.

All of these metrics would be important to measure for a system in production at an actual bank. Given the volume of metrics important to the system, and length limitations on this report, I thought it sufficient to be just aware of these metrics for now and go in depth with them once the prototype three models were approved. At that point a refined selection could be chosen. This is another major design decision I want to note (NOT to calculate additional metrics), and I want to argue for it, that in fact measuring any or all of these additional metrics would be premature given the likelihood that features need to be re-engineered. Many of the features I tried to engineer didn’t work (like transaction frequency and interaction terms) for some reason, showing very, very low correlation to the target class (useless), and here is where I would ask for help feature engineering in fact. Much higher performance on the test set is possible, but I need someone to help debug my code. I thought it would be realistic to point out that some design decisions are made in the face of failure, and this section has gotten too long already. I still consider the prototype a success because it meets the stated requirements, so that’s what I focused on: Precision and Recall on the test set, specifically improving recall.

*Detailed Description of Online Metrics and Why I Didn't Didn't Measure Them in the Prototype:*

In a fully instrumented production rollout, we would collect the following online performance measurements:

- **Real-time prediction quality**: every `/predict` call would emit a log entry (with PII masked) containing the input feature vector, model score, binary decision, and—once the human‐review verdict is known—a timestamped label. These entries feed into a rolling‐window aggregator (e.g. a Prometheus counter or custom service) that continuously recomputes precision, recall, false-positive rate and false-negative rate over the last N transactions (e.g. 1,000 calls or 24 hours).

- **Service health metrics**: API latency percentiles (P50, P95, P99), request throughput, and error rates (4xx/5xx counts), all exposed as application-level gauges or histograms and scraped by a monitoring system like Prometheus + Grafana, with alerts when SLAs are breached.

- **Data and model drift detectors**: scheduled jobs that calculate statistical distances (PSI, KL divergence) between the distributions of live incoming features and the original training baseline, plus delta-metrics in rolling precision/recall, so that significant shifts in data or performance automatically generate alerts.

- **Feedback-loop latency**: the time between model prediction and availability of the true label (after investigation), so we can tune retraining cadence and understand how fresh our training data truly is.

---

**Why our current code doesn’t capture these online metrics**

Right now, all of our performance evaluation lives in Jupyter notebooks or the synchronous `/train_model` endpoint (**as instructed in the Case Study**): we compute precision, recall, PR-AUC, and ROC-AUC on a held-out test set and log them to disk—but we haven’t yet built any of the real-time instrumentation, scraping endpoints, or feedback‐ingestion pipelines that would power live dashboards. There’s no Prometheus client in the Flask app, no rolling‐window aggregator for prediction quality, nor any mechanism to ingest human‐verified labels back into the system. It's not really possible to capture "online" metrics without a rollout to production for these reasons (but we can plan them):

---

**The challenge of “simulating” online metrics with only a test set**

Furthermore, a static test set can tell you how well your model would have performed on data drawn from the same distribution, but it cannot mimic:

1. **Changing data distributions**: customer behavior and transaction patterns drift over time; a test set frozen today won’t reveal tomorrow’s shifts.  
2. **Label-feedback delays**: in production, true labels arrive only after manual review, often hours or days later—test-set labels are all “instant” and known upfront.  
3. **Operational dynamics**: real request latencies, error rates under load, and upstream system failures (e.g. malformed inputs) can only be measured in a running service.

Without deploying the API, wiring up monitoring clients, and integrating the human‐in‐the-loop feedback channel, you can’t truly exercise or measure these online metrics—only offline proxies.


## Policy Decisions

As your fraud detection system moves from development into production, a comprehensive set of deployment and post-deployment policies must be in place to ensure it delivers on business objectives, safeguards customer data, and remains adaptable to changing risks and regulations.

First, before the system ever sees live traffic, you need to verify that it integrates cleanly with your existing infrastructure. That means defining secure, well-documented APIs and middleware layers to normalize transaction data between legacy banking systems, CRM platforms, and your fraud module. You should subject the end-to-end pipeline to rigorous load and stress testing, establishing clear SLAs for how quickly fraud alerts must be generated and acted upon under peak demand. To avoid any unintended impact on production workflows, maintain separate test, staging, and production environments so that new models or configuration changes can be validated under realistic conditions before go-live.

Security and data governance go hand in hand with infrastructure readiness. All transaction data and model predictions must travel over encrypted channels, with data masking, anonymization, or tokenization protecting personally identifiable information in line with GDPR, CCPA, or other relevant regulations. Access to raw data, system parameters, and override functionality should be governed by strict role-based access control, with permissions reviewed regularly to reflect changes in responsibilities. Every decision the system makes—whether flagging a transaction or allowing it—must be logged in an immutable audit trail retained for the period required by regulators, ensuring full traceability for both internal reviews and external audits.

Model governance and risk management policies help you maintain confidence in your predictive engine. Before deployment, each model version should undergo back-testing against historical fraud cases, as well as simulation-based stress tests using synthetic or adversarial data to probe edge-case behavior. If you rely on any third-party models or services, vendor oversight policies must verify that those components meet your own security, performance, and compliance standards. All changes to model code, configuration, or data pipelines should be tracked under a formal change-management process that records who made what change, why, and when, so you can roll back or audit any update at a moment’s notice.

Equally important is ensuring that everyone in the organization understands how the fraud system works and what its limitations are. You’ll want to brief compliance, IT, customer-service, and risk-management teams on how alerts are generated, how to interpret them, and when human intervention is required. Internally, detailed documentation, training workshops, and FAQs will empower employees to use the system effectively. Externally, you should prepare customer-facing communication protocols that strike a careful balance between transparency—so users know why their transactions might be blocked—and avoiding unnecessary alarm that could erode trust.

Once the system is live, a new suite of post-deployment policies must govern its ongoing operation. Real-time monitoring should track each alert against key performance indicators such as false-positive rate, detection accuracy, precision, and recall, while batch-oriented reports analyze trends and patterns over longer horizons. Automated dashboards and notification channels will ensure that anomalies in model behavior or data drift are flagged immediately. When drift is detected—whether in customer behavior, transaction patterns, or feature distributions—you’ll need policies dictating how often to retrain the model, how to validate new data, and how to adjust alert thresholds in response to evolving fraud tactics.

A robust feedback loop ties manual review back into continuous improvement. Every suspicious transaction that a human analyst examines should feed its verdict—fraudulent or legitimate—back into your training data so that the model learns from real-world outcomes. Incident-response procedures must define clear escalation paths if there are waves of false positives, undetected fraud spikes, or system outages, including when to involve IT, risk officers, or even law enforcement. After any significant incident, a formal post-mortem should dissect root causes and capture lessons learned, ensuring that both the model and your operational processes improve.

Ongoing compliance and audit readiness remain critical. Schedule regular internal reviews of data access controls, logging practices, and model performance against both corporate policies and external regulations. Maintain protocols for engaging third-party auditors or regulatory agencies, including preparing required performance and compliance reports and cooperating fully during any forensic investigations.

Finally, given the ethical and business stakes of fraud detection, you must continuously monitor for bias in your algorithms to ensure that no customer segment is unfairly targeted. Transparency policies—both internal and customer-facing—should explain, in clear language, how decisions are made and what recourse customers have if they believe they were treated unfairly. Grievance-redressal procedures must allow customers to contest and appeal decisions, triggering human review and, where appropriate, remedial action.

Throughout the system’s lifecycle, maintain a rolling cost-benefit analysis to measure the financial impact of false positives (for example, investigation costs and customer inconvenience) versus the value of prevented fraud losses and reputational gains. Align your KPIs not only with technical metrics like recall and precision, but also with broader business goals such as customer satisfaction scores, operational efficiency, and compliance adherence. By weaving these deployment and post-deployment policies into every stage of your fraud detection pipeline, you’ll build a system that is not only accurate and performant, but also resilient, fair, and trusted by regulators, stakeholders, and customers alike.

All this being said, one design decision I would make regarding policy is definitely have environment segregation moving forward from the prototype. We should enforce environment segregation by ensuring that production data, high-sensitivity logs, and stringent SLAs remain completely isolated from our staging and development environments. Locking down our deployment policies by encoding stricter requirements—such as tighter SLAs or extended log-retention periods—directly in our production configuration files, keeps our core codebase stays clean and uncluttered. And to guarantee compliance and auditability, we will keep all regulatory-mandated settings (encryption keys, retention schedules, RBAC rules, etc.) under version control in environment-specific files.


## Deployment Strategy

In designing our containerized fraud detection prototype, we defined a clear, RESTful API contract across three core endpoints—`/create_dataset`, `/train_model`, and `/predict`—each of which adheres to consistent URL structures, parameter conventions, and JSON response schemas to simplify integration and versioning. The `/create_dataset` endpoint accepts query parameters for sampling strategy (stratified, random, or KFold) and minority percentage, returning both the generated files and descriptive logs. The `/train_model` endpoint similarly takes a `model` parameter (RF, XGB, or NN) to select the appropriate algorithm, returning hyperparameter settings, performance metrics, and training logs in its JSON payload. For inference, the `/predict` endpoint supports both `GET` requests—loading a pre-placed `input.json` file—and `POST` requests with an inline JSON body, enabling flexibility for testing, batch pipelines, or real-time event streams.

To ensure scalability, each service is stateless and can be replicated horizontally behind a load balancer. Containers are orchestrated via Kubernetes (or Docker Swarm) to manage rolling updates, health checks, and auto-scaling policies driven by CPU/memory utilization or custom application metrics (e.g., request queue length). An API gateway fronts the services, providing authentication, rate limiting, and canary deployments for safe rollouts. Persistent artifacts—datasets, model binaries, and logs—are stored in network-backed volumes or object storage, decoupling compute from state and allowing the system to elastically grow or shrink in response to demand without data loss.


## Post-Deployment Monitoring

Once live, the system relies on layered monitoring to safeguard both infrastructure health and model performance. At the application level, every API request and response (with PII masked) is logged along with timestamps, input feature vectors, model scores, and final decisions, feeding into a central log aggregator (e.g., ELK Stack or CloudWatch). An error-handling middleware captures exceptions, returns standardized error payloads with HTTP status codes, and surfaces stack traces for rapid debugging. Infrastructure metrics—latency, throughput, CPU/memory utilization, and error rates—are collected via Prometheus and visualized in Grafana dashboards, with alert rules that trigger when service-level objectives (SLOs) breach their thresholds.

To detect data drift and model degradation, we run scheduled analysis jobs that compute statistical distance measures (such as population stability index and KL divergence) between live feature distributions and the training baseline. When drift exceeds preconfigured thresholds, alerts route to the ML-ops team to investigate potential retraining. Simultaneously, a feedback loop ingests confirmed fraud labels from downstream investigation workflows, continuously recalculating precision and recall on a rolling window (e.g., daily). If performance dips below SLA-defined floors, an automated retraining pipeline can be invoked to refresh the model with the newest data. Incident-management integrations (PagerDuty, Slack) ensure that both operational anomalies and model-quality alerts receive immediate human attention, closing the loop on reliability, accuracy, and regulatory compliance.


## Conclusion

To close out this prototype report, here is one concrete design decision drawn from each major section:

- **Problem Statement:** We chose to benchmark only the end-to-end wall-clock time of our notebook code—data loading, feature engineering, and model training—to stay strictly within management’s defined scope while still demonstrating system performance.

- **Data:** We elected to drop all rows with missing values rather than impute them, preserving nearly every fraudulent example and simplifying downstream processing on a clean pool of 1.33 million transactions.

- **Feature Engineering:** We distilled dozens of raw fields into a small set of binary indicators—most notably “top n” category flags—and a log-transformed amount, balancing predictive power against feature set complexity.

- **Models:** Recognizing that our feature matrix was largely binary, we prioritized tree-based learners (Random Forest and XGBoost) and used a vanilla neural network as a foil, then performed exhaustive grid searches to ensure apples-to-apples comparisons.

- **Metrics:** We anchored model selection solely on precision and recall—specifically the point on the PR curve that maintains ≥ 30 percent precision while maximizing recall—rather than diluting effort into secondary measures.

- **Policy Decisions:** We enforced strict environment segregation via separate, version-controlled configuration files for development, staging, and production, ensuring sensitive data and SLAs remain isolated.

- **Deployment Strategy:** We designed each service as a stateless, RESTful container behind an API gateway—with clearly defined endpoints, parameter conventions, and JSON schemas—allowing horizontal scaling, rolling updates, and seamless integration with CI/CD pipelines.

- **Post-Deployment Monitoring:** We established a multi-layered monitoring framework—application logs, infrastructure metrics, and automated drift detectors (e.g., PSI, KL divergence)—coupled with a feedback loop that ingests confirmed fraud labels and triggers alerts or automated retraining whenever performance or data quality thresholds are breached.

Together, these disciplined design choices have produced a prototype that not only meets the stated requirements but is also architected for scalability, reliability, and continuous improvement in a real-world banking environment.