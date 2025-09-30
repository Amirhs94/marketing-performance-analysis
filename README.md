
# Project: Marketing Channel Performance Analysis

## Objective
The Parts Avatar marketing team has allocated its budget across three primary channels for the first half of the year: Pay-Per-Click (PPC), Email Marketing, and Social Media advertising. The team needs a comprehensive analysis to understand the effectiveness and return on investment (ROI) of each channel.

Your goal is to ingest and integrate data from these disparate sources, calculate key performance indicators (KPIs), and deliver a clear report with actionable recommendations on where to focus future marketing spend.

## The Challenge
The data for this analysis is fragmented across four different files, each representing a piece of the customer journey. Your primary challenge is to create a single, unified view of marketing performance. You must decide how to join these datasets, handle any potential discrepancies, and calculate meaningful metrics that will drive business decisions. The conceptual part of this task is to define what "performance" means and to tell a clear story with the data.

## Datasets
* `data/ppc_spend.csv`: Daily spend on PPC campaigns.
* `data/email_campaigns.csv`: Data on email campaigns, including number of emails sent and clicks generated.
* `data/social_media_ads.csv`: Daily spend and performance data for social media ads.
* `data/website_conversions.csv`: A log of all sales conversions, including revenue and the marketing channel that sourced the customer.

## Your Tasks
1.  **Data Ingestion & Integration (ETL):**
    * Write a Python script (`src/process_data.py`) to load, clean, and merge the four data sources into a single, aggregated dataset. This dataset should be structured to allow for daily or weekly performance analysis per channel.

2.  **KPI Calculation:**
    * Using your integrated dataset, calculate the following key metrics for each marketing channel:
        * Total Spend
        * Total Clicks & Total Conversions
        * Total Revenue
        * Click-Through Rate (CTR, for social media)
        * Conversion Rate (Conversions / Clicks)
        * Cost Per Click (CPC)
        * Cost Per Acquisition (CPA, or cost per conversion)
        * Return on Investment (ROI) or Return on Ad Spend (ROAS)

3.  **Analysis & Insights:**
    * In a Jupyter Notebook or as part of your report, analyze the calculated KPIs to answer these questions:
        * Which channel has the highest ROI?
        * Which channel is the most and least expensive for acquiring a customer (CPA)?
        * Which channel is most effective at converting clicks into sales?
        * Is there any observable trend in performance over time for any of the channels?

4.  **Visualization & Recommendations:**
    * Create a summary dashboard or a set of clear visualizations (using Matplotlib, Seaborn, Plotly, etc.) that effectively communicate the performance of each channel.
    * Conclude your analysis in this README with a summary of your findings and provide a data-driven recommendation to the marketing team on how they should consider allocating their budget for the next quarter.

## Evaluation Criteria
* **Data Integration Logic:** Your approach to cleaning and merging the different data sources into a coherent model.
* **Accuracy of KPIs:** Correct calculation of standard marketing performance metrics.
* **Analytical Thinking:** Your ability to interpret the KPIs and extract meaningful, non-obvious insights from the data.
* **Communication & Visualization:** The clarity of your charts and the persuasiveness of your final recommendations.

## Disclaimer: Data and Evaluation Criteria
Please be advised that the datasets utilized in this project are synthetically generated and intended for illustrative purposes only. Furthermore, they have been significantly reduced in terms of sample size and the number of features to streamline the exercise. They do not represent or correspond to any actual business data. The primary objective of this evaluation is to assess the problem-solving methodology and the strategic approach employed, not necessarily the best possible tailored solution for the data. 


------------------------------------------------------------------------------
Summary of findings

ROI Leader – Email Marketing

       Email campaigns generated strong conversion volumes with nearly zero direct spend (since it leveraged an existing subscriber list).

       This channel delivered the highest Return on Investment (ROI), essentially outperforming paid channels on cost-effectiveness.

Cost Efficiency – Among Paid Channels

       When excluding Email’s $0 cost, PPC proved to be the cheapest channel for acquiring a customer (lowest CPA).

       Social Media ads had the highest CPA, meaning it was the most expensive way to win a conversion.

Click-to-Sale Effectiveness

       Email had the highest Conversion Rate (CVR) — converting a large share of clicks into actual sales.

       Between the two paid channels, PPC out-converted Social Media, indicating PPC’s traffic is more purchase-ready.

Revenue Contribution

       PPC contributed the largest share of paid-media revenue, supported by both solid CVR and relatively low CPA.

       Social Media drove awareness and clicks but showed weaker conversion efficiency and ROI.

Trend Over Time

       Weekly trends for all channels were relatively steady with only mild fluctuations — no sharp rise or decline was observed in ROAS or CVR.

       This suggests performance is in a steady-state and predictable heading into the next quarter.

****Recommendations for Next Quarter:

 Prioritize Budget Efficiency

       Increase investment in PPC: it delivers better ROI and lower CPA than Social Media; scaling it is likely to drive more cost-effective sales.

       Sustain Email efforts: even small improvements in targeting or creative could yield high-margin gains since its variable cost is near zero.

 Re-evaluate Social Media Spend

       Social Media appears to be underperforming on CPA and ROI.

       Either optimize the creative/audience mix to improve conversions or consider reallocating part of its budget to PPC where returns are stronger.

Maintain Consistency in Messaging

       Since trends are stable, incremental optimizations (ad copy testing, better audience segmentation) rather than drastic channel shifts will likely be most effective.

Monitor & Test

       Track weekly KPIs for any trend changes after budget reallocation.

       Continue A/B testing in Social Media to discover higher-converting formats before cutting spend entirely.

Final Quote:

       Keep Email strong, grow PPC share, trim or optimize Social Media.

       This mix maximizes return on every marketing dollar spent while preserving healthy conversion efficiency.
