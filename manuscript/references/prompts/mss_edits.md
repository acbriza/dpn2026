## Methodology
1. Update the methodology section so that it is faithful to the experiments done. Specifically, running the pipeline using the yml configs in the `experiments/` folder

2. Create illustrations of the following and add to `main.tex` (the illustrations can be placed in `documentation/illustration` folder) 
	- general pipeline
	- repeated stratified k-fold and optimization
	- counterfactual generation

You may find these files helpful: `nested_cv_optimization.md` and `nested_cv_optimization.svg` in `manuscript/illustration` 

3. based on the experiment design explain clearly and emphasize:
- the prevention of data leakage 
- how the small dataset was handled; explain why SMOTE was not used and deferred to class imbalance handling in CatBoost.
- preempt the possible confusion of "4 models" with "4 experiments," raising questions about which result to trust. Emphasize: _one modeling pipeline evaluated via 4-fold CV, yielding aggregated performance metrics_, not four separate studies.

4. The following tables and figures will be presented in the Discussion section. Make sure that the methodology covers the details necessary for discussing them:
- manuscript/references/hyperparameter_optimization/optimization_metrics_ci.csv 
- manuscript/references/selection/summaries/auprc_summary_table_mean.png
- manuscript/references/selection/violins/All/auprc_violin.png
- manuscript/references/explainability/catboost_all_splits_auprc.png
- manuscript/references/explainability/catboost_all_splits_feature_importances.png
- manuscript/references/explainability/catboost_all_splits_shap.png
- manuscript/references/explainability/catboost_pooled_auprc.png
- manuscript/references/explainability/catboost_pooled_calibration.png
- manuscript/references/explainability/catboost_pooled_dca.png

Make a plan for executing these tasks and confirm with me first before executing 

## Results and Discussion

*Provide Clear Framing/Storyline/Clinical Interpretation*
- Clearly frame as decision-support research, not deployment-ready tool
- Mention that it is probably the first study to systematically integrate counterfactual explanations for actionable DPN screening in a low-resource clinical setting
- Clearly maintain that the counterfactuals is the main contribution
   - clinical actionability (HbA1c, CKD, dyslipidemia)
   - avoid any claim of clinical readiness

*Strong discussion on Clinical Actionability of Counterfactual Explanations*
- strong discussion on  counterfactuals, especially on plausibility of generated counterfactuals
- include an illustration on how clinicians could use this model. 
	- create an illustration for it in png and svg format and add in references/illustrations
- clarification of the DiCE genetic algorithm configuration and why only 10 of 53 candidates produced usable counterfactuals

*Clearly acknowledge the following limitations*
- The data comes from one clinic or source
- Proof-of-concept decision support model
- Deployment risks 
- small dataset
- there no external validation
- possible selection bias 
- lack of prospective testing

Make a plan for executing these tasks and confirm with me first before executing 


## Conclusion, Recommendation, Abstract

*Include in Recommendation*
- Need for larger, multicenter validation
- Future work should assess model performance across sites with varying prevalence rates to characterize  sensitivity to shifts in case mix.

Make a plan for executing these tasks and confirm with me first before executing 


## Quotation Sourcing
The file `references/quotables/quotables.md` contains a list of claims that I am considering to include in the manuscript. 
Some of them already indicate a possible source (author, year). Determine the actual sources for them. 
The others are claims I want to include in the discussion and wish to strengthen by citing studies that support them.
In your search, give priority to sources coming from `BMC Medical Informatics and Decision Making`, since that is my target journal.

Whenever possible, download the actual document (pdf preferred) and store them in `manuscript/rrl` with the format `year_author_title.pdf`. 
Note that some claims may be present in the same document or research. Thus, check first the downloaded documents to keep the number of downloaded documents to a minimum.
Only 1 source is required per claim.

Finally, make a summary in markdown format. 
It should have the following information: 
- the claim
- verdict (e.g. accurate, unsupported, contrary)
- snippet from the document that supports validates claim (if accurate) OR the corrected interpretation (if not accurate)
- source document (if found, and hyperlinked to actual document if downloaded) and page 
- bibtex entry from Google Scholar

Make a plan for executing these tasks and confirm with me first before executing 

*Follow up prompt*
- create a git branch called codereviewed_methods and switch to it.  All changes in this plan will be committed to this branch
- Save the plan to documentation/notes/methods_edit.md. Include my exact prompt for this session at the start of  methods_edit.md. commit methods_edit.md
- Execute the plan in auto-accept mode, but commit per task accomplished.

## Claims Review
Review the entire document. 
- For cited sources:
	- Verify that claim in the citation is accurate.
	- Verify that entry in `bibfile.bib` is accurate by comparing it with the entry in Google Scholar
- For strong and important claims in the text without citations
	- Search the web for possible sources that support this claim.
	- In your search, give priority to sources coming from BMC Medical Informatics and Decision Making, since that is my target journal.
	- Whenever possible, download the actual documents (pdf preferred) and store them in `manuscript/rrl` with the format `year_author_title.pdf`. 

Make a summary of this session in markdown format and save it as `citations.md` in the `manuscript/rrl` folder

Make a plan for executing these tasks and confirm with me first before executing 

