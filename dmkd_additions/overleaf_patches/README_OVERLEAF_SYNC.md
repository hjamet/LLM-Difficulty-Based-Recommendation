# Overleaf Synchronization Guide (`dmkd_additions/overleaf_patches/`)

This directory contains the revised LaTeX source files for the **Data Mining and Knowledge Discovery (DMKD)** Major Revision of *"Leveraging LLMs for Written Text Difficulty Estimation and Simplification in Reading Recommendation Systems"*, with all substantive additions highlighted in `\textcolor{red}{...}` for peer review.

Once co-author validation (Prof. Michalis Vlachos) is complete, copy each file below into the Overleaf repository clone (`paper/`):

## File-by-File Mapping

1. `dmkd_additions/overleaf_patches/tex/Introduction.tex` $\rightarrow$ `paper/tex/Introduction.tex`
   - **Tasks covered**: `T16` (moderation of the language-agnostic claim on line 34), `T17` (in-text walkthrough and citation of `\autoref{fig:framework}`), `T21` (new subsection `\subsection{Extension Beyond the RecSys 2024 Short Paper}` with Table `tab:recsys_vs_dmkd`).

2. `dmkd_additions/overleaf_patches/tex/Difficulty_Estimation.tex` $\rightarrow$ `paper/tex/Difficulty_Estimation.tex`
   - **Tasks covered**: `T04` (citation of Martinc et al., 2021 and comparison of Accuracy, MAE/MSE, QWK, and $PM_{\text{norm}}$), `T29` (uncommented `\subsubsection{System Prompt Design}\label{subsubsec:system_prompt}`), `T01` & `T02` (numbered equation for $PM_{\text{norm}} \in [0, 1]$ and revised Algorithm 1 aligned with confusion matrix cells $C_{r,c}$), `T17` (explicit walkthrough of `\autoref{fig:pm_explanation}`).

3. `dmkd_additions/overleaf_patches/tex/Simplification.tex` $\rightarrow$ `paper/tex/Simplification.tex`
   - **Tasks covered**: `T05` & `T06` (numbered equations for semantic similarity $S$, harmonic $w\text{-Score}$, simplification accuracy $A$, generalized $K$-level formula $A_K$, and step-by-step 6-level numerical example), `T07` (analytical derivation of $w_1^* = 0.8055$ from CamemBERT cosine anisotropy floor $S_0 \approx 0.810$).

4. `dmkd_additions/overleaf_patches/tex/RS.tex` $\rightarrow$ `paper/tex/RS.tex`
   - **Tasks covered**: `T19`, `T20`, `T22` (ranking literature citations, numbered equations for item/user LLM embeddings with dimensions $d=768$ and $d=1536$ and mean pooling `\autoref{fig:init_embeddings}`, cosine KNN scoring `\autoref{fig:recommend_with_knn}`, LightGCN bipartite message passing and BPR loss `\autoref{fig:recommend_with_lightgcn}`).

5. `dmkd_additions/overleaf_patches/tex/UI.tex` $\rightarrow$ `paper/tex/UI.tex`
   - **Tasks covered**: `T15` (moderation of the p. 18 claim on sentence splitting and vocabulary replacement, cross-referencing the empirical and human evaluations in Section 6).

6. `dmkd_additions/overleaf_patches/tex/Experiments.tex` $\rightarrow$ `paper/tex/Experiments.tex`
   - **Tasks covered**: `T13` (Section 6 opening roadmap), `T08` (cross-entropy loss with label smoothing vs. ordinal evaluation), `T12` & `T03` & `T10` & `T11` (formal Accuracy and QWK definitions, 1,000-iteration bootstrap and paired Wilcoxon analysis including non-significance on `LjL`), `T29` (removal of duplicated system prompt block), `T07` & `T17` (in-text citation and analysis of `\autoref{fig:sensitivity_and_ranks}` and `\autoref{fig:pareto_finetuning}`), `T18` (fix `Evaluation Dataset:` heading typo), `T09` (clarification of iterative simplification step 3 comparing $x^{(t)}$ to $x^{(0)}$), `T14` (new subsection `\subsubsection{Human Evaluation of Simplification Quality}` with Split-and-Overlap protocol on $N=150$ pairs and Table `tab:human_eval_results`).

7. `dmkd_additions/overleaf_patches/tex/Experiments_RS.tex` $\rightarrow$ `paper/tex/Experiments_RS.tex`
   - **Tasks covered**: `T23` (formal mathematical definition of $\text{DCG}@5$ and $\text{NDCG}@5$, and Xavier uniform initialization equation vs. LLM-pretrained embedding warm-start), `T25` (fixed broken reference `\autoref{tab:best_hyperparameters}` $\rightarrow$ `\autoref{tab:hyperparameters}`), `T24` (three concrete architectural deployment guidelines in Section 6.3.4).

8. `dmkd_additions/overleaf_patches/tex/Conclusion.tex` $\rightarrow$ `paper/tex/Conclusion.tex`
   - **Tasks covered**: `T16` (moderation of the language-agnostic claim in Section 7), `T14` (updated limitations paragraph referencing the 150-pair double-blind human evaluation and future longitudinal classroom studies).

9. `dmkd_additions/overleaf_patches/figures/difficulty_estimation.tex` $\rightarrow$ `paper/figures/difficulty_estimation.tex`
   - **Tasks covered**: `T03`, `T10`, `T11` (updated Table 2 reporting $\mu \pm \sigma$ across $B=1,000$ bootstrap resamples, QWK, normalized $PM_{\text{norm}}$ in %, and paired Wilcoxon significance markers $^{**}, ^{***}$).

## Bibliography Note (`T26`, `T27`, `T28`)
When synchronizing with Overleaf, also add the two new BibTeX entries (`martinc2021supervised` and `jamet2024evaluation`) to `paper/bibliography.bib` and fix the malformed entries (`Lungu18` title braces, `als1` pages `263--272`, `camembert` title capitalization `{CamemBERT}`) as documented in Section 3 (`R2.22`) of `dmkd_additions/rebuttal/Response_to_Reviewers.tex`.
