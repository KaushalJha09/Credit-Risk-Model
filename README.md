# 🏥 SaveIN Credit Risk Model
### Early Delinquency Prediction & Intelligent Underwriting Strategy

## 📋 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Highlights](#-key-highlights)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Critical Findings](#-critical-findings)
- [Underwriting Strategy](#-underwriting-strategy)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project develops a **production-ready credit risk model** for SaveIN, a healthcare lending platform that provides instant financing at point-of-care. The solution identifies customers likely to become delinquent early in their loan tenure (by Month-On-Book 3), enabling proactive risk management and optimized approval strategies.

### Why This Matters
- **Healthcare Access**: Enables fair, data-driven lending decisions for medical financing
- **Portfolio Health**: Maintains <5% bad rate while achieving 60%+ approval rates
- **Early Warning**: Detects delinquency risk within 3 months (vs. traditional 6-month horizon)
- **NTC Inclusion**: Special framework for New-to-Credit customers (20% of portfolio)

---

## 🎯 Problem Statement

**Objective**: Build a credit risk model to identify customers who are likely to go delinquent early in their loan tenure. The model must answer:

1. **Which customers should be approved?**
2. **At what risk stance?** (credit limits, pricing, terms)

### Business Context
SaveIN uses credit bureau data (CRIF) along with demographic and income signals to predict early-stage delinquency and optimize approval/limit strategies. The challenge: **20% of customers are New-to-Credit (NTC)** with no bureau history.

---

## ✨ Key Highlights

### 🏆 Model Performance
- **Test AUC**: 0.7131 (Gini: 0.4261)
- **KS Statistic**: 0.3619
- **Selected Model**: Logistic Regression (beats tree models due to superior generalization)
- **Training Approach**: SMOTE for class balancing, stratified train-test split

### 🔍 Critical Discovery
**NTC customers require a different approach:**
- NTC Bad Rate: **22.56%** (vs 6.11% for Non-NTC)
- Model AUC on NTC: **0.5752** (weak predictive power)
- **Solution**: Two-tier strategy (ML for Non-NTC, rules-based for NTC)

### 💡 Innovation
- **Top Predictor**: Bureau-Income Interaction (engineered feature)
- **40+ Features**: Including tradeline aggregations, utilization metrics, risk ratios
- **Smart Threshold**: 0.50 balances growth (62.7% approval) with quality (4.86% bad rate)

---

### Key Statistics
| Metric | Value |
|--------|-------|
| Total Customers | 10,000 |
| Tradeline Records | 32,774 |
| NTC Customers | 2,000 (20%) |
| Positive Class Rate | 9.64% (MOB3_Bad) |
| Features Engineered | 40+ |

---

## 🔬 Methodology

### 1️⃣ Target Selection

**Selected: MOB3_Bad** (30+ DPD by Month 3)

| Target | Prevalence | Decision |
|--------|-----------|----------|
| MOB2_Bad | 4.72% | ❌ Too early, insufficient pattern |
| **MOB3_Bad** | **9.64%** | ✅ **SELECTED** - Early warning + actionable |
| MOB6_Bad | 4.45% | ❌ Too late for intervention |

**Justification**:
- ✅ Industry standard (30 DPD = official delinquent status)
- ✅ Early detection enables proactive intervention
- ✅ Strong indicator: 30.5% of MOB3_Bad → MOB6_Bad
- ✅ Balanced prevalence for modeling

### 2️⃣ Feature Engineering

```

#### Engineered Features (Top Performers)
1. **bureau_income_interaction** ⭐ (Strongest predictor)
2. **debt_to_income** (Total debt / Annual income)
3. **loan_to_income** (Requested loan / Monthly income)
4. **emi_to_income** (EMI burden analysis)
5. **is_ntc** (New-to-Credit flag)
6. **device_ntc_interaction** (Alternative risk signal for NTC)

#### NTC Handling Strategy
```python
# Identification
is_ntc = (bureau_score.isna()) | (bureau_vintage_months.isna()) | (total_tradelines == 0)

# Feature Imputation
- Tradeline features → 0 (no history)
- Bureau score → Median for Non-NTC (710)
- Created device_score × is_ntc interaction
```

### 3️⃣ Model Development

#### Models Tested
| Model | Train AUC | Test AUC | Overfitting Gap | Decision |
|-------|-----------|----------|-----------------|----------|
| **Logistic Regression** | 0.7230 | **0.7131** | 0.0099 | ✅ **SELECTED** |
| Random Forest | 0.8809 | 0.6735 | 0.2074 | ❌ Overfitting |
| XGBoost | 0.9364 | 0.6604 | 0.2760 | ❌ Overfitting |
| LightGBM | 0.9273 | 0.6614 | 0.2659 | ❌ Overfitting |

**Why Logistic Regression?**
- Minimal overfitting (0.0099 gap)
- Better generalization to unseen data
- Interpretable for regulatory compliance
- Stable performance across segments

### 4️⃣ Threshold Optimization

Tested thresholds from 0.1 to 0.85, analyzing:
- Approval rate vs. bad rate trade-off
- Business profitability (revenue - losses)
- Operational efficiency (auto-approval %)

**Selected Threshold: 0.50**
- Approval Rate: 62.7%
- Bad Rate: 4.86%
- Balances growth with portfolio quality

---

## 📈 Model Performance

### Overall Metrics
```
Test AUC:       0.7131
Gini:           0.4261
KS Statistic:   0.3619
Precision:      0.1769 (at 0.5 threshold)
Recall:         0.6839
F1-Score:       0.2812
```

### Segment-Wise Performance

| Segment | AUC | Bad Rate | Avg Score | Approval @ 0.5 |
|---------|-----|----------|-----------|----------------|
| Non-NTC (78.5%) | 0.6216 | 6.11% | 0.358 | 79.4% |
| NTC (21.5%) | 0.5752 | 22.56% | 0.719 | 1.9% |

### Feature Importance (Top 10)
```
1. bureau_income_interaction    1.028 ⭐
2. income_monthly              0.983
3. is_ntc                      0.578
4. bureau_score_filled         0.305
5. age                         0.249
6. device_score                0.235
7. bureau_score_band           0.203
8. age_band                    0.196
9. debt_to_income              0.192
10. total_sanctioned_tl        0.172
```

---

## 🚨 Critical Findings

### 1. NTC Model Limitation
**Problem**: Model AUC of 0.5752 for NTC customers (barely better than random)

**Impact**:
- At threshold 0.5, only 1.9% of NTC customers approved
- Model essentially auto-rejects all NTC customers
- 22.56% actual bad rate vs 6.11% for Non-NTC

**Solution**: Two-tier underwriting strategy

### 2. Risk Concentration
- Decile 10 has 26% bad rate (vs 2.5% in Decile 1)
- High-risk customers (score ≥ 0.7) account for 15% of portfolio
- Clear risk separation validates model effectiveness

---

## 🎯 Underwriting Strategy

### Risk-Based Approval Tiers

| Risk Tier | Score Range | Decision | Credit Limit | Interest Rate | Expected Bad Rate |
|-----------|------------|----------|--------------|---------------|-------------------|
| **Low** | < 0.20 | AUTO-APPROVE | Up to ₹150K | 16-18% | <3% |
| **Medium-Low** | 0.20-0.30 | AUTO-APPROVE | Up to ₹100K | 18-19% | 3-5% |
| **Medium** | 0.30-0.40 | APPROVE | Up to ₹75K | 19-20% | 4-6% |
| **Medium-High** | 0.40-0.50 | MANUAL REVIEW | Up to ₹50K | 22% | 10-15% |
| **High** | ≥ 0.50 | REJECT | ₹0 | N/A | >20% |

### Two-Tier Approach

#### Non-NTC Customers (78.5% of portfolio)
```
✅ Use ML model with threshold 0.50
✅ Expected approval: ~80%
✅ Expected bad rate: ~5%
✅ Automated decision making
```

#### NTC Customers (21.5% of portfolio)
```
⚠️ Model is weak (AUC 0.58) → Use rules-based scorecard

Approval Criteria (must meet ALL):
├── Income: ≥ ₹25,000/month
├── Employment: Salaried (preferred)
├── Device Score: ≥ 0.6
├── Loan-to-Income: < 6x
└── Residence: Owned/Family preferred

Credit Limits:
├── 50-60% of standard limits
├── Initial cap: ₹25K-50K
├── Graduation: Increase after 3 on-time payments
└── Enhanced monitoring: First 6 months

Expected Outcomes:
├── Approval rate: 30-40%
├── Bad rate: ~15% (vs 22.56% baseline)
└── Build credit history for future model training
```

### Business Impact
```
Overall Portfolio:
├── Approval Rate: 70-75%
├── Expected Bad Rate: 6-8%
├── Auto-Approval: 40-50%
└── Manual Review: 10-15%

Risk Mitigation:
├── High-risk rejected: 37%
├── True positives caught: 97 bad customers
└── Avoided losses: ~₹2.91 Cr (estimated)
```

## 📊 Results

### Model Outputs

#### 1. Risk Scores
- Continuous probability scores (0-1) for each customer
- Higher score = higher delinquency risk

#### 2. Approval Decisions
```
Distribution:
├── AUTO-APPROVE:     231 customers (11.6%)
├── MANUAL REVIEW:    1,335 customers (66.8%)
├── CONDITIONAL:      2 customers (0.1%)
└── REJECT:          432 customers (21.6%)
```

#### 3. Credit Limit Assignment
```
Portfolio Exposure: ₹8.63 Crores (test set)
├── Average Limit: ₹47,368
├── Median Limit: ₹35,000
└── Range: ₹5,000 - ₹170,000
```

#### 4. Segment Performance
| Decision | Count | Avg Risk Score | Actual Bad Rate | Avg Limit |
|----------|-------|----------------|-----------------|-----------|
| APPROVE | 231 | 0.143 | 3.0% | ₹96,466 |
| MANUAL REVIEW | 1,335 | 0.394 | 6.7% | ₹52,069 |
| REJECT | 432 | 0.721 | 22.5% | ₹0 |

</div>
