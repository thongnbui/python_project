# Retina AI Project

This repository contains a comprehensive data science project focused on retail analytics, sales analysis, and organizational structure analysis. The project demonstrates various data science techniques including data processing, statistical analysis, machine learning approaches, and visualization.

## 📁 Project Structure

```
retina_ai/
├── README.md                    # This file
├── Retina_ai.ipynb             # Main analysis notebook (955KB, 3539 lines)
├── databrick.ipynb             # Data visualization notebook (53KB, 375 lines)
├── databrick.csv               # Sample dataset for visualization
├── Retina_ai.pdf               # PDF documentation (567KB)
└── test/                       # Unit tests
    ├── TestStringMethods.py    # String method tests
    └── TestWidgets.py          # Widget functionality tests
```

## 🎯 Project Overview

This project contains multiple data science analyses covering:

### 1. **Data Gathering & SQL Analysis**
- SQL queries for visitor analytics
- Page view analysis
- Customer behavior tracking

### 2. **Sales Data Analysis**
- Weekly sales data processing from 2012-2013
- Statistical significance testing for sales changes
- Gender-based customer analysis
- Daypart analysis (night, morning, afternoon, evening)

### 3. **Machine Learning & Recommendation Engine**
- Product recommendation system design
- Vendor profitability analysis
- Transaction pattern analysis

### 4. **Organizational Analysis**
- Organizational chart visualization
- Manager-to-employee ratio analysis
- Maximum depth calculation
- Hierarchical structure mapping

## 📊 Key Findings

### Sales Analysis
- **Significant Sales Increase**: Detected a statistically significant increase in daily sales starting April 29, 2013
- **Gender Shift**: Male customers gradually increased from 1:4 ratio in 2012 to surpassing female customers by May 2013
- **Daypart Distribution**: Sales distribution across different dayparts (night, morning, afternoon, evening)

### Product Analysis
- **Top Performers**: Dahlia, Anthurium, and Dozen Roses were the most profitable products
- **Vendor Analysis**: Identified most and least profitable vendors
- **Profit Margins**: Calculated adjusted sales (total_sale_amount_usd - cost_of_sale)

### Organizational Insights
- **Management Structure**: Average of 3.53 employees per manager, median of 2
- **Hierarchy Depth**: Calculated maximum organizational depth
- **Visual Representation**: Generated organizational chart using pydot

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **SciPy**: Statistical analysis (t-tests)
- **Jupyter Notebooks**: Interactive analysis environment
- **Pydot**: Organizational chart generation
- **Unittest**: Testing framework

## 📈 Data Sources

The project uses multiple datasets:
- Weekly sales data from 2012-2013 (downloaded from S3)
- Flower shop transaction data
- Organizational employee data
- Sample visualization data (databrick.csv)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib scipy jupyter pydot
```

### Running the Analysis
1. Open `Retina_ai.ipynb` in Jupyter Notebook
2. Execute cells sequentially to reproduce the analysis
3. For visualization examples, see `databrick.ipynb`

### Running Tests
```bash
cd test
python TestStringMethods.py
python TestWidgets.py
```

## 📋 Analysis Sections

### Section 1: Data Gathering
- SQL queries for page analytics
- Visitor behavior analysis
- Customer segmentation

### Section 2: Data Processing/Analysis
- Sales trend analysis
- Statistical significance testing
- Customer demographic shifts
- Time-based sales patterns

### Section 3: Machine Learning
- Recommendation engine design
- Product profitability analysis
- Vendor performance evaluation

### Section 4: Organizational Analysis
- Hierarchical structure mapping
- Management ratio calculations
- Organizational depth analysis

## 📊 Visualizations

The project includes various visualizations:
- Time series plots of sales data
- Gender-based customer trends
- Product profitability charts
- Organizational hierarchy diagrams
- Stacked bar charts for activity analysis

## 🔍 Statistical Analysis

- **T-Test Results**: p-value ≈ 3.487e-138 (highly significant)
- **Confidence Intervals**: Used for sales change validation
- **Descriptive Statistics**: Mean, median, standard deviation analysis

## 📝 Notes

- The main analysis notebook (`Retina_ai.ipynb`) contains 3539 lines of comprehensive analysis
- All data downloads are automated within the notebooks
- Statistical significance is thoroughly tested and documented
- The project demonstrates both quantitative and qualitative analysis approaches

## 🤝 Contributing

This is a data science project showcasing various analytical techniques. Feel free to:
- Extend the analysis with additional datasets
- Improve visualizations
- Add more statistical tests
- Enhance the recommendation engine design

## 📄 License

This project is for educational and analytical purposes. Please ensure proper attribution when using any findings or methodologies.

---

*Last updated: Based on analysis of retail and organizational data from 2012-2013* 