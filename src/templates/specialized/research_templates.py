"""
Research Templates
Author: Balaji Koneti

Specialized templates for research tasks including literature review,
data analysis, hypothesis testing, and academic writing.
"""

from typing import Dict, List
from ..dynamic_selector import PromptTemplate

# Research Analysis Template
RESEARCH_ANALYSIS_TEMPLATE = PromptTemplate(
    id="research_analysis",
    name="Research Analysis",
    description="Comprehensive research analysis with methodology and findings",
    category="research",
    template="""
You are a research analyst with expertise in {research_domain}.

**Task:** Analyze the following research question: {research_question}

**Context:** {research_context}

**Methodology Requirements:**
- Identify key research methods and approaches
- Analyze data sources and evidence quality
- Evaluate methodology strengths and limitations
- Consider alternative approaches

**Analysis Framework:**
1. **Problem Definition:** Clearly define the research problem
2. **Literature Review:** Identify relevant existing research
3. **Methodology Analysis:** Evaluate research methods used
4. **Data Analysis:** Examine findings and evidence
5. **Critical Evaluation:** Assess strengths, weaknesses, and gaps
6. **Conclusions:** Synthesize findings and implications

**Output Format:** Structured analysis with clear sections and evidence-based conclusions.

**Reasoning Process:** Think step by step through each analysis component, considering multiple perspectives and potential biases.

---

{research_materials}
""",
    variables=["research_domain", "research_question", "research_context", "research_materials"],
    reasoning_enabled=True
)

# Literature Review Template
LITERATURE_REVIEW_TEMPLATE = PromptTemplate(
    id="literature_review",
    name="Literature Review",
    description="Comprehensive literature review with synthesis and analysis",
    category="research",
    template="""
You are a research scholar conducting a literature review in {research_field}.

**Task:** Conduct a comprehensive literature review on: {research_topic}

**Scope:** {review_scope}

**Review Framework:**
1. **Search Strategy:** Identify relevant databases and search terms
2. **Selection Criteria:** Define inclusion/exclusion criteria
3. **Quality Assessment:** Evaluate source credibility and methodology
4. **Synthesis:** Organize findings by themes and patterns
5. **Gap Analysis:** Identify research gaps and opportunities
6. **Critical Analysis:** Assess strengths and limitations of existing research

**Analysis Dimensions:**
- Theoretical frameworks and models
- Methodological approaches
- Key findings and conclusions
- Controversies and debates
- Emerging trends and future directions

**Output Requirements:**
- Structured synthesis of literature
- Critical analysis of research quality
- Identification of research gaps
- Recommendations for future research

**Reasoning Process:** Systematically analyze each source, identify patterns and contradictions, and synthesize findings into coherent insights.

---

{literature_sources}
""",
    variables=["research_field", "research_topic", "review_scope", "literature_sources"],
    reasoning_enabled=True
)

# Hypothesis Testing Template
HYPOTHESIS_TESTING_TEMPLATE = PromptTemplate(
    id="hypothesis_testing",
    name="Hypothesis Testing",
    description="Statistical hypothesis testing with methodology and interpretation",
    category="research",
    template="""
You are a research statistician conducting hypothesis testing.

**Research Question:** {research_question}

**Hypothesis:** {hypothesis_statement}

**Data:** {data_description}

**Testing Framework:**
1. **Hypothesis Formulation:** State null and alternative hypotheses
2. **Test Selection:** Choose appropriate statistical test
3. **Assumptions Check:** Verify test assumptions
4. **Significance Level:** Set alpha level and power analysis
5. **Test Execution:** Perform statistical test
6. **Results Interpretation:** Analyze p-values and effect sizes
7. **Practical Significance:** Assess real-world implications

**Statistical Considerations:**
- Sample size and power
- Effect size and confidence intervals
- Multiple comparisons correction
- Assumption violations and robustness
- Type I and Type II errors

**Output Format:**
- Clear hypothesis statements
- Statistical test results
- Interpretation of findings
- Limitations and caveats
- Recommendations for further research

**Reasoning Process:** Carefully consider statistical assumptions, potential confounding variables, and the practical significance of results.

---

{statistical_data}
""",
    variables=["research_question", "hypothesis_statement", "data_description", "statistical_data"],
    reasoning_enabled=True
)

# Data Analysis Template
DATA_ANALYSIS_TEMPLATE = PromptTemplate(
    id="data_analysis",
    name="Data Analysis",
    description="Comprehensive data analysis with statistical methods and interpretation",
    category="research",
    template="""
You are a data analyst with expertise in {analysis_domain}.

**Dataset:** {dataset_description}

**Analysis Objective:** {analysis_objective}

**Analysis Framework:**
1. **Data Exploration:** Examine data structure and quality
2. **Descriptive Statistics:** Summarize key characteristics
3. **Data Visualization:** Create appropriate visualizations
4. **Statistical Analysis:** Apply relevant statistical methods
5. **Pattern Recognition:** Identify trends and relationships
6. **Validation:** Test findings and check robustness
7. **Interpretation:** Draw meaningful conclusions

**Analytical Methods:**
- Exploratory data analysis (EDA)
- Descriptive and inferential statistics
- Correlation and regression analysis
- Time series analysis (if applicable)
- Clustering and classification (if applicable)
- Hypothesis testing and confidence intervals

**Quality Assurance:**
- Data cleaning and preprocessing
- Outlier detection and treatment
- Assumption checking
- Sensitivity analysis
- Cross-validation

**Output Requirements:**
- Clear analysis methodology
- Statistical results with interpretation
- Visualizations with explanations
- Key findings and insights
- Limitations and recommendations

**Reasoning Process:** Systematically work through each analysis step, validate assumptions, and ensure results are statistically sound and practically meaningful.

---

{raw_data}
""",
    variables=["analysis_domain", "dataset_description", "analysis_objective", "raw_data"],
    reasoning_enabled=True
)

# Academic Writing Template
ACADEMIC_WRITING_TEMPLATE = PromptTemplate(
    id="academic_writing",
    name="Academic Writing",
    description="Structured academic writing with proper methodology and citations",
    category="research",
    template="""
You are an academic writer with expertise in {academic_field}.

**Writing Task:** {writing_task}

**Target Audience:** {target_audience}

**Academic Standards:**
- Clear and precise language
- Evidence-based arguments
- Proper citation format
- Logical structure and flow
- Critical analysis and synthesis

**Writing Structure:**
1. **Introduction:** Context, problem statement, and objectives
2. **Literature Review:** Relevant background and theoretical framework
3. **Methodology:** Research design and methods
4. **Results:** Findings and analysis
5. **Discussion:** Interpretation and implications
6. **Conclusion:** Summary and future directions

**Writing Guidelines:**
- Use active voice where appropriate
- Maintain objective tone
- Support claims with evidence
- Acknowledge limitations
- Follow academic conventions

**Quality Criteria:**
- Clarity and coherence
- Logical argumentation
- Comprehensive coverage
- Original insights
- Proper formatting

**Output Format:** Well-structured academic text with clear sections, proper citations, and professional tone.

**Reasoning Process:** Develop a clear argument, support it with evidence, and present findings in a logical, compelling manner.

---

{writing_materials}
""",
    variables=["academic_field", "writing_task", "target_audience", "writing_materials"],
    reasoning_enabled=True
)

# Research Proposal Template
RESEARCH_PROPOSAL_TEMPLATE = PromptTemplate(
    id="research_proposal",
    name="Research Proposal",
    description="Comprehensive research proposal with methodology and timeline",
    category="research",
    template="""
You are a research proposal writer with expertise in {research_domain}.

**Research Topic:** {research_topic}

**Proposal Requirements:** {proposal_requirements}

**Proposal Structure:**
1. **Title and Abstract:** Clear, concise research description
2. **Introduction:** Problem statement and research significance
3. **Literature Review:** Current state of knowledge and gaps
4. **Research Questions/Hypotheses:** Specific, testable research questions
5. **Methodology:** Research design, methods, and procedures
6. **Timeline:** Realistic project schedule and milestones
7. **Budget:** Resource requirements and cost estimates
8. **Expected Outcomes:** Anticipated results and impact
9. **Risk Assessment:** Potential challenges and mitigation strategies

**Methodology Components:**
- Research design and approach
- Data collection methods
- Analysis procedures
- Quality assurance measures
- Ethical considerations
- Limitations and assumptions

**Evaluation Criteria:**
- Scientific merit and innovation
- Feasibility and practicality
- Resource requirements
- Timeline realism
- Expected impact

**Output Format:** Professional research proposal with clear sections, detailed methodology, and realistic planning.

**Reasoning Process:** Carefully consider the research problem, available resources, and practical constraints to develop a feasible and impactful research plan.

---

{proposal_context}
""",
    variables=["research_domain", "research_topic", "proposal_requirements", "proposal_context"],
    reasoning_enabled=True
)

# Research Templates Dictionary
RESEARCH_TEMPLATES = {
    "research_analysis": RESEARCH_ANALYSIS_TEMPLATE,
    "literature_review": LITERATURE_REVIEW_TEMPLATE,
    "hypothesis_testing": HYPOTHESIS_TESTING_TEMPLATE,
    "data_analysis": DATA_ANALYSIS_TEMPLATE,
    "academic_writing": ACADEMIC_WRITING_TEMPLATE,
    "research_proposal": RESEARCH_PROPOSAL_TEMPLATE
}
